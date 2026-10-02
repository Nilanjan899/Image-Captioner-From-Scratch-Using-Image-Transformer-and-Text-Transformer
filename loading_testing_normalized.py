r"""
load_alpha_beta_models.py

Loads and tests both captioning models:
  - Alpha: YOLO11n (frozen) -> LDPv2 -> Flan-T5-small (+LoRA)
  - Beta : MobileCLIP-S0 (frozen) -> PixelShuffle -> SmolLM2-135M (+LoRA)

Fixes vs previous version (marked with # FIX):
  1. BetaCaptioner.encode_visual casts connector output to decoder dtype
  2. load_beta_model picks amp_dtype by device (fp32 on CPU, bf16 on CUDA)
  3. torch_dtype -> dtype (deprecation warning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from timm.data import create_transform, resolve_model_data_config

ALPHA_CHECKPOINT_PATH = r"alpha\alpha_final_trainable\alpha_final_trainable_fixed.pt"
BETA_CHECKPOINT_PATH = r"beta\stage2_best\stage2_best_fixed.pt"
TEST_IMAGE_PATH = r"images\cateshwar.jpeg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# PART 1 - ALPHA
# ======================================================================

class LDPv2(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.pw1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.peg = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        x = self.pool(x)
        x = x + self.peg(x)
        tokens = x.flatten(2).transpose(1, 2)
        return self.norm(tokens)


class YOLOSPPFFeatures(nn.Module):
    def __init__(self, detection_model, sppf_module):
        super().__init__()
        self.detection_model = detection_model
        self._feature = None
        self._hook = sppf_module.register_forward_hook(self._capture)

    class _StopAtSPPF(Exception):
        pass

    def _capture(self, module, inputs, output):
        self._feature = output
        raise self._StopAtSPPF()

    def forward(self, x):
        self._feature = None
        try:
            with torch.no_grad():
                _ = self.detection_model(x)
        except self._StopAtSPPF:
            pass
        if self._feature is None:
            raise RuntimeError("SPPF hook did not capture a feature tensor.")
        return self._feature.detach()

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


class AlphaCaptioner(nn.Module):
    def __init__(self, vision_backbone, text_model, vision_hidden, t5_hidden):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.connector = LDPv2(in_channels=vision_hidden, hidden_dim=t5_hidden, out_dim=t5_hidden)
        self.text_model = text_model

    def _encode_visual_tokens(self, pixel_values):
        if pixel_values.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                visual_map = self.vision_backbone(pixel_values)
        else:
            visual_map = self.vision_backbone(pixel_values)
        return self.connector(visual_map.float())

    def _encoder_inputs(self, pixel_values, input_ids, attention_mask):
        visual_tokens = self._encode_visual_tokens(pixel_values).float()
        text_embeddings = self.text_model.get_input_embeddings()(input_ids).float()
        inputs_embeds = torch.cat([visual_tokens, text_embeddings], dim=1)
        visual_attention_mask = torch.ones(
            (input_ids.shape[0], visual_tokens.shape[1]),
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        encoder_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)
        return inputs_embeds, encoder_attention_mask

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        inputs_embeds, encoder_attention_mask = self._encoder_inputs(pixel_values, input_ids, attention_mask)
        return self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=False,
        )

    @torch.no_grad()
    def generate_caption(self, pixel_values, input_ids, attention_mask, max_new_tokens=32, num_beams=4):
        from transformers.modeling_outputs import BaseModelOutput

        was_training = self.training
        self.eval()

        inputs_embeds, encoder_attention_mask = self._encoder_inputs(pixel_values, input_ids, attention_mask)

        encoder_outputs = self.text_model.get_encoder()(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        generated_ids = self.text_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state),
            attention_mask=encoder_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=1.0,
            repetition_penalty=1.15,
            do_sample=False,
            use_cache=True,
        )

        if was_training:
            self.train()
        return generated_ids


def load_alpha_model(checkpoint_path, device=DEVICE):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from peft import LoraConfig, TaskType, get_peft_model
    from ultralytics import YOLO

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    yolo_weights = ckpt.get("vision_weights", "yolo11n.pt")
    text_model_name = ckpt.get("text_model", "google/flan-t5-small")
    image_size = ckpt.get("image_size", 512)

    lora_cfg = ckpt.get("lora_config", {})
    lora_r = lora_cfg.get("r", 16)
    lora_alpha = lora_cfg.get("alpha", 32)
    lora_dropout = lora_cfg.get("dropout", 0.05)
    lora_targets = lora_cfg.get("target_modules", ["q", "k", "v", "o", "wi_0", "wi_1", "wo"])

    # YOLO
    yolo_wrapper = YOLO(yolo_weights)
    yolo_detection_model = yolo_wrapper.model
    yolo_detection_model.eval()
    for p in yolo_detection_model.parameters():
        p.requires_grad = False

    sppf_modules = [
        (name, m) for name, m in yolo_detection_model.named_modules()
        if m.__class__.__name__.lower() == "sppf"
    ]
    if not sppf_modules:
        raise RuntimeError("Could not find an SPPF module in the loaded YOLO model.")
    _, sppf_module = sppf_modules[-1]

    vision_backbone = YOLOSPPFFeatures(yolo_detection_model, sppf_module).to(device)
    vision_backbone.eval()

    # Flan-T5
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = T5ForConditionalGeneration.from_pretrained(text_model_name).to(device)
    for p in text_model.parameters():
        p.requires_grad = False
    text_model.config.use_cache = False
    t5_hidden = text_model.config.d_model

    # Probe SPPF
    with torch.no_grad():
        probe = torch.zeros((1, 3, image_size, image_size), device=device, dtype=torch.float32)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probe_features = vision_backbone(probe)
        else:
            probe_features = vision_backbone(probe)
    sppf_channels = int(probe_features.shape[1])

    # Assemble
    model = AlphaCaptioner(
        vision_backbone=vision_backbone,
        text_model=text_model,
        vision_hidden=sppf_channels,
        t5_hidden=t5_hidden,
    ).to(device)

    # LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_targets,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model.text_model = get_peft_model(model.text_model, lora_config)

    # Load trained weights
    missing, unexpected = model.load_state_dict(ckpt["trainable_state"], strict=False)
    if unexpected:
        print(f"[alpha] WARNING - unexpected keys in checkpoint: {unexpected}")

    model.eval()

    print(f"[alpha] Loaded checkpoint: {checkpoint_path}")
    print(
        f"[alpha] YOLO weights: {yolo_weights} | "
        f"Text model: {text_model_name} | "
        f"SPPF channels: {sppf_channels}"
    )
    return model, tokenizer


# ======================================================================
# PART 2 - BETA
# ======================================================================

class BetaPixelShuffleConnector(nn.Module):
    def __init__(self, in_channels, llm_hidden, downscale=2, dropout=0.0):
        super().__init__()
        self.downscale = downscale
        self.unshuffle = nn.PixelUnshuffle(downscale)
        self.proj = nn.Linear(in_channels * (downscale ** 2), llm_hidden)
        self.norm = nn.LayerNorm(llm_hidden)
        self.dropout = nn.Dropout(dropout)

    def set_dropout(self, probability):
        self.dropout.p = float(probability)

    def forward(self, x):
        assert x.ndim == 4, f"Expected NCHW tensor, got {x.shape}"
        x = self.unshuffle(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        x = self.dropout(x)
        x = self.norm(x)
        return x


class BetaCaptioner(nn.Module):
    def __init__(self, vision_encoder, feature_index, connector, text_decoder, tokenizer):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.feature_index = feature_index
        self.connector = connector
        self.text_decoder = text_decoder
        self.tokenizer = tokenizer

    def encode_visual(self, pixel_values):
        self.vision_encoder.eval()
        with torch.no_grad():
            features = self.vision_encoder(pixel_values)
            fmap = features[self.feature_index]
        tokens = self.connector(fmap)
        # FIX 1: connector is fp32, decoder may be bf16 -> match decoder dtype
        return tokens.to(self.text_decoder.get_input_embeddings().weight.dtype)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        visual_tokens = self.encode_visual(pixel_values)
        text_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        image_mask = torch.ones(
            pixel_values.size(0), visual_tokens.size(1),
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        full_attention = torch.cat([image_mask, attention_mask], dim=1)

        outputs = self.text_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            use_cache=False,
        )

        if labels is None:
            return outputs

        prefix_ignored = torch.full(
            (labels.size(0), visual_tokens.size(1)), -100,
            dtype=labels.dtype, device=labels.device,
        )
        full_labels = torch.cat([prefix_ignored, labels], dim=1)

        shift_logits = outputs.logits[:, :-1, :].contiguous().to(torch.float32)
        shift_labels = full_labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss, outputs

    @torch.no_grad()
    def generate_caption(self, pixel_values, prompt, max_new_tokens=30):
        self.vision_encoder.eval()
        self.text_decoder.eval()

        visual_tokens = self.encode_visual(pixel_values)

        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("Tokenizer does not expose a BOS token.")

        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(pixel_values.device)

        bos = torch.full(
            (prompt_ids.size(0), 1), bos_id,
            dtype=prompt_ids.dtype, device=pixel_values.device,
        )
        prompt_ids = torch.cat([bos, prompt_ids], dim=1)

        prompt_embeds = self.text_decoder.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([visual_tokens, prompt_embeds], dim=1)

        attention = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=pixel_values.device)

        out = self.text_decoder(inputs_embeds=inputs_embeds, attention_mask=attention, use_cache=True)
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            next_embed = self.text_decoder.get_input_embeddings()(next_token)
            attention = torch.cat(
                [attention, torch.ones((attention.size(0), 1), dtype=attention.dtype, device=attention.device)],
                dim=1,
            )
            out = self.text_decoder(
                inputs_embeds=next_embed,
                attention_mask=attention,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)

            if torch.all(next_token.squeeze(-1) == self.tokenizer.eos_token_id):
                break

        generated_ids = torch.cat(generated, dim=1)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def load_beta_model(checkpoint_path, device=DEVICE):
    import timm
    from timm.data import resolve_model_data_config
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_info = ckpt.get("model", {})
    vision_model_name = model_info.get("vision", "hf_hub:apple/mobileclip_s0_timm")
    text_model_name = model_info.get("text", "HuggingFaceTB/SmolLM2-135M")
    feature_index = model_info.get("feature_index")
    pixel_unshuffle_factor = 2
    lora_targets = model_info.get(
        "lora_targets",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # FIX 2: bf16 only on CUDA; fp32 on CPU
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # MobileCLIP
    vision_encoder = timm.create_model(vision_model_name, pretrained=True, features_only=True)
    vision_encoder.eval()
    vision_encoder.requires_grad_(False)
    vision_data_config = resolve_model_data_config(vision_encoder)

    image_size = 256
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size)
        feature_pyramid = vision_encoder(dummy)

    if feature_index is None:
        target_grid = 16
        candidate_indices = [
            i for i, f in enumerate(feature_pyramid)
            if f.ndim == 4 and f.shape[-2] == target_grid and f.shape[-1] == target_grid
        ]
        feature_index = candidate_indices[-1]

    vision_channels = model_info.get("vision_channels", int(feature_pyramid[feature_index].shape[1]))

    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()

    # SmolLM2
    tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_decoder = AutoModelForCausalLM.from_pretrained(
        text_model_name,
        low_cpu_mem_usage=True,
        dtype=amp_dtype,  # FIX 3: was torch_dtype (deprecated)
    )
    text_decoder.config.use_cache = True
    text_decoder.to(device)
    text_hidden = int(text_decoder.config.hidden_size)

    # Connector
    connector = BetaPixelShuffleConnector(
        in_channels=vision_channels,
        llm_hidden=text_hidden,
        downscale=pixel_unshuffle_factor,
        dropout=0.0,
    ).to(device)

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.10,
        target_modules=lora_targets,
        bias="none",
    )
    text_decoder = get_peft_model(text_decoder, lora_config)

    # Assemble
    model = BetaCaptioner(
        vision_encoder=vision_encoder,
        feature_index=feature_index,
        connector=connector,
        text_decoder=text_decoder,
        tokenizer=tokenizer,
    ).to(device)

    model.vision_encoder.requires_grad_(False)
    model.vision_encoder.eval()

    # Load trained weights
    model.connector.load_state_dict(ckpt["connector_state"])
    if ckpt.get("lora_state"):
        set_peft_model_state_dict(model.text_decoder, ckpt["lora_state"])

    model.eval()

    best_val = ckpt.get("best_val_loss")
    print(f"[beta] Loaded checkpoint: {checkpoint_path}")
    print(
        f"[beta] Vision: {vision_model_name} | "
        f"Text: {text_model_name} | "
        f"feature_index={feature_index} | "
        f"val_loss={best_val} | dtype={amp_dtype}"
    )
    return model, tokenizer


# ======================================================================
# IMAGE TESTING
# ======================================================================

def letterbox_pil(image: Image.Image, size: int = 512, fill: int = 114) -> torch.Tensor:
    image = image.convert("RGB")
    w, h = image.size
    scale = min(size / max(w, 1), size / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    image = image.resize((nw, nh), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(image, (left, top))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor

def test_models_on_image(alpha_model, alpha_tokenizer, beta_model, beta_tokenizer, image_path):
    print("\n" + "=" * 70)
    print("IMAGE CAPTIONING TEST")
    print("=" * 70)
    
    image = Image.open(image_path).convert("RGB")

    # ---------------- ALPHA ----------------
    print("\nRunning Blueprint Alpha...")
    
    # FIX: Use the exact letterbox transform from training (No ImageNet Norm!)
    alpha_pixel_values = letterbox_pil(image, size=512).unsqueeze(0).to(DEVICE)

    alpha_tokens = alpha_tokenizer("Describe this image.", return_tensors="pt", padding=True)
    alpha_input_ids = alpha_tokens["input_ids"].to(DEVICE)
    alpha_attention_mask = alpha_tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        alpha_output_ids = alpha_model.generate_caption(
            alpha_pixel_values, alpha_input_ids, alpha_attention_mask,
            max_new_tokens=32, num_beams=4,
        )
    alpha_caption = alpha_tokenizer.decode(alpha_output_ids[0], skip_special_tokens=True)
    print(f"ALPHA CAPTION:\n{alpha_caption}")

    # ---------------- BETA ----------------
    print("\nRunning Blueprint Beta...")
    
    # FIX: Dynamically pull the exact MobileCLIP transform configuration using TIMM
    vision_data_config = resolve_model_data_config(beta_model.vision_encoder)
    beta_transform = create_transform(**vision_data_config, is_training=False)
    beta_pixel_values = beta_transform(image).unsqueeze(0).to(DEVICE)

    beta_prompt = "Describe this image in one sentence:"

    with torch.no_grad():
        beta_captions = beta_model.generate_caption(beta_pixel_values, beta_prompt, max_new_tokens=30)
    beta_caption = beta_captions[0]
    print(f"BETA CAPTION:\n{beta_caption}")

# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}\n")

    print("Loading Blueprint Alpha ...")
    alpha_model, alpha_tokenizer = load_alpha_model(ALPHA_CHECKPOINT_PATH, device=DEVICE)
    print("Alpha ready.\n")

    print("Loading Blueprint Beta ...")
    beta_model, beta_tokenizer = load_beta_model(BETA_CHECKPOINT_PATH, device=DEVICE)
    print("Beta ready.\n")

    print("Both models loaded successfully.")

    test_models_on_image(
        alpha_model=alpha_model,
        alpha_tokenizer=alpha_tokenizer,
        beta_model=beta_model,
        beta_tokenizer=beta_tokenizer,
        image_path=TEST_IMAGE_PATH,
    )