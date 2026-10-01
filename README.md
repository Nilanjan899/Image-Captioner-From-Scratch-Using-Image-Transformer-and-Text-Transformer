# COCO Image Captioning — Blueprint Alpha & Blueprint Beta

Two multimodal image-captioning implementations are presented here, both trained on the COCO 2017 captioning data but using substantially different visual-language bridges and language backbones.

### Notebooks

- **[Blueprint Alpha — YOLO11n + LDPv2 + Flan-T5-small Image Captioner](https://www.kaggle.com/code/nilanjansaha123/yolo11n-ldpv2-flan-t5-small-image-captioner5)**  
  A compact CNN-to-encoder/decoder design: frozen YOLO11n spatial features are compressed by LDPv2 into 64 visual tokens and consumed by Flan-T5-small.

- **[Blueprint Beta — MobileCLIP-S0 + PixelShuffle Pooling + SmolLM2-135M](https://www.kaggle.com/code/nilanjansaha123hello/mobileclip-s0-pixelshuffle-pooling-smollm2-135-2)**  
  A lightweight CLIP-style visual encoder is converted into a 64-token visual prefix for SmolLM2-135M, with staged connector training followed by LoRA-based language adaptation.

The detailed architecture, optimization strategy, hardware decisions, training curves, and qualitative results are documented below.

---

## 1. Problem and Dataset

Both notebooks perform **image captioning**: an image is encoded into a compact visual representation and a language model generates a natural-language caption.

The experiments use the **COCO 2017 captioning dataset** from the Kaggle dataset path used in the notebooks:

```text
/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/
```

The Alpha notebook reports:

- 591,753 caption records from the COCO train2017 annotations.
- 118,287 unique train2017 images available for the image-level split.
- An 80/20 image-level split:
  - 94,629 training images
  - 23,658 validation images
- 5,000 COCO val2017 images reserved for qualitative testing.

A key data-throughput optimization in Alpha is to treat each **unique image as one training example per epoch**, while rotating through its multiple COCO reference captions between epochs. This avoids repeatedly running the frozen visual encoder on the same image multiple times inside one epoch.

---

# 2. Blueprint Alpha

## 2.1 Architecture

The Alpha pipeline is:

```text
COCO image
    │
    ▼
512×512 letterboxed image
    │
    ▼
Frozen YOLO11n
    │
    ├── SPPF feature map: 256 × 16 × 16
    │
    ▼
LDPv2 connector
    │
    ├── 1×1 pointwise projection
    ├── GELU
    ├── 1×1 pointwise projection
    ├── 2×2 average pooling
    └── depthwise PEG residual
    │
    ▼
64 × 512 visual tokens
    │
    ├───────────────┐
    ▼               ▼
Visual tokens    Text prompt
    │               │
    └───────┬───────┘
            ▼
     Flan-T5-small encoder
            │
            ▼
     Autoregressive decoder
            │
            ▼
      Generated caption
```

### Why 64 visual tokens?

With a 512×512 input and the YOLO stride-32 SPPF feature stage:

\[
512 / 32 = 16
\]

so the spatial feature map is:

\[
256 \times 16 \times 16
\]

The LDPv2 2×2 average-pooling operation changes the spatial grid from:

\[
16\times16 \rightarrow 8\times8
\]

which gives:

\[
8\times8 = 64
\]

visual tokens.

The notebook explicitly verifies this geometry and reports an LDPv2 output of:

```text
(1, 64, 512)
```

The 512-dimensional token width matches Flan-T5-small's hidden dimension.

---

## 2.2 LDPv2

The connector follows the Alpha formulation:

\[
f_0 = PW_2(GELU(PW_1(f_v)))
\]

\[
f_1 = AvgPool_{2\times2}(f_0)
\]

\[
H_v = DW(f_1) + f_1
\]

where:

- `PW` = 1×1 pointwise convolution
- `DW` = depthwise convolution
- average pooling performs the spatial token reduction
- PEG provides a lightweight spatial residual

The concrete implementation contains approximately **0.400M trainable connector parameters**, including LayerNorm.

The connector is deliberately initialized conservatively so that its initially random visual representation does not overwhelm the pretrained T5 embedding space.

---

## 2.3 Language model integration

Unlike a causal decoder receiving a long visual prefix directly, Alpha feeds the visual representation into the **encoder side** of Flan-T5:

```text
[64 visual embeddings] + [text prompt embeddings]
                    │
                    ▼
             T5 encoder
                    │
                    ▼
             T5 decoder
                    │
                    ▼
              caption tokens
```

This keeps the visual information on the bidirectional encoder side while the decoder remains responsible for autoregressive caption generation.

The notebook uses prompts such as:

```text
Describe this image in one concise caption.
What is shown in this image?
Write a natural caption for this image.
Describe the main objects and scene visible here.
```

---

## 2.4 Training strategy

Alpha uses two stages.

### Stage 1 — cross-modal alignment

- 2 epochs
- YOLO11n frozen
- Flan-T5 base frozen
- LDPv2 connector trainable
- Learning rate: `2e-4`

The purpose is to learn a usable mapping from YOLO's spatial feature space into the language model's representation space without simultaneously changing the language model.

### Stage 2 — visual instruction/caption adaptation

- 10 epochs
- YOLO11n remains frozen
- LDPv2 remains trainable
- Flan-T5 is adapted using LoRA
- Connector learning rate: `1e-4`
- LoRA learning rate: `1.5e-4`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- Weight decay: `0.01`
- Warmup followed by cosine decay
- Gradient clipping at `1.0`

The executed notebook reports approximately:

```text
Total Flan-T5 parameters: 76.96M
Total multimodal parameters after LoRA: 79.52M
Trainable parameters after LoRA: 2.56M
Trainable percentage: 3.21%
```

---

## 2.5 T4 optimization

The notebook was explicitly designed for a **single Kaggle Tesla T4**, even when two T4s are available.

Runtime detection reported:

```text
GPU: Tesla T4
VRAM: 14.56 GiB
Visible GPUs: 1
```

Instead of selecting a batch size from theory alone, the notebook benchmarks real forward/backward passes on the target GPU.

| Batch | Images/s | Peak VRAM |
|---:|---:|---:|
| 16 | 8.51 | 1.72 GiB |
| 24 | 24.55 | 2.38 GiB |
| 32 | 30.76 | 3.04 GiB |
| 48 | 39.42 | 4.36 GiB |
| **64** | **44.50** | **5.69 GiB** |

The selected physical batch size was therefore **64**, with no gradient accumulation.

The training path uses FP16 autocasting for the frozen YOLO feature extraction while keeping the T5 computation in FP32 for numerical stability.

---

## 2.6 Alpha training results

The final executed Alpha run reports the following late-stage metrics:

| Stage | Epoch | Train loss | Validation loss |
|---|---:|---:|---:|
| Stage 2 | 9 | 2.2272 | 2.0811 |
| Stage 2 | **10** | **2.2193** | **2.0804** |

The best reported Stage-2 validation loss is therefore **2.0804**, reached at epoch 10.

The last Stage-2 epoch took approximately **16m 37s**.

The compact final trainable checkpoint was reported as:

```text
alpha_final_trainable.pt
Size: 11.38 MB
```

Frozen YOLO and base Flan-T5 weights are intentionally not duplicated in the checkpoint.

---

## 2.7 Alpha training curve

![Blueprint Alpha training curve](assets/alpha_cell50_img00.png)

The notebook's training-curve cell plots train and validation cross-entropy across the complete two-stage run, with the Stage-1/Stage-2 transition marked.

---

## 2.8 Alpha qualitative predictions

The notebook saves a six-image qualitative panel after every epoch.

One of the final Stage-2 panels is shown below:

![Blueprint Alpha qualitative results](assets/alpha_cell46_img80.png)

The final reported examples included:

| Ground truth | Prediction |
|---|---|
| A church with a large steeple and a clock mounted to it. | A clock tower with a clock on top of it. |
| A dinner plate with a green vegetable and a chicken rice meal. | A plate of food with broccoli and broccoli on it. |
| A baseball player is at bat at a baseball game | A baseball player swinging a bat at a ball. |
| A man skiing down a snow covered ski slope with two ski poles. | A man skiing down a snow covered slope. |
| A boat sitting on top of a body of water next to tall buildings. | A group of people walking down a city street. |
| A man wearing a life jacket while engaging in a water sport. | A man riding a wave on a surfboard in the water. |

These examples show both successful semantic captioning and clear failure cases. In particular, the boat/city example demonstrates that a decreasing validation loss does not imply that every generated caption is grounded correctly in the image.

---

# 3. Blueprint Beta

## 3.1 Architecture

The Beta pipeline replaces YOLO + T5 with a CLIP-family visual encoder and a small causal language model:

```text
COCO image
    │
    ▼
MobileCLIP-S0
    │
    ▼
Spatial visual feature map
    │
    ▼
PixelShuffle / space-to-depth token reduction
    │
    ▼
64 visual tokens
    │
    ├───────────────┐
    ▼               ▼
Visual prefix    BOS + prompt
    │               │
    └───────┬───────┘
            ▼
       SmolLM2-135M
            │
            ▼
       Caption tokens
```

The Beta notebook describes this as:

**MobileCLIP-family vision encoder → PixelShuffle/space-to-depth token reduction → SmolLM2-135M causal decoder.**

The target visual representation is again **64 tokens**, keeping the multimodal context compact.

---

## 3.2 Token reduction

Beta starts from a 16×16 spatial visual feature grid and reduces the spatial resolution by a factor of two in each direction:

\[
16\times16 \rightarrow 8\times8
\]

giving:

\[
8\times8=64
\]

visual tokens.

The purpose is to keep the image representation small enough that a 135M-parameter causal language model does not spend most of its context on visual tokens.

---

## 3.3 Language-model conditioning

Beta uses an explicit:

```text
[visual prefix] + [BOS] + [prompt] + [caption] + [EOS]
```

sequence.

The notebook also extends the generation attention mask at each generated token, avoiding repeated transformer attention-mask warnings while retaining the intended causal context.

---

## 3.4 Training strategy

Beta uses a shorter five-epoch schedule.

### Stage 1 — cross-modal alignment

- 3 epochs
- MobileCLIP-S0 frozen
- SmolLM2 base + LoRA frozen
- Connector trainable

### Stage 2 — caption adaptation

- Maximum 2 epochs
- MobileCLIP-S0 frozen
- Connector + LoRA trainable
- Stronger augmentation
- Higher weight decay
- Dropout
- Earlier/steeper cosine learning-rate decay
- Validation-loss early stopping with patience 1
- Low-rank adapter also applied to `lm_head`; pretrained head weights remain frozen

The notebook reports:

```text
Total multimodal parameters: 148.28M
Stage-1 trainable parameters: 0.59M
Stage-2 trainable parameters: 3.03M
```

The final compact checkpoint was:

```text
beta_final.pt
Size: 11.71 MB
```

---

## 3.5 Beta batch-size benchmark

Beta benchmarks physical batches on the actual Kaggle GPU rather than assuming a theoretical maximum.

The executed benchmark reported:

| Batch | Fits? | Peak VRAM | Images/s |
|---:|:---:|---:|---:|
| 48 | Yes | 8.28 GiB | 19.38 |
| **64** | **Yes** | **10.93 GiB** | **24.30** |
| 80 | No | 13.58 GiB | 23.83 |

The selected batch size was therefore **64**.

The notebook also scales learning rates relative to a reference batch size of 48.

---

## 3.6 Beta training results

The complete five-epoch run produced:

| Global epoch | Stage | Train loss | Validation loss | Epoch time |
|---:|---|---:|---:|---:|
| 1 | Stage 1 | 2.9554 | 2.7371 | 71m 00s |
| 2 | Stage 1 | 2.7268 | 2.6601 | 70m 54s |
| 3 | Stage 1 | 2.6853 | **2.6419** | 71m 02s |
| 4 | Stage 2 | 2.5219 | 2.3054 | 81m 40s |
| **5** | **Stage 2** | **2.3736** | **2.2846** | **81m 39s** |

The final Stage-2 epoch reached the lowest reported validation loss:

\[
\boxed{2.2846}
\]

The Stage-2 transition produced a substantial drop in validation loss from `2.6419` to `2.3054` in its first epoch.

---

## 3.7 Beta training curve

![Blueprint Beta training curve](assets/beta_cell38_img02.png)

The plot is generated directly from the notebook's saved epoch metrics.

---

## 3.8 Beta qualitative predictions

Beta generates a fixed six-image qualitative panel after every training epoch and also performs a final qualitative test on unseen `val2017` images.

### Training-time qualitative panel

![Blueprint Beta final training qualitative panel](assets/beta_cell34_img19.png)

The final Stage-2 examples included:

| Ground truth | Prediction |
|---|---|
| A brown and white dog is laying on the beach. | A brown and white dog laying on the ground. |
| A man cleaning a surf board on top of a deck. | A man in a suit and tie is sitting on a bench. |
| There are three giraffes walking in the wild together | A group of giraffes standing in a field. |
| a group of people flying kites at the beach. | A group of people flying kites in the ocean. |
| A yellow bus on street next to a building. | A bus is parked on the side of the road. |
| a yellow and red trains engine on its track and trees and signs | A train is traveling down a train track. |

### Unseen COCO val2017 test

The final notebook also evaluates six unseen validation images qualitatively:

![Blueprint Beta unseen COCO val2017 test](assets/beta_cell40_img01.png)

Examples reported by the notebook include:

| Ground truth | Prediction |
|---|---|
| A person walking down a street while holding an umbrella. | A man walking down a street with a umbrella. |
| A woman wearing glasses looking at slices of pizza | A woman is sitting at a table with a pizza. |
| People sitting on benches in the park, and traffic in the street | A man and woman sitting on a bench in a park. |
| A hand is holding a carrot for a llama to chew. | A dog is holding a red frisbee in its mouth. |
| A shirtless man doing skateboarding tricks near a stream | A man riding a skateboard on a sidewalk. |
| a polar bear near rocks made to look like ice | A large bear is standing in the water. |

The unseen examples again illustrate the difference between **linguistically plausible captions** and captions that are actually grounded in the input image.

---

# 4. Alpha vs. Beta — Architectural Comparison

| Component | Blueprint Alpha | Blueprint Beta |
|---|---|---|
| Vision encoder | YOLO11n | MobileCLIP-S0 |
| Vision encoder training | Frozen | Frozen |
| Visual connector | LDPv2 | PixelShuffle/space-to-depth based reduction |
| Visual tokens | 64 | 64 |
| Language model | Flan-T5-small | SmolLM2-135M |
| LM type | Encoder-decoder | Causal decoder |
| Stage 1 | Connector-only | Connector-only |
| Stage 2 | Connector + LoRA | Connector + LoRA |
| Stage-2 `lm_head` adapter | Not described as separate head adapter | Yes |
| Selected batch | 64 | 64 |
| Final reported validation loss | 2.0804 | 2.2846 |
| Training epochs | 12 total | 5 total |
| Compact checkpoint | 11.38 MB | 11.71 MB |

The two experiments should **not** be interpreted as a controlled scientific comparison solely from these numbers. They use different architectures, optimization schedules, preprocessing details, and training lengths. The validation-loss values are useful as run-level results, but they are not a standalone measure of general captioning quality across the two designs.

---

# 5. Training Philosophy

Both notebooks share several important engineering ideas.

### 5.1 Keep the visual representation compact

Both designs deliberately target **64 visual tokens**. This reduces the amount of multimodal context consumed by the language model.

### 5.2 Align first, adapt second

Both use staged optimization:

```text
Stage 1:
visual encoder frozen
language model frozen
connector learns alignment

        ↓

Stage 2:
visual encoder frozen
connector remains trainable
language model receives LoRA adaptation
```

This avoids asking randomly initialized connector weights and language-model adaptation layers to solve the entire cross-modal problem simultaneously.

### 5.3 Measure hardware-sensitive parameters

Rather than relying entirely on theoretical VRAM calculations, both notebooks benchmark candidate physical batch sizes on the actual Kaggle T4.

This matters because activation memory, CUDA allocator behavior, model implementation details, and precision choices all affect the real usable batch size.

### 5.4 Monitor qualitative grounding

Both notebooks save image + ground-truth + prediction panels throughout training.

This is important because captioning loss alone can hide a failure mode where the language model produces fluent, statistically plausible captions that are not actually supported by the image.

---

# 6. Important Result Interpretation

The reported validation losses demonstrate that both training pipelines learn a useful image-to-language mapping, with both showing steadily decreasing validation loss during their respective runs.

However, the qualitative panels also expose residual grounding errors:

- object identity can be wrong;
- actions can be hallucinated;
- scene location can be confused;
- multiple objects can collapse into one generic description;
- a fluent sentence can still be visually incorrect.

For that reason, the notebooks intentionally retain **qualitative prediction artifacts alongside scalar loss curves**.

Neither notebook reports COCO BLEU, METEOR, ROUGE-L, CIDEr, or SPICE scores in the executed result sections. Consequently, this README does not invent or infer those metrics.

---

# 7. Reproducibility

The notebooks use deterministic seeds and save compact checkpoints containing trainable multimodal parameters and training state.

### Alpha

Base models:

```text
YOLO11n: yolo11n.pt
Language model: google/flan-t5-small
```

Important final artifact:

```text
alpha_final_trainable.pt
```

### Beta

Base models are reconstructed from the same model identifiers used by the notebook, then the compact trained connector/LoRA state is applied.

Important final artifact:

```text
beta_final.pt
```

Both approaches deliberately avoid duplicating the large frozen pretrained backbones inside their final compact checkpoints.

---

# 8. Notebook Artifacts

The Alpha notebook records:

```text
/kaggle/working/alpha_captioner/
├── checkpoints/
├── epoch_images/
└── logs/
    ├── training.log
    └── training_log.jsonl
```

The Beta notebook records its final artifacts under:

```text
/kaggle/working/blueprint_beta_artifacts/
```

and checkpoints under:

```text
/kaggle/working/blueprint_beta_checkpoints/
```

The notebooks also contain the source code for rebuilding the multimodal models, loading the compact checkpoints, continuing training, and generating captions.

---

## 9. Embedded Figures Extracted From the Notebooks

The `assets/` directory accompanying this README contains the PNG figures extracted directly from the executed notebook outputs, including:

- Alpha training curve
- Alpha qualitative prediction panels
- Alpha final qualitative test panel
- Beta training curve
- Beta per-epoch qualitative panels
- Beta unseen COCO val2017 qualitative test

This makes the README self-contained when the entire folder is kept together.

---

## 10. References

- [Blueprint Alpha Kaggle notebook](https://www.kaggle.com/code/nilanjansaha123/yolo11n-ldpv2-flan-t5-small-image-captioner5)
- [Blueprint Beta Kaggle notebook](https://www.kaggle.com/code/nilanjansaha123hello/mobileclip-s0-pixelshuffle-pooling-smollm2-135-2)
- COCO 2017 dataset used by both notebooks: `awsaf49/coco-2017-dataset`
