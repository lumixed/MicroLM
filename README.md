# MicroLM 🧠

A **125M-parameter GPT-style language model built entirely from scratch** in PyTorch — custom BPE tokenizer, Grouped Query Attention, RoPE, SwiGLU, full training loop, SFT + DPO fine-tuning, and a live Gradio demo.

> **Every component is implemented from first principles.** No HuggingFace Transformers. No pre-built tokenizers. No shortcuts.

---

## Architecture

```
Parameters:  ~125M
Layers:      12 transformer decoder blocks
Attention:   Grouped Query Attention (12 Q heads, 4 KV heads)
Position:    Rotary Position Embeddings (RoPE, θ=10k)
FFN:         SwiGLU (gate · up · down projections)
Norm:        Pre-RMSNorm
Tokenizer:   Custom byte-level BPE (32k vocab)
Training:    bfloat16 AMP · grad accumulation · cosine LR · grad checkpointing
Inference:   KV-cache · temperature / top-p / top-k sampling
```

| Component | Implementation |
|---|---|
| Position Embedding | Rotary Position Embedding (RoPE) |
| Normalization | RMSNorm (pre-norm, no bias) |
| Activation | SwiGLU |
| Attention | Grouped Query Attention (GQA) + Flash Attention fallback |
| Tokenizer | Custom byte-level BPE |
| Precision | bfloat16 + AMP |

---

## Project Structure

```
MicroLM/
├── tokenizer/          # Custom BPE tokenizer (bpe.py, train_tokenizer.py)
├── model/              # Architecture: config, RMSNorm, RoPE, SwiGLU, GQA, MicroLM
├── training/           # Trainer, dataset, LR scheduler, CLI, YAML configs
├── finetuning/         # SFT (CodeAlpaca) + DPO (HH-RLHF) fine-tuning
├── eval/               # Perplexity · HumanEval pass@1 benchmark
├── inference/          # KV-cache autoregressive sampler
├── demo/               # FastAPI server + Gradio UI (HuggingFace Spaces ready)
├── data/               # Preprocessing pipeline (text → binary shards)
├── notebooks/          # Kaggle T4 GPU training notebook
└── tests/              # 22 unit tests (all passing)
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/lumixed/MicroLM.git
cd MicroLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run tests

```bash
python -m pytest tests/ -v
# 22 passed ✅
```

### 3. Prepare data & train tokenizer

```bash
# Download WikiText-103 (~500MB)
python -c "
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
open('data/raw/wikitext103.txt', 'w').write('\n'.join(ds['text']))
"

# Train tokenizer
python tokenizer/train_tokenizer.py \
    --input data/raw/ --vocab-size 8000 --output tokenizer/tokenizer.json

# Preprocess → binary shards
python data/preprocess.py \
    --input data/raw/ --tokenizer tokenizer/tokenizer.json --output data/tokens/
```

### 4. Local smoke test (tiny model, CPU)

```bash
python training/train.py --config training/configs/tiny_config.yaml
```

### 5. Cloud pretraining (Kaggle T4 GPU)

Upload `notebooks/kaggle_train.ipynb` to Kaggle and run with a free T4 GPU accelerator.

```bash
# Inside Kaggle notebook:
python training/train.py --config training/configs/125m_config.yaml
```

### 6. Fine-tune on CodeAlpaca (SFT)

```bash
python finetuning/finetune.py --mode sft \
    --checkpoint checkpoints/microlm-125m-final.pt \
    --output checkpoints/sft/
```

### 7. DPO alignment

```bash
python finetuning/finetune.py --mode dpo \
    --checkpoint checkpoints/sft/microlm-sft-best.pt \
    --output checkpoints/dpo/
```

### 8. Generate text

```bash
python inference/generate.py \
    --checkpoint checkpoints/sft/microlm-sft-best.pt \
    --tokenizer tokenizer/tokenizer.json \
    --prompt "def fibonacci(n):"
```

### 9. Run the API server

```bash
MICROLM_CHECKPOINT=checkpoints/sft/microlm-sft-best.pt \
MICROLM_TOKENIZER=tokenizer/tokenizer.json \
uvicorn demo.server:app --port 8000
```

### 10. Launch Gradio demo

```bash
python demo/app.py
# → http://localhost:7860
```

---

## Evaluation

```bash
# Perplexity on WikiText-103
python eval/perplexity.py \
    --checkpoint checkpoints/microlm-125m-final.pt \
    --tokenizer tokenizer/tokenizer.json \
    --text data/raw/wikitext103.txt

# HumanEval pass@1 benchmark
python eval/humaneval.py \
    --checkpoint checkpoints/sft/microlm-sft-best.pt \
    --tokenizer tokenizer/tokenizer.json
```

| Metric | Score |
|---|---|
| Perplexity (WikiText-103) | *run after pretraining* |
| HumanEval pass@1 | *run after SFT* |

---

## Training Details

| Stage | Data | Steps | LR |
|---|---|---|---|
| Pretraining | WikiText-103 | 50,000 | 3e-4 → 3e-5 (cosine) |
| SFT | CodeAlpaca-20K | 3 epochs | 1e-5 |
| DPO | Anthropic HH-RLHF | 1,000 | 1e-5 |

**Effective batch during pretraining (Kaggle T4):** 4 × 16 × 1024 = 65,536 tokens/step

---

## Tests

```
22 passed in 7.34s  (Python 3.14, Apple Silicon, CPU)

✅ RMSNorm — shape, scale, bfloat16, gradient flow
✅ RoPE — freq shapes, rotation, norm preservation (isometry)
✅ SwiGLU — shape, no NaN
✅ GQA — shape, causal mask, n_kv_heads < n_heads
✅ MicroLM — param count, forward, inference mode, loss ↓, KV cache, weight tying
✅ BPETokenizer — roundtrip encode/decode, special tokens, save/load
```

---

## Author

Built as a personal research project to deeply understand LLM internals from the ground up.
No pre-built transformer libraries were used — every component is implemented from scratch.
