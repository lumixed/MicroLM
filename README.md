# MicroLM 🧠

A **125M-parameter GPT-style language model built entirely from scratch** in PyTorch, featuring a modern architecture with RoPE, SwiGLU, Grouped Query Attention, and Flash Attention. Trained on code corpora with SFT and DPO alignment.

> **This project implements every component from first principles** — custom BPE tokenizer, full transformer architecture, training infrastructure, and inference engine.

---

## Architecture

| Component | Implementation |
|---|---|
| Position Embedding | Rotary Position Embedding (RoPE) |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |
| Attention | Grouped Query Attention (GQA) + Flash Attention |
| Tokenizer | Custom Byte-Pair Encoding (BPE) |
| Precision | bfloat16 + AMP |

**Config (125M):**
```
n_layers=12, n_heads=12, n_kv_heads=4, d_model=768, d_ff=2048, vocab_size=32000, ctx_len=1024
```

---

## Project Structure

```
microlm/
├── data/           # dataset download, filtering, deduplication
├── tokenizer/      # custom BPE tokenizer from scratch
├── model/          # transformer architecture components
├── training/       # training loop, LR scheduler, checkpointing
├── finetuning/     # SFT and DPO fine-tuning
├── eval/           # HumanEval / MBPP evaluation
├── inference/      # KV-cache sampler, INT8 quantization, FastAPI server
├── demo/           # Gradio demo
└── notebooks/      # Kaggle/Colab training notebooks
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Train a tiny tokenizer for testing
python tokenizer/train_tokenizer.py --input data/sample.txt --vocab-size 1000 --output tokenizer/tok_test.json

# 3. Run architecture unit tests
python -m pytest tests/ -v

# 4. Launch training (local tiny run)
python training/train.py --config training/configs/tiny_config.yaml

# 5. Generate text
python inference/generate.py --checkpoint checkpoints/latest.pt --prompt "def fibonacci("
```

---

## Phases

- [x] **Phase 0**: Project setup
- [ ] **Phase 1**: Data pipeline + custom BPE tokenizer
- [ ] **Phase 2**: Model architecture (RMSNorm, RoPE, GQA, SwiGLU)
- [ ] **Phase 3**: Training infrastructure
- [ ] **Phase 4**: Pretraining on code corpus (cloud)
- [ ] **Phase 5**: SFT + DPO fine-tuning
- [ ] **Phase 6**: Inference engine + deployment
- [ ] **Phase 7**: Evaluation + documentation

---

## Author

Built as a personal research project to understand LLM internals from the ground up.
