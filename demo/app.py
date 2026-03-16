"""
demo/app.py

Gradio demo for MicroLM — deployable to HuggingFace Spaces.

Set environment variables before running:
    MICROLM_CHECKPOINT=checkpoints/microlm-sft-best.pt
    MICROLM_TOKENIZER=tokenizer/tokenizer.json

Run locally:
    python demo/app.py

Deploy to HuggingFace Spaces:
    1. Create a new Space (Gradio SDK) at huggingface.co/spaces
    2. Upload your repo (or link your GitHub repo)
    3. Add MICROLM_CHECKPOINT and MICROLM_TOKENIZER as Space secrets
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────────────────────────
# Load model at module level (cached across requests on HF Spaces)
# ──────────────────────────────────────────────

CHECKPOINT_PATH = os.environ.get("MICROLM_CHECKPOINT", "checkpoints/microlm-125m-final.pt")
TOKENIZER_PATH = os.environ.get("MICROLM_TOKENIZER", "tokenizer/tokenizer.json")

_model = None
_tokenizer = None
_device = None


def _load_model():
    global _model, _tokenizer, _device

    _device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    from tokenizer.bpe import BPETokenizer
    from model import MicroLM, MicroLMConfig

    _tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config", MicroLMConfig())
    _model = MicroLM(model_cfg).to(_device)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    print(f"Model loaded on {_device}")


_load_model()


# ──────────────────────────────────────────────
# Inference function
# ──────────────────────────────────────────────

def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    from inference.generate import generate

    if not prompt.strip():
        return "Please enter a prompt."

    prompt_ids = _tokenizer.encode(prompt)
    completion_ids = generate(
        model=_model,
        prompt_ids=prompt_ids,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        device=_device,
    )
    return _tokenizer.decode(completion_ids)


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

EXAMPLES = [
    ["def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n", 150, 0.7, 0.95, 50],
    ["The theory of relativity states that", 200, 0.9, 0.95, 50],
    ["Once upon a time in a land far away,", 250, 1.0, 0.95, 50],
    ["# Python quicksort implementation\ndef quicksort(arr):", 200, 0.5, 0.95, 40],
]

def build_demo():
    import gradio as gr

    with gr.Blocks(
        title="MicroLM — Custom 125M LLM Demo",
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        css="""
        #title { text-align: center; }
        #subtitle { text-align: center; color: #888; margin-bottom: 1rem; }
        .generate-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; }
        """,
    ) as demo:

        gr.Markdown(
            "# 🧠 MicroLM\n"
            "**A 125M-parameter language model built from scratch** — "
            "custom BPE tokenizer · GQA attention · RoPE · SwiGLU · trained on WikiText-103",
            elem_id="title",
        )
        gr.Markdown(
            "[GitHub](https://github.com/lumixed/MicroLM) · "
            "12 transformer layers · 12 Q heads / 4 KV heads (GQA) · "
            "bfloat16 · KV-cache inference",
            elem_id="subtitle",
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt_box = gr.Textbox(
                    label="Prompt",
                    placeholder="def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"",
                    lines=6,
                    max_lines=20,
                )
                output_box = gr.Textbox(
                    label="Completion",
                    lines=10,
                    max_lines=30,
                    interactive=False,
                )

                with gr.Row():
                    generate_btn = gr.Button(
                        "✨ Generate", variant="primary", elem_classes=["generate-btn"]
                    )
                    clear_btn = gr.Button("🗑 Clear", variant="secondary")

            with gr.Column(scale=1):
                max_tokens = gr.Slider(
                    label="Max new tokens", minimum=20, maximum=512, value=200, step=10
                )
                temperature = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.05,
                    info="Higher = more creative, lower = more focused"
                )
                top_p = gr.Slider(
                    label="Top-p (nucleus)", minimum=0.5, maximum=1.0, value=0.95, step=0.01
                )
                top_k = gr.Slider(
                    label="Top-k", minimum=1, maximum=200, value=50, step=1
                )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[prompt_box, max_tokens, temperature, top_p, top_k],
            label="Example prompts",
        )

        generate_btn.click(
            fn=generate_text,
            inputs=[prompt_box, max_tokens, temperature, top_p, top_k],
            outputs=output_box,
        )
        clear_btn.click(lambda: ("", ""), outputs=[prompt_box, output_box])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
