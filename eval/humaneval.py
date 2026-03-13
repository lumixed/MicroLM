"""
eval/humaneval.py

HumanEval benchmark runner for MicroLM.

HumanEval (Chen et al., 2021) contains 164 Python programming problems.
Each problem provides a function signature + docstring; the model must complete
the function body. Solutions are evaluated by running hidden test cases.

Reports pass@1: fraction of problems solved on the first attempt (greedy decoding).

Paper: https://arxiv.org/abs/2107.03374
Dataset: openai/openai_humaneval on HuggingFace

Usage:
    python eval/humaneval.py \\
        --checkpoint checkpoints/microlm-sft-best.pt \\
        --tokenizer tokenizer/tokenizer.json \\
        --temperature 0.0 \\
        --max-new-tokens 256
"""

from __future__ import annotations

import sys
import argparse
import subprocess
import tempfile
import textwrap
import json
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import MicroLM, MicroLMConfig
from tokenizer.bpe import BPETokenizer
from inference.generate import generate


STOP_TOKENS = ["\ndef ", "\nclass ", "\n#", "\nif __name__"]


def trim_completion(completion: str) -> str:
    """
    Trim generated completion at common stop sequences.
    Keeps only the function body.
    """
    for stop in STOP_TOKENS:
        if stop in completion:
            completion = completion[: completion.index(stop)]
    return completion


def execute_solution(
    problem: dict,
    completion: str,
    timeout: int = 5,
) -> bool:
    """
    Execute the completed function against the problem's test suite.

    Args:
        problem:    HumanEval problem dict (keys: prompt, test, entry_point).
        completion: Model-generated function body.
        timeout:    Seconds before subprocess is killed.

    Returns:
        True if all tests pass, False otherwise.
    """
    full_program = (
        problem["prompt"]
        + completion
        + "\n\n"
        + problem["test"]
        + f"\n\ncheck({problem['entry_point']})\n"
    )

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full_program)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_humaneval(
    model: MicroLM,
    tokenizer: BPETokenizer,
    device: torch.device,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    max_problems: Optional[int] = None,
) -> dict:
    """
    Run full HumanEval benchmark.

    Returns:
        dict with keys: pass_at_1, passed, total, per_problem_results
    """
    from datasets import load_dataset
    problems = list(load_dataset("openai/openai_humaneval", split="test"))

    if max_problems:
        problems = problems[:max_problems]

    passed = 0
    results = []

    for i, problem in enumerate(problems):
        prompt = problem["prompt"]
        prompt_ids = tokenizer.encode(prompt)

        # Greedy decode (temperature=0) for deterministic pass@1
        completion_ids = generate(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1e-8,
            top_k=1 if temperature == 0 else 50,
            top_p=top_p,
            device=device,
        )
        completion = tokenizer.decode(completion_ids)
        completion = trim_completion(completion)

        ok = execute_solution(problem, completion)
        passed += ok
        results.append({
            "task_id": problem["task_id"],
            "passed": ok,
            "completion": completion,
        })

        status = "✅" if ok else "❌"
        print(f"  [{i+1:3d}/{len(problems)}] {problem['task_id']} {status}")

    pass_at_1 = passed / len(problems)
    print(f"\n{'='*40}")
    print(f"  pass@1:  {pass_at_1:.1%}  ({passed}/{len(problems)} problems)")
    print(f"{'='*40}")

    return {
        "pass_at_1": pass_at_1,
        "passed": passed,
        "total": len(problems),
        "per_problem_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="HumanEval benchmark for MicroLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.0, help="0=greedy")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-problems", type=int, default=None, help="Limit for quick tests")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = BPETokenizer.load(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config", MicroLMConfig())
    model = MicroLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model: {model.param_count():,} parameters")

    results = run_humaneval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_problems=args.max_problems,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
