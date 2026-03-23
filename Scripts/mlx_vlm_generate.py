#!/usr/bin/env python3
"""Real mlx_vlm inference workload.

Usage: python3 mlx_vlm_generate.py --model mlx-community/Qwen2-VL-7B-Instruct-4bit [--duration SECS]

If mlx-vlm is installed, runs real vision-language inference.
Otherwise falls back to mlx_lm text generation with the same process name pattern.
"""
import sys
import time
import argparse

def run_vlm(args):
    """Try real VLM inference."""
    try:
        from mlx_vlm import load, generate
        print(f"[mlx_vlm] Loading VLM model {args.model}...")
        sys.stdout.flush()
        model, processor = load(args.model)
        print("[mlx_vlm] Model loaded.")
        sys.stdout.flush()

        prompts = [
            "Describe what you see in this image in detail.",
            "What objects are present and how are they arranged?",
            "Describe the colors and composition of this scene.",
        ]

        iteration = 0
        end_time = time.time() + args.duration
        while time.time() < end_time:
            prompt = prompts[iteration % len(prompts)]
            iteration += 1
            print(f"\n[mlx_vlm] Iteration {iteration}: {prompt[:60]}...")
            sys.stdout.flush()

            t0 = time.time()
            response = generate(model, processor, prompt, max_tokens=args.max_tokens, verbose=False)
            elapsed = time.time() - t0

            print(f"[mlx_vlm] Response ({elapsed:.2f}s): {response[:200]}")
            sys.stdout.flush()

        print(f"\n[mlx_vlm] Done. {iteration} iterations.")
        return True
    except ImportError:
        return False

def run_lm_fallback(args):
    """Fall back to text-only mlx_lm generation."""
    try:
        from mlx_lm import load, generate
        # Use a smaller text model as fallback
        fallback_model = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        print(f"[mlx_vlm] mlx-vlm not installed, falling back to mlx_lm with {fallback_model}")
        sys.stdout.flush()

        model, tokenizer = load(fallback_model)
        print("[mlx_vlm] Fallback model loaded.")
        sys.stdout.flush()

        prompts = [
            "Describe a sunset over the ocean.",
            "What would a photo of a busy city street look like?",
            "Describe an abstract painting with warm colors.",
        ]

        iteration = 0
        end_time = time.time() + args.duration
        while time.time() < end_time:
            prompt = prompts[iteration % len(prompts)]
            iteration += 1

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted = prompt

            print(f"\n[mlx_vlm] Iteration {iteration}: {prompt[:60]}...")
            sys.stdout.flush()

            t0 = time.time()
            response = generate(model, tokenizer, prompt=formatted, max_tokens=args.max_tokens, verbose=False)
            elapsed = time.time() - t0
            token_count = len(tokenizer.encode(response))
            tps = token_count / elapsed if elapsed > 0 else 0

            print(f"[mlx_vlm] Response ({token_count} tokens, {tps:.1f} tok/s, {elapsed:.2f}s):")
            print(response[:200])
            sys.stdout.flush()

        print(f"\n[mlx_vlm] Done. {iteration} iterations (text fallback).")
        return True
    except ImportError:
        return False

def main():
    parser = argparse.ArgumentParser(description="mlx_vlm inference workload")
    parser.add_argument("--model", "-m", default="mlx-community/Qwen2-VL-7B-Instruct-4bit",
                        help="Model name")
    parser.add_argument("--memory", type=int, default=384,
                        help="(ignored, kept for CLI compat)")
    parser.add_argument("--duration", type=int, default=45,
                        help="Duration in seconds (default: 45)")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Max tokens per generation (default: 150)")
    args = parser.parse_args()

    print(f"[mlx_vlm] Model: {args.model}")
    print(f"[mlx_vlm] PID: {__import__('os').getpid()}")
    print(f"[mlx_vlm] Duration: {args.duration}s")
    sys.stdout.flush()

    if not run_vlm(args):
        if not run_lm_fallback(args):
            print("[mlx_vlm] ERROR: Neither mlx-vlm nor mlx-lm installed.")
            sys.exit(1)

if __name__ == "__main__":
    main()
