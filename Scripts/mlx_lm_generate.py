#!/usr/bin/env python3
"""Real mlx_lm.generate inference workload for testing.

Usage: python3 mlx_lm_generate.py --model mlx-community/Llama-3.2-1B-Instruct-4bit [--duration SECS]

Downloads (if needed) and runs actual model inference using mlx-lm.
"""
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="mlx_lm.generate inference workload")
    parser.add_argument("--model", "-m", default="mlx-community/Llama-3.2-1B-Instruct-4bit",
                        help="HuggingFace model to load (default: Llama-3.2-1B-Instruct-4bit)")
    parser.add_argument("--memory", type=int, default=256,
                        help="(ignored, kept for CLI compat)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration in seconds — will loop generation until time is up (default: 60)")
    parser.add_argument("--prompt", default="Explain how Apple Silicon unified memory works in three sentences.",
                        help="Prompt to send to the model")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens per generation (default: 200)")
    args = parser.parse_args()

    print(f"[mlx_lm.generate] Model: {args.model}")
    print(f"[mlx_lm.generate] PID: {__import__('os').getpid()}")
    print(f"[mlx_lm.generate] Duration: {args.duration}s")
    sys.stdout.flush()

    try:
        from mlx_lm import load, generate
    except ImportError:
        print("[mlx_lm.generate] ERROR: mlx-lm not installed. Run: pip3 install mlx-lm")
        sys.exit(1)

    print(f"[mlx_lm.generate] Loading model {args.model}...")
    sys.stdout.flush()
    model, tokenizer = load(args.model)
    print("[mlx_lm.generate] Model loaded.")
    sys.stdout.flush()

    prompts = [
        "Explain how Apple Silicon unified memory works in three sentences.",
        "Write a haiku about machine learning on a laptop.",
        "What are the key differences between GPU and Neural Engine on M-series chips?",
        "Describe the MLX framework in one paragraph.",
        "Why is quantization useful for running LLMs locally?",
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

        print(f"\n[mlx_lm.generate] --- Iteration {iteration} ---")
        print(f"[mlx_lm.generate] Prompt: {prompt[:80]}...")
        sys.stdout.flush()

        t0 = time.time()
        response = generate(model, tokenizer, prompt=formatted, max_tokens=args.max_tokens, verbose=False)
        elapsed = time.time() - t0

        # Rough token count from response length
        token_count = len(tokenizer.encode(response))
        tps = token_count / elapsed if elapsed > 0 else 0

        print(f"[mlx_lm.generate] Response ({token_count} tokens, {tps:.1f} tok/s, {elapsed:.2f}s):")
        print(response[:300])
        sys.stdout.flush()

    print(f"\n[mlx_lm.generate] Done. {iteration} iterations completed.")

if __name__ == "__main__":
    main()
