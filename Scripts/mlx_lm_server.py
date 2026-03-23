#!/usr/bin/env python3
"""Real mlx_lm.server workload — loads a model and serves repeated inference.

Usage: python3 mlx_lm_server.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit [--duration SECS]

Loads the model and runs continuous generation to simulate a server handling requests.
"""
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="mlx_lm.server inference workload")
    parser.add_argument("--model", "-m", default="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                        help="HuggingFace model to load")
    parser.add_argument("--memory", type=int, default=512,
                        help="(ignored, kept for CLI compat)")
    parser.add_argument("--duration", type=int, default=120,
                        help="Duration in seconds (default: 120)")
    parser.add_argument("--port", type=int, default=8080,
                        help="(simulated server port, for display)")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Max tokens per request (default: 150)")
    args = parser.parse_args()

    print(f"[mlx_lm.server] Model: {args.model}")
    print(f"[mlx_lm.server] PID: {__import__('os').getpid()}")
    print(f"[mlx_lm.server] Simulated port: {args.port}")
    print(f"[mlx_lm.server] Duration: {args.duration}s")
    sys.stdout.flush()

    try:
        from mlx_lm import load, generate
    except ImportError:
        print("[mlx_lm.server] ERROR: mlx-lm not installed. Run: pip3 install mlx-lm")
        sys.exit(1)

    print(f"[mlx_lm.server] Loading model {args.model}...")
    sys.stdout.flush()
    model, tokenizer = load(args.model)
    print("[mlx_lm.server] Model loaded. Serving requests...")
    sys.stdout.flush()

    # Simulate incoming requests with varied prompts
    requests = [
        "Summarize the benefits of edge AI inference.",
        "Write Python code to sort a list of dictionaries by a key.",
        "What is the transformer architecture?",
        "Explain quantization in neural networks.",
        "How does key-value caching speed up autoregressive generation?",
        "Compare MLX with PyTorch for on-device inference.",
        "Write a short story about a robot learning to paint.",
        "What are the advantages of running models locally vs cloud?",
    ]

    req_num = 0
    end_time = time.time() + args.duration
    while time.time() < end_time:
        prompt = requests[req_num % len(requests)]
        req_num += 1

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        print(f"\n[mlx_lm.server] Request #{req_num}: {prompt[:60]}...")
        sys.stdout.flush()

        t0 = time.time()
        response = generate(model, tokenizer, prompt=formatted, max_tokens=args.max_tokens, verbose=False)
        elapsed = time.time() - t0

        token_count = len(tokenizer.encode(response))
        tps = token_count / elapsed if elapsed > 0 else 0
        print(f"[mlx_lm.server] Response: {token_count} tokens, {tps:.1f} tok/s, {elapsed:.2f}s")
        print(response[:200])
        sys.stdout.flush()

        # Small pause between "requests"
        time.sleep(0.5)

    print(f"\n[mlx_lm.server] Shutting down. Served {req_num} requests.")

if __name__ == "__main__":
    main()
