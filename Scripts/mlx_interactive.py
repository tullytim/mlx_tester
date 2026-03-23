#!/usr/bin/env python3
"""Interactive mlx_lm chat — reads prompts from stdin, streams responses to stdout.

Protocol:
  - Reads one line from stdin = user prompt
  - Writes response tokens to stdout
  - Writes __DONE__ tps=X.X tokens=N on its own line when generation finishes
  - Writes __ERROR__ message on failure
  - Writes __READY__ when model is loaded and ready for prompts
  - Writes __LOADED__ model_name when model finishes loading
"""
import sys
import os
import time
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    print(f"__STATUS__ Loading {args.model}...", flush=True)

    try:
        from mlx_lm import load, stream_generate, generate
    except ImportError:
        print("__ERROR__ mlx-lm not installed", flush=True)
        sys.exit(1)

    try:
        model, tokenizer = load(args.model)
    except Exception as e:
        print(f"__ERROR__ Failed to load model: {e}", flush=True)
        sys.exit(1)

    print(f"__LOADED__ {args.model}", flush=True)
    print("__READY__", flush=True)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break  # EOF
            prompt_text = line.strip()
            if not prompt_text:
                continue

            # Apply chat template if available
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt_text}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = prompt_text

            t0 = time.time()
            token_count = 0

            # Try streaming first
            try:
                for token_obj in stream_generate(
                    model, tokenizer, prompt=formatted, max_tokens=args.max_tokens
                ):
                    # stream_generate yields objects with .text
                    text = token_obj.text if hasattr(token_obj, 'text') else str(token_obj)
                    token_count += 1
                    sys.stdout.write(text)
                    sys.stdout.flush()
            except Exception:
                # Fallback to non-streaming
                response = generate(
                    model, tokenizer, prompt=formatted,
                    max_tokens=args.max_tokens, verbose=False
                )
                token_count = len(tokenizer.encode(response))
                sys.stdout.write(response)
                sys.stdout.flush()

            elapsed = time.time() - t0
            tps = token_count / elapsed if elapsed > 0 else 0
            print(f"\n__DONE__ tps={tps:.1f} tokens={token_count}", flush=True)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n__ERROR__ {e}", flush=True)

if __name__ == "__main__":
    main()
