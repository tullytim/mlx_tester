#!/usr/bin/env python3
"""Real mlx_whisper transcription workload.

Usage: python3 mlx_whisper_transcribe.py [--duration SECS]

If mlx-whisper is installed, runs real audio transcription.
Otherwise falls back to MLX core compute to exercise the GPU.
"""
import sys
import time
import argparse

def run_whisper(args):
    """Try real whisper transcription."""
    try:
        import mlx_whisper
        print(f"[mlx_whisper] Loading model {args.model}...")
        sys.stdout.flush()

        # Generate synthetic audio data for transcription
        import numpy as np
        sample_rate = 16000
        duration_audio = 5  # 5 second clips

        iteration = 0
        end_time = time.time() + args.duration
        while time.time() < end_time:
            iteration += 1
            # Create random audio (will produce gibberish transcription, but exercises the model)
            audio = np.random.randn(sample_rate * duration_audio).astype(np.float32) * 0.01

            print(f"\n[mlx_whisper] Transcription {iteration}...")
            sys.stdout.flush()

            t0 = time.time()
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=args.model)
            elapsed = time.time() - t0

            text = result.get("text", "")[:200]
            print(f"[mlx_whisper] Result ({elapsed:.2f}s): {text}")
            sys.stdout.flush()

        print(f"\n[mlx_whisper] Done. {iteration} transcriptions.")
        return True
    except ImportError:
        return False

def run_mlx_fallback(args):
    """Fall back to MLX core compute."""
    try:
        import mlx.core as mx
        print("[mlx_whisper] mlx-whisper not installed, running MLX core compute fallback...")
        sys.stdout.flush()

        # Simulate whisper-like compute: conv1d + attention-like matmuls
        iteration = 0
        end_time = time.time() + args.duration
        while time.time() < end_time:
            iteration += 1
            # Simulate encoder: large matmul (like audio feature extraction)
            features = mx.random.normal((1, 1500, 512))
            weight = mx.random.normal((512, 512))
            encoded = features @ weight
            mx.eval(encoded)

            # Simulate decoder: attention-like ops
            q = mx.random.normal((1, 8, 100, 64))
            k = mx.random.normal((1, 8, 100, 64))
            v = mx.random.normal((1, 8, 100, 64))
            attn = (q @ mx.transpose(k, (0, 1, 3, 2))) * 0.125
            mx.eval(attn)

            t = time.time()
            remaining = int(end_time - t)
            if remaining % 5 == 0 and remaining > 0 and iteration % 10 == 0:
                print(f"[mlx_whisper] {remaining}s remaining, {iteration} compute cycles...")
                sys.stdout.flush()

        print(f"\n[mlx_whisper] Done. {iteration} compute cycles (MLX fallback).")
        return True
    except ImportError:
        return False

def main():
    parser = argparse.ArgumentParser(description="mlx_whisper transcription workload")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx",
                        help="Model name")
    parser.add_argument("--memory", type=int, default=128,
                        help="(ignored, kept for CLI compat)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration in seconds (default: 30)")
    args = parser.parse_args()

    print(f"[mlx_whisper] Model: {args.model}")
    print(f"[mlx_whisper] PID: {__import__('os').getpid()}")
    print(f"[mlx_whisper] Duration: {args.duration}s")
    sys.stdout.flush()

    if not run_whisper(args):
        if not run_mlx_fallback(args):
            print("[mlx_whisper] ERROR: Neither mlx-whisper nor mlx installed.")
            sys.exit(1)

if __name__ == "__main__":
    main()
