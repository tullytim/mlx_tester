#!/usr/bin/env python3
"""Real mlx.core benchmark — runs matrix operations and reports throughput.

Usage: python3 mlx_core_bench.py [--duration SECS]

Exercises MLX GPU compute with various matrix sizes and operations.
"""
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="mlx.core benchmark workload")
    parser.add_argument("--memory", type=int, default=512,
                        help="(ignored, kept for CLI compat)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration in seconds (default: 60)")
    args = parser.parse_args()

    print(f"[mlx.core bench] PID: {__import__('os').getpid()}")
    print(f"[mlx.core bench] Duration: {args.duration}s")
    sys.stdout.flush()

    try:
        import mlx.core as mx
    except ImportError:
        print("[mlx.core bench] ERROR: mlx not installed. Run: pip3 install mlx")
        sys.exit(1)

    sizes = [512, 1024, 2048, 4096]
    end_time = time.time() + args.duration
    total_ops = 0
    total_flops = 0

    print("[mlx.core bench] Running GPU matrix benchmarks...")
    sys.stdout.flush()

    while time.time() < end_time:
        for size in sizes:
            if time.time() >= end_time:
                break

            a = mx.random.normal((size, size))
            b = mx.random.normal((size, size))

            t0 = time.time()
            c = a @ b
            mx.eval(c)
            elapsed = time.time() - t0

            flops = 2 * size**3  # approximate FLOPs for matmul
            gflops = flops / elapsed / 1e9 if elapsed > 0 else 0
            total_ops += 1
            total_flops += flops

            print(f"[mlx.core bench] matmul {size}x{size}: {elapsed*1000:.1f}ms ({gflops:.1f} GFLOPS)")
            sys.stdout.flush()

    total_elapsed = args.duration
    avg_gflops = total_flops / total_elapsed / 1e9 if total_elapsed > 0 else 0
    print(f"\n[mlx.core bench] Done. {total_ops} ops, avg {avg_gflops:.1f} GFLOPS")

if __name__ == "__main__":
    main()
