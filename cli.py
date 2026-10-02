import argparse
import sys

from engine import Engine, Options, download


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Run Qwen3-0.6B with the AUREI engine")
    p.add_argument("prompt", nargs="?", default="Explain KV caching in three sentences.")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.download:
        print(download())
        return
    try:
        e = Engine(args.device, args.dtype, args.context)
        options = Options(args.max_tokens, args.temperature, args.top_p, args.seed)
        for chunk in e.generate(e.prompt(args.prompt), options, not args.no_cache):
            print(chunk["text"], end="", flush=True)
        print()
    except (ValueError, RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
