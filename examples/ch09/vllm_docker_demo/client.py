"""Send one small request to the local vLLM OpenAI-compatible server."""

import argparse
import time
from urllib.request import Request, urlopen

from openai import OpenAI


def _post_control_endpoint(url: str) -> None:
    """Call one of vLLM's local profiler control endpoints."""
    request = Request(url, method="POST")
    with urlopen(request, timeout=30) as response:
        response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--prompt",
        default="Explain why GPU memory capacity matters during LLM inference.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="capture this request with the vLLM PyTorch profiler",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="allow Qwen3 to include its reasoning output",
    )
    args = parser.parse_args()

    # vLLM exposes an OpenAI-compatible API. The local demo does not require
    # authentication, but the client expects a non-empty placeholder key.
    client = OpenAI(base_url=args.base_url, api_key="local-demo-token")

    profile_url = args.base_url.removesuffix("/v1")
    if args.profile:
        # vLLM writes trace files only after the profiling range is stopped.
        _post_control_endpoint(f"{profile_url}/start_profile")

    try:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": args.enable_thinking,
                }
            },
        )
        elapsed = time.perf_counter() - start
    finally:
        if args.profile:
            _post_control_endpoint(f"{profile_url}/stop_profile")

    response_text = response.choices[0].message.content or ""
    completion_tokens = getattr(response.usage, "completion_tokens", None)
    finish_reason = response.choices[0].finish_reason

    print(f"Model: {args.model}")
    print(f"Elapsed time: {elapsed * 1000:.2f} ms")
    print(f"Finish reason: {finish_reason}")
    if completion_tokens:
        print(f"Generated tokens: {completion_tokens}")
        print(f"Generation throughput: {completion_tokens / elapsed:.2f} tokens/s")
    print("\nResponse:\n")
    print(response_text)


if __name__ == "__main__":
    main()
