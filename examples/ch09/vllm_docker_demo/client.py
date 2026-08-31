"""Send requests to the local vLLM OpenAI-compatible server."""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI


@dataclass
class RequestResult:
    index: int
    elapsed: float
    completion_tokens: int
    finish_reason: str
    response_text: str
    ttft: float | None = None
    mean_inter_token_latency: float | None = None
    error: str | None = None


def _post_control_endpoint(url: str, timeout_seconds: float) -> None:
    """Call one of vLLM's local profiler control endpoints."""
    request = Request(url, method="POST")
    with urlopen(request, timeout=timeout_seconds) as response:
        response.read()


def _wait_for_server(base_url: str, timeout_seconds: float) -> None:
    if timeout_seconds <= 0:
        return

    health_url = base_url.removesuffix("/v1") + "/health"
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not respond"

    while time.monotonic() < deadline:
        try:
            with urlopen(health_url, timeout=2) as response:
                response.read()
            return
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1)

    raise SystemExit(
        f"vLLM is not ready at {health_url} after {timeout_seconds:.0f} seconds "
        f"({last_error}). Start or restart the server before running the client."
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _request_prompt(base_prompt: str, index: int) -> str:
    return f"{base_prompt}\n\nRequest id: {index}"


def _extra_body(enable_thinking: bool) -> dict[str, object]:
    return {
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
        }
    }


def _run_streaming_request(args: argparse.Namespace, index: int) -> RequestResult:
    client = OpenAI(base_url=args.base_url, api_key="local-demo-token")
    prompt = _request_prompt(args.prompt, index)
    start = time.perf_counter()
    chunks = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=_extra_body(args.enable_thinking),
    )

    first_token_time: float | None = None
    previous_token_time: float | None = None
    inter_token_latencies: list[float] = []
    response_parts: list[str] = []
    finish_reason = "unknown"
    chunk_count = 0
    usage_completion_tokens: int | None = None

    for chunk in chunks:
        now = time.perf_counter()
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            usage_completion_tokens = getattr(usage, "completion_tokens", None)
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        content = choice.delta.content or ""
        if content:
            chunk_count += 1
            if first_token_time is None:
                first_token_time = now
            if previous_token_time is not None:
                inter_token_latencies.append(now - previous_token_time)
            previous_token_time = now
            response_parts.append(content)
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    elapsed = time.perf_counter() - start
    ttft = first_token_time - start if first_token_time is not None else None
    mean_itl = statistics.mean(inter_token_latencies) if inter_token_latencies else None
    return RequestResult(
        index=index,
        elapsed=elapsed,
        completion_tokens=usage_completion_tokens or chunk_count,
        finish_reason=finish_reason,
        response_text="".join(response_parts),
        ttft=ttft,
        mean_inter_token_latency=mean_itl,
    )


def _run_non_streaming_request(args: argparse.Namespace, index: int) -> RequestResult:
    client = OpenAI(base_url=args.base_url, api_key="local-demo-token")
    prompt = _request_prompt(args.prompt, index)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        extra_body=_extra_body(args.enable_thinking),
    )
    elapsed = time.perf_counter() - start
    completion_tokens = getattr(response.usage, "completion_tokens", None) or 0
    return RequestResult(
        index=index,
        elapsed=elapsed,
        completion_tokens=completion_tokens,
        finish_reason=response.choices[0].finish_reason,
        response_text=response.choices[0].message.content or "",
    )


def _run_request(args: argparse.Namespace, index: int) -> RequestResult:
    start = time.perf_counter()
    try:
        if args.stream:
            return _run_streaming_request(args, index)
        return _run_non_streaming_request(args, index)
    except Exception as exc:  # noqa: BLE001 - request errors belong in lab output.
        return RequestResult(
            index=index,
            elapsed=time.perf_counter() - start,
            completion_tokens=0,
            finish_reason="error",
            response_text="",
            error=f"{type(exc).__name__}: {exc}",
        )


def _print_summary(args: argparse.Namespace, results: list[RequestResult], total_elapsed: float) -> None:
    successful = [result for result in results if result.error is None]
    failed = [result for result in results if result.error is not None]
    latencies = [result.elapsed for result in successful]
    total_tokens = sum(result.completion_tokens for result in successful)

    print(f"Model: {args.model}")
    print(f"Requests: {len(results)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max tokens per request: {args.max_tokens}")
    print(f"Streaming: {args.stream}")
    print(f"Successful requests: {len(successful)}")
    print(f"Failed requests: {len(failed)}")
    print(f"Total wall time: {total_elapsed * 1000:.2f} ms")
    print(f"Generated tokens: {total_tokens}")
    if total_elapsed > 0:
        print(f"Aggregate throughput: {total_tokens / total_elapsed:.2f} tokens/s")

    if latencies:
        print(f"Mean latency: {statistics.mean(latencies) * 1000:.2f} ms")
        print(f"p50 latency: {_percentile(latencies, 0.50) * 1000:.2f} ms")
        print(f"p95 latency: {_percentile(latencies, 0.95) * 1000:.2f} ms")
        print(f"p99 latency: {_percentile(latencies, 0.99) * 1000:.2f} ms")

    ttfts = [result.ttft for result in successful if result.ttft is not None]
    itls = [
        result.mean_inter_token_latency
        for result in successful
        if result.mean_inter_token_latency is not None
    ]
    if ttfts:
        print(f"Mean TTFT: {statistics.mean(ttfts) * 1000:.2f} ms")
        print(f"p95 TTFT: {_percentile(ttfts, 0.95) * 1000:.2f} ms")
    if itls:
        print(f"Mean inter-token latency: {statistics.mean(itls) * 1000:.2f} ms")

    finish_reasons: dict[str, int] = {}
    for result in results:
        finish_reasons[result.finish_reason] = finish_reasons.get(result.finish_reason, 0) + 1
    print("Finish reasons:")
    for reason, count in sorted(finish_reasons.items()):
        print(f"  {reason}: {count}")

    if failed:
        print("Errors:")
        for result in failed:
            print(f"  request {result.index}: {result.error}")

    if args.show_responses:
        print("\nResponses:\n")
        for result in results:
            print(f"--- request {result.index} ---")
            print(result.error if result.error else result.response_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--prompt",
        default="Explain why GPU memory capacity matters during LLM inference.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="stream responses to estimate time to first token and inter-token latency",
    )
    parser.add_argument(
        "--show-responses",
        action="store_true",
        help="print every response body; summaries only are printed by default",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="capture this run with the vLLM PyTorch profiler",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="allow Qwen3 to include its reasoning output",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=120.0,
        help="seconds to wait for the vLLM health endpoint before sending requests",
    )
    parser.add_argument(
        "--profile-control-timeout",
        type=float,
        default=180.0,
        help="seconds to wait for vLLM start_profile and stop_profile control calls",
    )
    args = parser.parse_args()

    if args.requests < 1:
        raise SystemExit("--requests must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")

    _wait_for_server(args.base_url, args.wait_timeout)

    profile_url = args.base_url.removesuffix("/v1")
    if args.profile:
        # vLLM writes trace files only after the profiling range is stopped.
        _post_control_endpoint(f"{profile_url}/start_profile", args.profile_control_timeout)

    try:
        start = time.perf_counter()
        results: list[RequestResult] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(_run_request, args, index)
                for index in range(args.requests)
            ]
            for future in as_completed(futures):
                results.append(future.result())
        total_elapsed = time.perf_counter() - start
    finally:
        if args.profile:
            _post_control_endpoint(f"{profile_url}/stop_profile", args.profile_control_timeout)

    results.sort(key=lambda result: result.index)
    _print_summary(args, results, total_elapsed)


if __name__ == "__main__":
    main()
