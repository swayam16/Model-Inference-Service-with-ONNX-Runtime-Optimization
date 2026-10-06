"""Benchmark the PyTorch baseline against ONNX Runtime on CPU.

    python -m scripts.benchmark
    python -m scripts.benchmark --requests 500 --batch-sizes 1,8,16 --threads 2

Every engine runs the same texts through the same tokenizer, so the difference
is the model runtime. Results go to ``artifacts/benchmark.json`` (served by the
API at ``/benchmark`` and shown on the dashboard) and ``artifacts/benchmark.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.engines import OnnxEngine, PyTorchEngine
from scripts.samples import SAMPLE_TEXTS


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def measure(engine, batch_size: int, requests: int, warmup: int) -> dict:
    """Time ``requests`` sequential calls of ``batch_size`` texts (tokenize + inference)."""
    cursor = 0

    def next_batch() -> list[str]:
        nonlocal cursor
        batch = [SAMPLE_TEXTS[(cursor + i) % len(SAMPLE_TEXTS)] for i in range(batch_size)]
        cursor += batch_size
        return batch

    for _ in range(warmup):
        engine.logits(next_batch())

    cursor = 0  # every engine sees the identical sequence of batches
    latencies = np.empty(requests)
    started = time.perf_counter()
    for i in range(requests):
        t0 = time.perf_counter()
        engine.logits(next_batch())
        latencies[i] = (time.perf_counter() - t0) * 1e3
    elapsed = time.perf_counter() - started

    p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
    return {
        "engine": engine.name,
        "batch_size": batch_size,
        "p50_ms": round(float(p50), 3),
        "p95_ms": round(float(p95), 3),
        "p99_ms": round(float(p99), 3),
        "mean_ms": round(float(latencies.mean()), 3),
        "throughput_items_s": round(requests * batch_size / elapsed, 1),
    }


def to_markdown(report: dict) -> str:
    env = report["environment"]
    lines = [
        f"Model: `{report['model_id']}` · {env['cpu']} · {env['threads']} threads · "
        f"{report['config']['requests']} requests per row · max {report['config']['max_seq_len']} "
        "tokens",
        "",
        "| Engine | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (items/s) | p95 vs PyTorch |",  # noqa: E501
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    baseline = {r["batch_size"]: r["p95_ms"] for r in report["results"] if r["engine"] == "pytorch"}
    for r in report["results"]:
        change = (r["p95_ms"] / baseline[r["batch_size"]] - 1) * 100
        delta = "baseline" if r["engine"] == "pytorch" else f"{change:+.1f}%"
        lines.append(
            f"| {r['engine']} | {r['batch_size']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | "
            f"{r['p99_ms']:.2f} | {r['throughput_items_s']:.1f} | {delta} |"
        )
    if report.get("model_files"):
        lines += [
            "",
            "| File | Size (MB) | Max logit diff vs PyTorch | Label agreement |",
            "|---|---:|---:|---:|",
        ]
        for name, info in report["model_files"].items():
            diff = info.get("max_abs_logit_diff")
            agree = info.get("label_agreement")
            lines.append(
                f"| {name} | {info['size_mb']:.1f} | "
                f"{'-' if diff is None else f'{diff:.2e}'} | "
                f"{'-' if agree is None else f'{agree:.0%}'} |"
            )
    return "\n".join(lines) + "\n"


def run(
    model_dir: Path, requests: int, warmup: int, batch_sizes: list[int], threads: int, max_len: int
) -> dict:
    import onnxruntime
    import torch

    threads = threads or os.cpu_count() or 1
    engines = [PyTorchEngine(model_dir, max_len, threads)]
    for file in ("model.onnx", "model.quant.onnx"):
        if (model_dir / file).exists():
            engines.append(OnnxEngine(model_dir, file, max_len, threads))

    results = []
    for batch_size in batch_sizes:
        for engine in engines:
            row = measure(engine, batch_size, requests, warmup)
            results.append(row)
            print(
                f"{row['engine']:<10} batch={batch_size:<3} p50={row['p50_ms']:>8.2f} ms  "
                f"p95={row['p95_ms']:>8.2f} ms  {row['throughput_items_s']:>8.1f} items/s"
            )

    export = {}
    report_path = model_dir / "export_report.json"
    if report_path.exists():
        export = json.loads(report_path.read_text())
    model_files = {}
    if "pytorch_size_mb" in export:
        model_files["model.safetensors (PyTorch)"] = {"size_mb": export["pytorch_size_mb"]}
    model_files.update(export.get("variants", {}))

    def p95(engine: str, batch_size: int) -> float | None:
        return next(
            (
                r["p95_ms"]
                for r in results
                if r["engine"] == engine and r["batch_size"] == batch_size
            ),
            None,
        )

    first = batch_sizes[0]
    base = p95("pytorch", first)
    summary = {"batch_size": first, "baseline_engine": "pytorch", "baseline_p95_ms": base}
    for name in ("onnx", "onnx-int8"):
        value = p95(name, first)
        if value:
            summary[name] = {
                "p95_ms": value,
                "p95_change_pct": round((value / base - 1) * 100, 1),
                "speedup": round(base / value, 2),
            }

    return {
        "model_id": export.get("model_id", model_dir.name),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "cpu": cpu_model(),
            "logical_cpus": os.cpu_count(),
            "threads": threads,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "config": {"requests": requests, "warmup": warmup, "max_seq_len": max_len},
        "results": results,
        "model_files": model_files,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmark.json"))
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--threads", type=int, default=0, help="0 = all logical CPUs")
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    report = run(
        args.model_dir,
        args.requests,
        args.warmup,
        [int(b) for b in args.batch_sizes.split(",")],
        args.threads,
        args.max_len,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    markdown = to_markdown(report)
    args.out.with_suffix(".md").write_text(markdown)
    print("\n" + markdown)
    print(f"Saved {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
