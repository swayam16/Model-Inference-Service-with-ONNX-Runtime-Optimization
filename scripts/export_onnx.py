"""Export a Hugging Face sequence-classification model to ONNX.

    python -m scripts.export_onnx                      # default sentiment model
    python -m scripts.export_onnx --model <hub-id-or-local-path>

Writes to ``artifacts/model``:

* tokenizer + config files and the PyTorch weights (the baseline),
* ``model.onnx``        - FP32 graph with dynamic batch and sequence axes,
* ``model.quant.onnx``  - dynamically quantized INT8 graph (the serving default),
* ``export_report.json`` - sizes and a parity check against PyTorch.

The export fails if the FP32 ONNX logits drift from PyTorch beyond a tolerance.
"""

from __future__ import annotations

import argparse
import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.engines import OnnxEngine, PyTorchEngine, TextEncoder
from scripts.samples import SAMPLE_TEXTS

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
FP32_TOLERANCE = 1e-3


class _LogitsOnly(torch.nn.Module):
    """Positional-input wrapper so the exported graph has named inputs and one output."""

    def __init__(self, model: torch.nn.Module, names: list[str]) -> None:
        super().__init__()
        self.model = model
        self.names = names

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.model(**dict(zip(self.names, inputs))).logits


def export(model_id: str, out_dir: Path, opset: int, max_len: int, quantize: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    if not (out_dir / "tokenizer.json").exists():
        raise SystemExit("This model has no fast tokenizer (tokenizer.json); pick another model.")

    print(f"[2/4] Exporting to ONNX (opset {opset})")
    encoder = TextEncoder(out_dir, max_len)
    accepted = set(inspect.signature(model.forward).parameters)
    sample = {k: torch.from_numpy(v) for k, v in encoder(SAMPLE_TEXTS[:4]).items() if k in accepted}
    names = list(sample)
    onnx_path = out_dir / "model.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            _LogitsOnly(model, names),
            tuple(sample.values()),
            str(onnx_path),
            input_names=names,
            output_names=["logits"],
            dynamic_axes={
                **{n: {0: "batch", 1: "sequence"} for n in names},
                "logits": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    quant_path = out_dir / "model.quant.onnx"
    if quantize:
        print("[3/4] Quantizing weights to INT8 (dynamic quantization)")
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QInt8)
    else:
        print("[3/4] Skipping quantization")
        quant_path.unlink(missing_ok=True)

    print("[4/4] Checking parity against PyTorch")
    reference, _, _ = PyTorchEngine(out_dir, max_len).logits(SAMPLE_TEXTS)
    report = {
        "model_id": model_id,
        "opset": opset,
        "max_seq_len": max_len,
        "inputs": names,
        "labels": [model.config.id2label[i] for i in sorted(model.config.id2label)],
        "variants": {},
    }
    for file in ("model.onnx", "model.quant.onnx"):
        if not (out_dir / file).exists():
            continue
        logits, _, _ = OnnxEngine(out_dir, file, max_len).logits(SAMPLE_TEXTS)
        report["variants"][file] = {
            "size_mb": round((out_dir / file).stat().st_size / 1e6, 2),
            "max_abs_logit_diff": float(np.abs(logits - reference).max()),
            "label_agreement": float((logits.argmax(-1) == reference.argmax(-1)).mean()),
        }
    weights = out_dir / "model.safetensors"
    if weights.exists():
        report["pytorch_size_mb"] = round(weights.stat().st_size / 1e6, 2)

    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["variants"], indent=2))

    drift = report["variants"]["model.onnx"]["max_abs_logit_diff"]
    if drift > FP32_TOLERANCE:
        raise SystemExit(f"FP32 ONNX output drifts from PyTorch by {drift:.2e}; export rejected.")
    print(f"Export OK -> {out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub model id or local directory.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    export(args.model, args.out, args.opset, args.max_len, not args.no_quantize)


if __name__ == "__main__":
    main()
