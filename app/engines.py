"""Inference engines.

Two interchangeable backends run the same exported model directory:

* ``OnnxEngine``    - ONNX Runtime on CPU (the optimized serving path).
* ``PyTorchEngine`` - native PyTorch (the baseline the service is measured against).

Both share one tokenizer implementation (the Rust ``tokenizers`` library), so a
benchmark between them isolates the model runtime rather than preprocessing.
The ONNX path does not import torch or transformers, which keeps the serving
image small.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    scores: dict[str, float]


@dataclass(frozen=True)
class BatchOutput:
    predictions: list[Prediction]
    tokenize_ms: float
    inference_ms: float


class Engine(Protocol):
    name: str
    model_id: str
    labels: list[str]

    def predict(self, texts: list[str]) -> BatchOutput: ...


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_labels(model_dir: Path) -> list[str]:
    id2label = _read_json(model_dir / "config.json").get("id2label") or {}
    if not id2label:
        raise ValueError(f"{model_dir}/config.json has no id2label mapping")
    return [id2label[k] for k in sorted(id2label, key=int)]


def load_model_id(model_dir: Path) -> str:
    return _read_json(model_dir / "export_report.json").get("model_id", model_dir.name)


class TextEncoder:
    """Tokenizes, truncates to ``max_len`` and pads.

    ``buckets`` is what the engines use: it groups a batch by token length so
    short texts are not padded out to the longest text in the batch. Padding
    tokens cost as much compute as real ones, so on mixed-length traffic one
    padded batch can be slower than no batching at all.
    """

    def __init__(
        self, model_dir: Path, max_len: int, bucket_ratio: float = 1.25, bucket_slack: int = 4
    ) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        pad_token = self._find_pad_token(model_dir)
        pad_id = self._tok.token_to_id(pad_token)
        if pad_id is None:
            raise ValueError(f"pad token {pad_token!r} is not in the tokenizer vocabulary")
        self._pad_id = pad_id
        self._tok.enable_truncation(max_length=max_len)
        self._tok.no_padding()
        self.bucket_ratio = bucket_ratio
        self.bucket_slack = bucket_slack

    @staticmethod
    def _find_pad_token(model_dir: Path) -> str:
        for name in ("special_tokens_map.json", "tokenizer_config.json"):
            token = _read_json(model_dir / name).get("pad_token")
            if isinstance(token, dict):
                token = token.get("content")
            if token:
                return token
        return "[PAD]"

    def _pad(self, encodings: list) -> dict[str, np.ndarray]:
        width = max(len(e.ids) for e in encodings)
        ids = np.full((len(encodings), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(encodings), width), dtype=np.int64)
        types = np.zeros((len(encodings), width), dtype=np.int64)
        for row, e in enumerate(encodings):
            n = len(e.ids)
            ids[row, :n] = e.ids
            mask[row, :n] = 1
            types[row, :n] = e.type_ids
        return {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}

    def __call__(self, texts: list[str]) -> dict[str, np.ndarray]:
        """One batch, padded to its longest text."""
        return self._pad(self._tok.encode_batch(texts))

    def buckets(self, texts: list[str]) -> list[tuple[list[int], dict[str, np.ndarray]]]:
        """Split a batch into length-sorted groups: ``[(original_indices, inputs), ...]``.

        A text joins the current group while it is at most ``bucket_ratio`` times
        (plus ``bucket_slack`` tokens) longer than the group's shortest text.
        """
        encodings = self._tok.encode_batch(texts)
        order = sorted(range(len(encodings)), key=lambda i: len(encodings[i].ids))
        groups: list[list[int]] = []
        limit = -1.0
        for i in order:
            length = len(encodings[i].ids)
            if length > limit:
                groups.append([])
                limit = length * self.bucket_ratio + self.bucket_slack
            groups[-1].append(i)
        return [(group, self._pad([encodings[i] for i in group])) for group in groups]


def _to_predictions(logits: np.ndarray, labels: list[str]) -> list[Prediction]:
    probs = softmax(logits.astype(np.float64))
    out = []
    for row in probs:
        best = int(row.argmax())
        out.append(
            Prediction(
                label=labels[best],
                score=float(row[best]),
                scores={label: float(p) for label, p in zip(labels, row)},
            )
        )
    return out


class _BaseEngine:
    """Shared batch path: tokenize, run each length bucket, restore request order."""

    labels: list[str]
    _encode: TextEncoder

    def _forward(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def logits(self, texts: list[str]) -> tuple[np.ndarray, float, float]:
        """Return ``(logits, tokenize_ms, inference_ms)`` for a batch of texts."""
        t0 = time.perf_counter()
        buckets = self._encode.buckets(texts)
        t1 = time.perf_counter()
        out = np.empty((len(texts), len(self.labels)), dtype=np.float32)
        for indices, inputs in buckets:
            out[indices] = self._forward(inputs)
        t2 = time.perf_counter()
        return out, (t1 - t0) * 1e3, (t2 - t1) * 1e3

    def predict(self, texts: list[str]) -> BatchOutput:
        logits, tok_ms, inf_ms = self.logits(texts)
        return BatchOutput(_to_predictions(logits, self.labels), tok_ms, inf_ms)


class OnnxEngine(_BaseEngine):
    def __init__(
        self,
        model_dir: Path,
        onnx_file: str = "model.quant.onnx",
        max_len: int = 128,
        intra_op_threads: int = 0,
    ) -> None:
        import onnxruntime as ort

        model_dir = Path(model_dir)
        path = model_dir / onnx_file
        if not path.exists():
            path = model_dir / "model.onnx"
        if not path.exists():
            raise FileNotFoundError(
                f"No ONNX model in {model_dir}. Run: python scripts/export_onnx.py"
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.inter_op_num_threads = 1
        if intra_op_threads > 0:
            opts.intra_op_num_threads = intra_op_threads
        self._session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._encode = TextEncoder(model_dir, max_len)

        self.labels = load_labels(model_dir)
        self.model_id = load_model_id(model_dir)
        self.name = "onnx-int8" if "quant" in path.name else "onnx"
        self.model_file = path.name

    def _forward(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        feed = {k: v for k, v in inputs.items() if k in self._input_names}
        return self._session.run(None, feed)[0]


class PyTorchEngine(_BaseEngine):
    """Baseline: the same weights executed by PyTorch in eval/inference mode."""

    name = "pytorch"

    def __init__(self, model_dir: Path, max_len: int = 128, intra_op_threads: int = 0) -> None:
        import inspect

        import torch
        from transformers import AutoModelForSequenceClassification

        model_dir = Path(model_dir)
        self._torch = torch
        if intra_op_threads > 0:
            torch.set_num_threads(intra_op_threads)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.eval()
        self._accepted = set(inspect.signature(self._model.forward).parameters)
        self._encode = TextEncoder(model_dir, max_len)

        self.labels = load_labels(model_dir)
        self.model_id = load_model_id(model_dir)
        self.model_file = "model.safetensors"

    def _forward(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        torch = self._torch
        feed = {k: torch.from_numpy(v) for k, v in inputs.items() if k in self._accepted}
        with torch.inference_mode():
            return self._model(**feed).logits.numpy()


def build_engine(
    kind: str, model_dir: Path, onnx_file: str, max_len: int, intra_op_threads: int
) -> Engine:
    if kind == "onnx":
        return OnnxEngine(model_dir, onnx_file, max_len, intra_op_threads)
    if kind == "pytorch":
        return PyTorchEngine(model_dir, max_len, intra_op_threads)
    raise ValueError(f"Unknown ENGINE {kind!r}; expected 'onnx' or 'pytorch'")
