"""Build a randomly initialised DistilBERT classifier and tokenizer, fully offline.

The weights are random, so predictions are meaningless. It exists for two jobs
that do not need a trained model:

* ``--size tiny``  - a few-hundred-KB model for unit tests and CI.
* ``--size base``  - the real DistilBERT-base architecture (6 layers, 768 hidden),
  for measuring latency where the Hugging Face Hub is unreachable. Latency
  depends on the architecture and input length, not on the weight values.

    python -m scripts.make_test_model --size tiny --out artifacts/tiny-src
    python -m scripts.export_onnx --model artifacts/tiny-src --out artifacts/model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from scripts.samples import SAMPLE_TEXTS

SIZES = {
    "tiny": dict(dim=32, n_layers=2, n_heads=2, hidden_dim=64, vocab_size=None),
    "base": dict(dim=768, n_layers=6, n_heads=12, hidden_dim=3072, vocab_size=30522),
}


def build_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tok.train_from_iterator(
        SAMPLE_TEXTS, trainers.WordPieceTrainer(vocab_size=2000, special_tokens=specials)
    )
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_input_names=["input_ids", "attention_mask"],
    )


def build(size: str, out_dir: Path, seed: int = 0) -> None:
    torch.manual_seed(seed)
    tokenizer = build_tokenizer()
    spec = dict(SIZES[size])
    vocab_size = spec.pop("vocab_size") or len(tokenizer)
    config = DistilBertConfig(
        vocab_size=max(vocab_size, len(tokenizer)),
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        **spec,
    )
    model = DistilBertForSequenceClassification(config).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    params = sum(p.numel() for p in model.parameters())
    print(f"Wrote random '{size}' model ({params / 1e6:.1f}M parameters) to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--size", choices=sorted(SIZES), default="tiny")
    parser.add_argument("--out", type=Path, default=Path("artifacts/tiny-src"))
    args = parser.parse_args()
    build(args.size, args.out)


if __name__ == "__main__":
    main()
