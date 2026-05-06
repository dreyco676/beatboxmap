#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert a BEATs pretrained checkpoint (.pt) to ONNX.

Prerequisites
-------------
1. Download a checkpoint (e.g. BEATs_iter3+ AS2M pretrained) from the links
   in the BEATs README and place it anywhere accessible.

2. Install torch and onnx into the project venv:

       pip install torch onnx

The BEATs Python source (BEATs.py, backbone.py, etc.) is fetched
automatically from GitHub if not already cached in scripts/.beats_src/.
No manual clone required.

Usage
-----
    python scripts/convert_beats_to_onnx.py \\
        --checkpoint models/BEATs_iter3_plus_AS2M.pt \\
        --output     models/beats_iter3plus_as2m.onnx

The exported model matches EmbeddingExtractor (substrate_id="beats"):
    input_names  : ["audio_data"]
    output_names : ["output"]
    output shape : (batch, 768)  — mean-pooled over time frames
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------
# BEATs source files needed for model architecture
# ---------------------------------------------------------------

_BEATS_RAW_BASE = (
    "https://raw.githubusercontent.com/microsoft/unilm/master/beats"
)
_BEATS_SOURCE_FILES = [
    "BEATs.py",
    "backbone.py",
    "modules.py",
    "Tokenizers.py",
]

_BEATS_SRC_CACHE = Path(__file__).parent / ".beats_src"


def _ensure_beats_source() -> Path:
    """Download BEATs source files to scripts/.beats_src/ if not already there."""
    _BEATS_SRC_CACHE.mkdir(exist_ok=True)
    missing = [f for f in _BEATS_SOURCE_FILES
               if not (_BEATS_SRC_CACHE / f).exists()]
    if missing:
        print(f"Fetching BEATs source files into {_BEATS_SRC_CACHE}/ ...")
        for fname in missing:
            url = f"{_BEATS_RAW_BASE}/{fname}"
            dest = _BEATS_SRC_CACHE / fname
            print(f"  {url}")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as exc:
                sys.exit(
                    f"ERROR: Could not download {url}: {exc}\n"
                    "Check your internet connection, or clone the repo manually\n"
                    "and pass --beats-src /path/to/unilm/beats"
                )
    return _BEATS_SRC_CACHE


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert BEATs pretrained checkpoint to ONNX"
    )
    p.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to the BEATs .pt checkpoint file",
    )
    p.add_argument(
        "--output", default=Path("models/beats_iter3plus_as2m.onnx"), type=Path,
        help="Destination ONNX file (default: models/beats_iter3plus_as2m.onnx)",
    )
    p.add_argument(
        "--beats-src", default=None, type=Path,
        help=(
            "Optional: path to an already-cloned beats/ source directory. "
            "If omitted, source files are fetched from GitHub automatically."
        ),
    )
    p.add_argument(
        "--input-length", default=160_000, type=int,
        help="Audio samples fed to the model (default: 160000 = 10 s @ 16 kHz)",
    )
    p.add_argument("--opset", default=17, type=int, help="ONNX opset version")
    return p.parse_args()


def _import_beats(beats_src: Path | None):
    """Add BEATs to sys.path and import BEATs, BEATsConfig."""
    src = beats_src if beats_src is not None else _ensure_beats_source()
    sys.path.insert(0, str(src.resolve()))
    try:
        from BEATs import BEATs, BEATsConfig  # type: ignore[import]
        return BEATs, BEATsConfig
    except ImportError as exc:
        sys.exit(
            f"ERROR: Cannot import BEATs from {src}: {exc}\n"
            "Try deleting scripts/.beats_src/ and re-running so the "
            "source files are re-downloaded."
        )


def _make_wrapper(beats_model):
    """Return a torch.nn.Module: fbank_features -> mean-pooled (batch, 768).

    The ONNX model takes pre-computed log-mel fbank features as input.
    fbank computation (ta_kaldi.fbank + normalise) is done in Python before
    inference because aten::fft_rfft is not exportable to ONNX.
    EmbeddingExtractor handles this transparently for the BEATs substrate.

    Input  'fbank_features': (batch, T_frames, 128)  float32
    Output 'output':         (batch, 768)             float32
    """
    import torch
    import torch.nn as nn

    m = beats_model  # BEATs instance

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embedding   = m.patch_embedding
            self.layer_norm        = m.layer_norm
            self.post_extract_proj = m.post_extract_proj
            self.dropout_input     = m.dropout_input
            self.encoder           = m.encoder

        def forward(self, fbank_features):
            # fbank_features: (batch, T_frames, 128)
            # Replicate extract_features() without the fbank preprocessing step.
            x = fbank_features.unsqueeze(1)                    # (B, 1, T, 128)
            x = self.patch_embedding(x)                        # (B, C, T', F')
            x = x.reshape(x.shape[0], x.shape[1], -1)         # (B, C, T'*F')
            x = x.transpose(1, 2)                              # (B, T'*F', C)
            x = self.layer_norm(x)
            if self.post_extract_proj is not None:
                x = self.post_extract_proj(x)
            x = self.dropout_input(x)
            x, _ = self.encoder(x, padding_mask=None)
            return x.mean(dim=1)                               # (B, 768)

    return _Wrapper()


def main() -> None:
    args = _parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("ERROR: torch is not installed.  Run: pip install torch")

    BEATs, BEATsConfig = _import_beats(args.beats_src)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper = _make_wrapper(model)
    wrapper.eval()

    # fbank for 10 s @ 16 kHz: (160000 - 400) // 160 + 1 = 998 frames, 128 bins
    dummy_fbank = torch.zeros(1, 998, 128, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to ONNX (opset {args.opset}): {args.output}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_fbank,
            str(args.output),
            input_names=["fbank_features"],
            output_names=["output"],
            dynamic_axes={
                "fbank_features": {0: "batch", 1: "frames"},
                "output":         {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    print("Verifying with onnxruntime ...")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not available — skipping verification)")
        print(f"Done → {args.output}")
        return

    sess = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    out = sess.run(None, {"fbank_features": dummy_fbank.numpy()})
    emb = out[0]
    print(f"  output shape : {emb.shape}    (expected: (1, 768))")
    print(f"  output dtype : {emb.dtype}")
    if emb.shape != (1, 768):
        sys.exit(f"ERROR: unexpected output shape {emb.shape}; expected (1, 768)")
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
