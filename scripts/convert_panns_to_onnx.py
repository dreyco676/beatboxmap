#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert a PANNs CNN14 checkpoint (.pth) to ONNX.

Prerequisites
-------------
1. Clone the PANNs source tree (model architecture code):

       git clone https://github.com/qiuqiangkong/audioset_tagging_cnn
       # The model code lives at audioset_tagging_cnn/pytorch/

2. Download the CNN14 checkpoint from Zenodo record 3987831:
       https://zenodo.org/record/3987831
   File to download: CNN14_mAP=0.431.pth  (or CNN14_16k_mAP=0.438.pth
   for the 16 kHz native variant — preferred for this project).

3. Install torch and onnx into the project venv:

       pip install torch onnx

Note on sample rates
--------------------
The standard CNN14 checkpoint was trained on 32 kHz audio.
A 16 kHz native variant (CNN14_16k_*) also exists on the Zenodo record
and is preferred here because EmbeddingExtractor expects 16 kHz input
(input_length = 16 000 samples).  If you use the 32 kHz checkpoint, pass
--sample-rate 32000 --input-length 32000 and update embeddings.py to match.

Usage
-----
    # 16 kHz variant (recommended):
    python scripts/convert_panns_to_onnx.py \\
        --checkpoint /path/to/CNN14_16k_mAP=0.438.pth \\
        --panns-src  /path/to/audioset_tagging_cnn/pytorch \\
        --output     models/panns_cnn14_16k.onnx

    # 32 kHz variant:
    python scripts/convert_panns_to_onnx.py \\
        --checkpoint /path/to/CNN14_mAP=0.431.pth \\
        --panns-src  /path/to/audioset_tagging_cnn/pytorch \\
        --output     models/panns_cnn14_32k.onnx \\
        --sample-rate 32000 --input-length 32000

The exported model matches the interface expected by EmbeddingExtractor
(substrate_id="panns_cnn14"):
    input_names  : ["waveform"]
    output_names : ["embedding"]
    output shape : (batch, 2048)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert PANNs CNN14 checkpoint to ONNX"
    )
    p.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to the CNN14 .pth checkpoint file",
    )
    p.add_argument(
        "--panns-src", required=True, type=Path,
        help=(
            "Path to the audioset_tagging_cnn/pytorch/ directory "
            "(contains models.py)"
        ),
    )
    p.add_argument(
        "--output", default=Path("models/panns_cnn14_16k.onnx"), type=Path,
        help="Destination ONNX file (default: models/panns_cnn14_16k.onnx)",
    )
    p.add_argument(
        "--sample-rate", default=16_000, type=int,
        help="Sample rate the checkpoint was trained on (default: 16000)",
    )
    p.add_argument(
        "--input-length", default=16_000, type=int,
        help="Input audio length in samples (default: 16000 = 1 s @ 16 kHz)",
    )
    p.add_argument(
        "--classes-num", default=527, type=int,
        help="Number of AudioSet classes (default: 527)",
    )
    p.add_argument("--opset", default=17, type=int, help="ONNX opset version")
    return p.parse_args()


def _load_cnn14(panns_src: Path, checkpoint: Path, sample_rate: int, classes_num: int):
    """Load CNN14 from the PANNs source tree and return an eval-mode model."""
    import torch

    sys.path.insert(0, str(panns_src.resolve()))
    try:
        from models import Cnn14  # type: ignore[import]
    except ImportError as exc:
        sys.exit(
            f"ERROR: Cannot import Cnn14 from {panns_src!r}: {exc}\n"
            "Clone the repo:  "
            "git clone https://github.com/qiuqiangkong/audioset_tagging_cnn\n"
            "Then pass:       --panns-src /path/to/audioset_tagging_cnn/pytorch"
        )

    # 16 kHz checkpoint: window_size=512, hop_size=160, fmax=8000.
    # 32 kHz checkpoint: window_size=1024, hop_size=320, fmax=14000.
    if sample_rate == 16_000:
        window_size, hop_size, fmax = 512, 160, 8000
    else:
        window_size, hop_size, fmax = 1024, 320, 14000

    model = Cnn14(
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=64,
        fmin=50,
        fmax=fmax,
        classes_num=classes_num,
    )

    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    # PANNs checkpoints may be wrapped under a "model" key
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


class _CNN14Wrapper:
    """Thin torch.nn.Module that exports waveform -> embedding (2048-dim)."""

    def __new__(cls, cnn14_model):
        import torch.nn as nn

        class _Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, waveform):
                # CNN14.forward returns a dict with keys
                # 'clipwise_output' and 'embedding'
                out = self.model(waveform)
                return out["embedding"]  # (batch, 2048)

        return _Wrapper(cnn14_model)


def main() -> None:
    args = _parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("ERROR: torch is not installed. Run: pip install torch")

    model = _load_cnn14(
        args.panns_src, args.checkpoint, args.sample_rate, args.classes_num
    )
    wrapper = _CNN14Wrapper(model)
    wrapper.eval()

    L = args.input_length
    dummy_waveform = torch.zeros(1, L, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to ONNX (opset {args.opset}): {args.output}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_waveform,
            str(args.output),
            input_names=["waveform"],
            output_names=["embedding"],
            dynamic_axes={
                "waveform":  {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,   # legacy TorchScript exporter; dynamo can't map hann_window
        )

    # --- verify with onnxruntime ---
    print("Verifying with onnxruntime...")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not available — skipping verification)")
        print(f"Done → {args.output}")
        return

    import numpy as np

    sess = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    out = sess.run(None, {"waveform": dummy_waveform.numpy()})
    emb = out[0]
    print(f"  output shape : {emb.shape}    (expected: (1, 2048))")
    print(f"  output dtype : {emb.dtype}")
    if emb.shape != (1, 2048):
        sys.exit(f"ERROR: unexpected output shape {emb.shape}; expected (1, 2048)")
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
