#!/usr/bin/env python3
"""Inference module for Hanafuda card detection.

Usage (as a module)
-------------------
    from src.inference import HanafudaDetector

    detector = HanafudaDetector("models/best.pt")
    result = detector.detect("path/to/image.jpg")
    print(result.class_names_detected)

Usage (as a CLI tool)
---------------------
    python -m src.inference --model models/best.pt --image sample.jpg
    python -m src.inference --model models/best.pt --image sample.jpg --output out.jpg
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


@dataclass
class DetectionResult:
    """Structured result returned by ``HanafudaDetector.detect()``."""

    image_path: Path
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    class_names_detected: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    inference_time_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.boxes)

    def __repr__(self) -> str:
        return (
            f"DetectionResult(path='{self.image_path.name}', "
            f"detections={len(self)}, "
            f"inference_time={self.inference_time_ms:.1f}ms)"
        )


class HanafudaDetector:
    """YOLO11n-based detector for Hanafuda playing cards.

    Args:
        model_path: Path to a trained ``.pt`` weight file.
        device: Inference device — ``"cpu"`` or ``"0"`` (first GPU).
        conf_threshold: Minimum confidence score to keep a detection.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        conf_threshold: float = 0.25,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.conf_threshold = conf_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.model_path}\n"
                "Download best.pt from the releases page or train your own model with:\n"
                "  python -m src.train --epochs 100"
            )

        # Lazy import keeps module importable without CUDA
        from ultralytics import YOLO  # noqa: PLC0415

        self._model = YOLO(str(self.model_path))

    def __repr__(self) -> str:
        return (
            f"HanafudaDetector(model='{self.model_path}', "
            f"device='{self.device}', conf={self.conf_threshold})"
        )

    def detect(self, image_path: str | Path) -> DetectionResult:
        """Run inference on a single image and return a structured result.

        Args:
            image_path: Path to the image file.

        Returns:
            A :class:`DetectionResult` with absolute-pixel bounding boxes.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        t0 = time.perf_counter()
        raw = self._model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = raw[0]
        boxes_tensor = result.boxes

        boxes: list[tuple[float, float, float, float]] = []
        labels: list[int] = []
        scores: list[float] = []
        class_names_detected: list[str] = []

        if boxes_tensor is not None and len(boxes_tensor):
            xyxy = boxes_tensor.xyxy.cpu().numpy()
            cls = boxes_tensor.cls.cpu().numpy().astype(int)
            conf = boxes_tensor.conf.cpu().numpy()

            for (x1, y1, x2, y2), label, score in zip(xyxy, cls, conf):
                boxes.append((float(x1), float(y1), float(x2), float(y2)))
                labels.append(int(label))
                scores.append(float(score))
                class_names_detected.append(result.names.get(int(label), str(label)))

        return DetectionResult(
            image_path=image_path,
            boxes=boxes,
            labels=labels,
            class_names_detected=class_names_detected,
            scores=scores,
            inference_time_ms=elapsed_ms,
        )

    def detect_and_draw(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Image.Image:
        """Detect cards and return an annotated PIL Image.

        Uses ``result.plot()`` from ultralytics, which handles letterbox
        inverse transforms internally — guaranteeing correct box positions
        regardless of the original image size or aspect ratio.

        Args:
            image_path: Path to the input image.
            output_path: If provided, save the result to this path.

        Returns:
            Annotated PIL Image with bounding boxes drawn.
        """
        import numpy as np  # noqa: PLC0415

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw = self._model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        # result.plot() handles all coordinate transforms and returns BGR ndarray
        annotated_bgr = raw[0].plot()
        annotated = Image.fromarray(annotated_bgr[..., ::-1])  # BGR → RGB

        if output_path is not None:
            annotated.save(str(output_path))

        return annotated

    def detect_batch(
        self,
        image_paths: Iterable[str | Path],
    ) -> list[DetectionResult]:
        """Run :meth:`detect` on multiple images with a progress bar.

        Note: For large-scale production use, passing a list directly to
        ``model.predict()`` would be faster; this loop is intentionally simple
        for readability and flexibility.
        """
        paths = list(image_paths)
        results: list[DetectionResult] = []
        for path in tqdm(paths, desc="Detecting", unit="img"):
            results.append(self.detect(path))
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Run Hanafuda card detection on an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default=None, help="Save annotated image to this path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    detector = HanafudaDetector(args.model, device=args.device, conf_threshold=args.conf)
    result = detector.detect(args.image)

    print(f"\nDetected {len(result)} card(s) in {result.inference_time_ms:.1f} ms")
    for name, score in zip(result.class_names_detected, result.scores):
        print(f"  {name:<25} {score:.3f}")

    annotated = detector.detect_and_draw(args.image, output_path=args.output)
    if args.output is None:
        annotated.show()
    else:
        print(f"\nSaved annotated image to: {args.output}")


if __name__ == "__main__":
    _cli()
