"""Descarga el video de ejemplo (cajas en un conveyor belt) y lo arma como .mp4.

Fuente: frames de un video real de cajas sobre una cinta transportadora publicados en
https://github.com/carrickcheah/Yolo-detection-for-manufacturing (licencia MIT).
El repo trae 55 frames (uno cada 10 del video original); acá se bajan y se
reensamblan en `data/conveyor_boxes.mp4`.
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

COMMIT = "a533617cd00c599cc8893bd29fabaaf3bf6b85c7"
BASE_URL = (
    "https://raw.githubusercontent.com/carrickcheah/Yolo-detection-for-manufacturing/"
    f"{COMMIT}/data/images/frame_{{:04d}}.jpg"
)
FRAME_IDS = range(0, 550, 10)


def fetch_frame(idx: int) -> np.ndarray:
    with urllib.request.urlopen(BASE_URL.format(idx), timeout=30) as resp:
        data = np.frombuffer(resp.read(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"No se pudo decodificar el frame {idx}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/conveyor_boxes.mp4")
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    for idx in FRAME_IDS:
        print(f"Descargando frame {idx:04d}...", end="\r")
        frame = fetch_frame(idx)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(frame)
    writer.release()
    print(f"\nVideo guardado en {out_path}")


if __name__ == "__main__":
    main()
