"""Segmentación de cajas con YOLO (YOLOE, open-vocabulary) + OpenCV.

Uso:
    python segment.py                      # corre sobre el video de ejemplo
    python segment.py --source otro.mp4    # corre sobre otro video
    python segment.py --camera             # usa la webcam (índice 0)
    python segment.py --camera 1           # usa la cámara con índice 1
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

DEFAULT_VIDEO = Path(__file__).parent / "data" / "conveyor_boxes.mp4"
# YOLOE es open-vocabulary: se le pasan "clases" como texto. Varios sinónimos
# mejoran el recall; en la visualización se muestran todos como "box".
DEFAULT_PROMPTS = ["storage box", "toolbox", "cardboard box", "box"]

# Paleta BGR para distinguir instancias.
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmenta y resalta cajas con YOLOE.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--source", type=str, default=str(DEFAULT_VIDEO), help="Ruta a un video o imagen.")
    src.add_argument(
        "--camera", type=int, nargs="?", const=0, default=None,
        help="Usar la cámara como input (índice opcional, por defecto 0).",
    )
    parser.add_argument("--model", default="yoloe-26l-seg.pt",
                        help="Pesos YOLOE de segmentación (ej. yoloe-26s-seg.pt para más velocidad).")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                        help="Qué segmentar, como texto (ej. --prompts box bottle).")
    parser.add_argument("--label", default="box", help="Etiqueta a mostrar; '' para usar el nombre del prompt.")
    parser.add_argument("--conf", type=float, default=0.2, help="Umbral de confianza.")
    parser.add_argument("--iou", type=float, default=0.5, help="Umbral IoU para NMS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de inferencia.")
    parser.add_argument("--device", default=None, help="cpu, 0 (GPU), mps, ...")
    parser.add_argument("--output", default=None,
                        help="Guardar el resultado en este .mp4 (por defecto runs/<nombre>_seg.mp4 para videos).")
    parser.add_argument("--no-show", action="store_true", help="No abrir ventana (útil en servidores).")
    return parser.parse_args()


def highlight(frame: np.ndarray, result, label: str, alpha: float = 0.45) -> np.ndarray:
    """Pinta máscara semitransparente, contorno y etiqueta para cada instancia."""
    out = frame.copy()
    if result.masks is None or len(result.boxes) == 0:
        return out

    overlay = out.copy()
    polygons = result.masks.xy  # contornos en coordenadas de la imagen original
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for i, poly in enumerate(polygons):
        if len(poly) < 3:
            continue
        color = PALETTE[i % len(PALETTE)]
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    for i, (x1, y1, _, _) in enumerate(boxes):
        color = PALETTE[i % len(PALETTE)]
        name = label or result.names[classes[i]]
        text = f"{name} {confs[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y1, th + 6)
        cv2.rectangle(out, (x1, y - th - 6), (x1 + tw + 6, y), color, -1)
        cv2.putText(out, text, (x1 + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_hud(frame: np.ndarray, count: int, fps: float) -> None:
    text = f"Detectadas: {count}   FPS: {fps:.1f}"
    cv2.rectangle(frame, (0, 0), (260, 28), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    model = YOLOE(args.model)
    model.set_classes(args.prompts)

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"camera{args.camera}"
    else:
        if not Path(args.source).exists():
            raise SystemExit(
                f"No existe {args.source}. Si es el video de ejemplo, corré: python scripts/fetch_sample_video.py"
            )
        cap = cv2.VideoCapture(args.source)
        source_name = Path(args.source).stem
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el input: {args.camera if args.camera is not None else args.source}")

    output = args.output
    if output is None and args.camera is None:
        output = str(Path("runs") / f"{source_name}_seg.mp4")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    window = "YOLOE segmentation (q / ESC para salir)"
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            result = model.predict(
                frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
                agnostic_nms=True, retina_masks=True, verbose=False,
            )[0]
            dt = time.perf_counter() - t0
            fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

            vis = highlight(frame, result, args.label)
            draw_hud(vis, len(result.boxes), fps)

            if output:
                if writer is None:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                writer.write(vis)

            if not args.no_show:
                cv2.imshow(window, vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Resultado guardado en {output}")
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
