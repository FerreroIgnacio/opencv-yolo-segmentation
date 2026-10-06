"""Segmentación de cajas o personas con YOLO + OpenCV.

Uso:
    python segment.py                      # cajas, sobre el video de ejemplo
    python segment.py --target person      # personas
    python segment.py --source otro.mp4    # corre sobre otro video
    python segment.py --camera             # usa la webcam (índice 0)
    python segment.py --camera 1           # usa la cámara con índice 1
"""

import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, YOLOE

DEFAULT_VIDEO = Path(__file__).parent / "data" / "conveyor_boxes.mp4"

# Presets por objetivo. Los modelos "yoloe-*" son open-vocabulary (las clases se pasan como
# texto); el resto son YOLO-seg entrenados en COCO y se filtran por nombre de clase.
# "labels" mapea cada prompt a la etiqueta que se muestra (los que no están se muestran tal cual).
BOX_PROMPTS = ["storage box", "toolbox", "cardboard box", "box"]
TARGETS = {
    # "Caja" no está en COCO: YOLOE con varios sinónimos (mejoran el recall), mostrados como "box".
    "box": dict(model="yoloe-26l-seg.pt", prompts=BOX_PROMPTS,
                labels={p: "box" for p in BOX_PROMPTS}, conf=0.2),
    # "person" sí está en COCO: un YOLO-seg común es más preciso y más rápido que YOLOE.
    "person": dict(model="yolo26l-seg.pt", prompts=["person"], labels={}, conf=0.35),
    # Ambos con un solo modelo: YOLOE segmenta personas casi igual de bien que el YOLO-seg de COCO.
    "both": dict(model="yoloe-26l-seg.pt", prompts=["person", *BOX_PROMPTS],
                 labels={p: "box" for p in BOX_PROMPTS}, conf=0.2),
}

# Color fijo para personas, así se distinguen de las cajas (que usan la paleta por instancia).
PERSON_COLOR = (255, 0, 255)

HUD_HEIGHT = 28

# Paleta BGR para distinguir instancias.
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmenta y resalta cajas o personas con YOLO.")
    parser.add_argument("--target", choices=TARGETS, default="box",
                        help="Qué segmentar: box (cajas), person (personas) o both (las dos). "
                             "Define modelo, prompts y etiquetas.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--source", type=str, default=str(DEFAULT_VIDEO), help="Ruta a un video o imagen.")
    src.add_argument(
        "--camera", type=int, nargs="?", const=0, default=None,
        help="Usar la cámara como input (índice opcional, por defecto 0).",
    )
    parser.add_argument("--model", default=None,
                        help="Pesos de segmentación (por defecto según --target; ej. yolo26s-seg.pt / "
                             "yoloe-26s-seg.pt para más velocidad).")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Clases a segmentar como texto (por defecto según --target). Con YOLOE vale "
                             "cualquier texto; con YOLO-seg tienen que ser clases de COCO.")
    parser.add_argument("--label", default=None,
                        help="Etiqueta única para todas las detecciones; '' para mostrar el nombre de cada "
                             "clase/prompt. Por defecto según --target.")
    parser.add_argument("--conf", type=float, default=None, help="Umbral de confianza (por defecto según --target).")
    parser.add_argument("--iou", type=float, default=0.5, help="Umbral IoU para NMS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de inferencia.")
    parser.add_argument("--device", default=None,
                        help="cuda:0, cpu, mps... Por defecto usa la GPU NVIDIA si hay una disponible.")
    parser.add_argument("--no-half", action="store_true",
                        help="Desactivar FP16 en GPU (usar FP32; más lento, solo si da problemas).")
    parser.add_argument("--cam-width", type=int, default=1280, help="Ancho pedido a la cámara.")
    parser.add_argument("--cam-height", type=int, default=720, help="Alto pedido a la cámara.")
    parser.add_argument("--output", default=None,
                        help="Guardar el resultado en este .mp4 (por defecto runs/<nombre>_<target>_seg.mp4 para videos).")
    parser.add_argument("--no-show", action="store_true", help="No abrir ventana (útil en servidores).")
    parser.add_argument("--countdown", type=int, default=5,
                        help="Segundos de cuenta regresiva al tocar 'Guardar imagen' (o la tecla s).")
    parser.add_argument("--captures-dir", default="runs/captures", help="Carpeta donde se guardan las capturas.")
    args = parser.parse_args()
    preset = TARGETS[args.target]
    for key in ("model", "prompts", "conf"):
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    if args.label is None:
        args.labels = preset["labels"]
    elif args.label == "":
        args.labels = {}
    else:
        args.labels = {p: args.label for p in args.prompts}
    return args


def load_model(weights: str, prompts: list[str]):
    """Devuelve (modelo, ids de clase a filtrar o None)."""
    if Path(weights).name.startswith("yoloe"):
        model = YOLOE(weights)
        model.set_classes(prompts)
        return model, None
    model = YOLO(weights)
    name_to_id = {name: i for i, name in model.names.items()}
    unknown = [p for p in prompts if p not in name_to_id]
    if unknown:
        raise SystemExit(
            f"{unknown} no son clases de {weights}. Clases válidas: {', '.join(model.names.values())}. "
            "Para texto libre usá un modelo YOLOE (ej. --model yoloe-26l-seg.pt)."
        )
    return model, [name_to_id[p] for p in prompts]


def display_names(result, labels: dict[str, str]) -> list[str]:
    """Etiqueta a mostrar para cada detección."""
    return [labels.get(result.names[int(c)], result.names[int(c)]) for c in result.boxes.cls]


def color_for(i: int, name: str) -> tuple[int, int, int]:
    return PERSON_COLOR if name == "person" else PALETTE[i % len(PALETTE)]


def highlight(frame: np.ndarray, result, names: list[str], alpha: float = 0.45) -> np.ndarray:
    """Pinta máscara semitransparente, contorno y etiqueta para cada instancia."""
    out = frame.copy()
    if result.masks is None or len(result.boxes) == 0:
        return out

    overlay = out.copy()
    polygons = result.masks.xy  # contornos en coordenadas de la imagen original
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for i, poly in enumerate(polygons):
        if len(poly) < 3:
            continue
        color = color_for(i, names[i])
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    for i, (x1, y1, _, _) in enumerate(boxes):
        color = color_for(i, names[i])
        text = f"{names[i]} {confs[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y1, th + 6 + HUD_HEIGHT)  # que no quede tapada por el HUD
        cv2.rectangle(out, (x1, y - th - 6), (x1 + tw + 6, y), color, -1)
        cv2.putText(out, text, (x1 + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def resolve_device(requested: str | None) -> str:
    """Elige el device: el pedido, o la GPU CUDA si existe, o CPU con un aviso."""
    if requested is not None:
        if requested not in ("cpu", "mps") and not torch.cuda.is_available():
            raise SystemExit(
                "Pediste GPU pero PyTorch no ve CUDA. Probablemente tenés el torch de solo-CPU: "
                "reinstalalo con CUDA (ver README, sección GPU)."
            )
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    print(
        "[aviso] No se detectó GPU CUDA, corriendo en CPU. Si tenés una NVIDIA, instalá torch con CUDA "
        "(ver README, sección GPU)."
    )
    return "cpu"


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # En Windows DirectShow abre la cámara mucho más rápido que el backend por defecto (MSMF).
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # menos latencia: no acumular frames viejos
    return cap


def draw_hud(frame: np.ndarray, names: list[str], fps: float) -> None:
    counts = {n: names.count(n) for n in dict.fromkeys(names)}
    detail = "  ".join(f"{n}: {c}" for n, c in counts.items())
    text = f"Detectadas: {len(names)}" + (f" ({detail})" if len(counts) > 1 else "") + f"   FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (0, 0), (tw + 16, HUD_HEIGHT), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


class Capturer:
    """Botón 'Guardar imagen' en pantalla + cuenta regresiva antes de guardar el frame."""

    BUTTON_TEXT = "Guardar imagen (s)"
    MESSAGE_SECONDS = 2.5

    def __init__(self, countdown: int, out_dir: str):
        self.countdown = countdown
        self.out_dir = Path(out_dir)
        self.deadline: float | None = None
        self.message = ""
        self.message_until = 0.0
        self.button = (0, 0, 0, 0)  # x1, y1, x2, y2 del último dibujo

    def start(self) -> None:
        if self.deadline is None:  # si ya hay una cuenta en curso, no la reinicia
            self.deadline = time.monotonic() + self.countdown

    def on_mouse(self, event, x, y, flags, param) -> None:
        x1, y1, x2, y2 = self.button
        if event == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 <= y <= y2:
            self.start()

    def update(self, raw: np.ndarray, vis: np.ndarray) -> None:
        """Si terminó la cuenta, guarda. Se llama con el frame ya resaltado, antes de dibujar la UI."""
        if self.deadline is None or time.monotonic() < self.deadline:
            return
        self.deadline = None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"capture_{stamp}.jpg"
        cv2.imwrite(str(path), vis)
        cv2.imwrite(str(self.out_dir / f"capture_{stamp}_raw.jpg"), raw)  # sin máscaras, útil para dataset
        print(f"Captura guardada en {path}")
        self.message = f"Guardada: {path}"
        self.message_until = time.monotonic() + self.MESSAGE_SECONDS

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Botón abajo a la derecha.
        (tw, th), _ = cv2.getTextSize(self.BUTTON_TEXT, font, 0.6, 2)
        x2, y2 = w - 12, h - 12
        x1, y1 = x2 - tw - 24, y2 - th - 20
        self.button = (x1, y1, x2, y2)
        active = self.deadline is not None
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255) if active else (40, 40, 40), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(frame, self.BUTTON_TEXT, (x1 + 12, y2 - 10), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Número grande en el centro durante la cuenta regresiva.
        if active:
            remaining = max(1, int(np.ceil(self.deadline - time.monotonic())))
            text = str(remaining)
            scale = h / 160
            thick = max(4, int(scale * 3))
            (nw, nh), _ = cv2.getTextSize(text, font, scale, thick)
            org = ((w - nw) // 2, (h + nh) // 2)
            cv2.putText(frame, text, org, font, scale, (0, 0, 0), thick + 6, cv2.LINE_AA)
            cv2.putText(frame, text, org, font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        if time.monotonic() < self.message_until:
            cv2.rectangle(frame, (0, h - 40), (min(w, 12 + 9 * len(self.message)), h - 12), (0, 0, 0), -1)
            cv2.putText(frame, self.message, (8, h - 20), font, 0.5, (80, 255, 80), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    half = device.startswith("cuda") and not args.no_half
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        print(f"Usando GPU: {torch.cuda.get_device_name(device)} ({'FP16' if half else 'FP32'})")
    else:
        print(f"Usando device: {device}")

    model, classes = load_model(args.model, args.prompts)
    print(f"Objetivo: {args.target} | modelo: {args.model} | clases: {', '.join(args.prompts)}")
    predict_kwargs = dict(
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=device, half=half, classes=classes,
        agnostic_nms=True, retina_masks=True, verbose=False,
    )
    # Warm-up: la primera inferencia en GPU es lenta (carga de kernels / cudnn benchmark).
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8), **predict_kwargs)

    if args.camera is not None:
        cap = open_camera(args.camera, args.cam_width, args.cam_height)
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
        output = str(Path("runs") / f"{source_name}_{args.target}_seg.mp4")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    window = f"YOLO segmentation: {args.target} (q / ESC para salir)"
    fps = 0.0
    capturer = Capturer(args.countdown, args.captures_dir)
    if not args.no_show:
        # GUI_NORMAL oculta la barra de Qt (su botón de guardar no tiene cuenta regresiva); usamos el nuestro.
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(window, capturer.on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            result = model.predict(frame, **predict_kwargs)[0]
            dt = time.perf_counter() - t0
            fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

            names = display_names(result, args.labels)
            vis = highlight(frame, result, names)
            draw_hud(vis, names, fps)

            if output:
                if writer is None:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                writer.write(vis)

            if not args.no_show:
                capturer.update(frame, vis)
                ui = vis.copy()
                capturer.draw(ui)
                cv2.imshow(window, ui)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    capturer.start()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Resultado guardado en {output}")
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
