# opencv-yolo-segmentation

Segmentación de instancias de **cajas en un conveyor belt** con YOLO + OpenCV.
Resalta cada caja con una máscara semitransparente, su contorno y una etiqueta con la confianza.
Funciona sobre un video o en vivo desde la cámara.

![demo](docs/demo.jpg)

## Modelo

Se usa **YOLOE** (`yoloe-26l-seg.pt`, de [Ultralytics](https://docs.ultralytics.com/models/yoloe/)),
un YOLO de segmentación *open-vocabulary*: en vez de estar limitado a las 80 clases de COCO
(que no incluye "caja"), recibe las clases como texto. Por defecto se le pasan los prompts
`storage box`, `toolbox`, `cardboard box`, `box` y todas las detecciones se muestran como `box`.

Los pesos (y el encoder de texto MobileCLIP) se descargan automáticamente la primera vez.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# Sobre el video de ejemplo (data/conveyor_boxes.mp4). Guarda el resultado en runs/conveyor_boxes_seg.mp4
python segment.py

# Sobre otro video
python segment.py --source mi_video.mp4

# Desde la cámara (índice 0 por defecto, o el que pases)
python segment.py --camera
python segment.py --camera 1
```

Se abre una ventana con el resultado; `q` o `ESC` para salir.

Opciones útiles:

| Flag | Descripción |
|------|-------------|
| `--camera [N]` | Usar la cámara N como input en lugar del video |
| `--model` | Pesos YOLOE. `yoloe-26s-seg.pt` es bastante más rápido en CPU (menos preciso) |
| `--prompts ...` | Qué segmentar, como texto. Ej: `--prompts box bottle person` |
| `--label` | Etiqueta a mostrar (por defecto `box`; `--label ""` muestra el prompt que matcheó) |
| `--conf` | Umbral de confianza (default 0.2) |
| `--device` | `cpu`, `0` (GPU CUDA), `mps` (Apple Silicon) |
| `--output` | Guardar el video resultante (con `--camera` solo se guarda si lo pasás) |
| `--no-show` | No abrir ventana (servidores / sin display) |

En CPU, `yoloe-26l-seg` corre a ~2 FPS; para la cámara en tiempo real conviene GPU o `--model yoloe-26s-seg.pt`.

## Video de ejemplo

`data/conveyor_boxes.mp4` son frames reales de cajas sobre una cinta transportadora, tomados de
[carrickcheah/Yolo-detection-for-manufacturing](https://github.com/carrickcheah/Yolo-detection-for-manufacturing)
(licencia MIT) y reensamblados a video. Para regenerarlo:

```bash
python scripts/fetch_sample_video.py
```
