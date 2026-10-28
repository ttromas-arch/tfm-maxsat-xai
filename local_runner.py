import os
import subprocess
import sys
import json
import shutil
from pathlib import Path
import torch
import kagglehub

# 1. Comprobación de la GPU
print("="*50)
if torch.cuda.is_available():
    print(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ ATENCIÓN: No se detecta CUDA. Se usará la CPU (irá muy lento).")
print("="*50)

# 2. Rutas locales
WORKING_ROOT = Path(os.getcwd())
PREPARED_DIR = WORKING_ROOT / 'jsrt_prepared'
RESULTS_DIR = WORKING_ROOT / 'tfm_results'

# 3. Descarga de datasets (Se guardan en tu caché local automáticamente)
print("\nDescargando/Verificando datasets JSRT desde Kaggle...")
lung_root = Path(kagglehub.dataset_download("abduzzami/jsrt-247-image-lung-segmentation-mask-dataset"))
metadata_root = Path(kagglehub.dataset_download("raddar/nodules-in-chest-xrays-jsrt"))

# Limpiamos ejecuciones anteriores (opcional, coméntalo si quieres reanudar)
# shutil.rmtree(PREPARED_DIR, ignore_errors=True)
# shutil.rmtree(RESULTS_DIR, ignore_errors=True)
PREPARED_DIR.mkdir(exist_ok=True)

# 4. Preparación de los metadatos (cruzar imágenes con el CSV)
print("\nPreparando anotaciones...")
subprocess.run([
    sys.executable, 'scripts/prepare_kaggle_jsrt.py',
    '--lung-dataset-root', str(lung_root),
    '--metadata-dataset-root', str(metadata_root),
    '--output-dir', str(PREPARED_DIR),
], check=True)

# Leer manifiesto generado
manifest_path = PREPARED_DIR / 'kaggle_jsrt_manifest.json'
with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

# 5. Ejecución del experimento
print("\n🚀 Lanzando TFM MaxSAT (Bucle CEGAR exacto con cardinalidad)...")
command = [
    sys.executable, 'scripts/run_experiments.py',
    '--jsrt-root', str(lung_root),
    '--image-dir', manifest['image_dir'],
    '--lung-mask-dir', manifest['lung_mask_dir'],
    '--nodule-csv', manifest['nodule_csv'],
    '--provenance-manifest', str(manifest_path),
    '--ground-truth-representation', manifest['nodule_gt_representation'],
    '--output-dir', str(RESULTS_DIR),
    '--num-superpixels', '30',
    '--weight-mode', 'cardinality',   # LA CLAVE PARA QUE NO SE CUELGUE
    '--threshold', '0.80',
    '--threshold-mode', 'relative',
    '--max-oracle-calls', '100',
    '--diverse-solutions', '1',
    '--cegar-cut-mode', 'exact',
    '--oracle-batch-size', '16',      # Batch para aprovechar los 12GB de tu 3060
    '--per-sample-timeout-sec', '240',
    '--save-successful-examples', '10', # Extraemos 10 ejemplos con imágenes
    '--seed', '42'
]

subprocess.run(command, check=True)
print(f"\n✅ ¡Terminado! Resultados en: {RESULTS_DIR.absolute()}")