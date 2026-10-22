# Explicaciones contrafactuales mínimas en JSRT mediante Partial MaxSAT

Implementación reproducible del TFM **“Explicaciones Contrafactuales Mínimas en
Radiografías de Tórax mediante Partial MaxSAT y Superpíxeles evaluadas sobre Ground
Truth”**. Para cada RX torácica con nódulo, busca el conjunto de menor coste de
superpíxeles SLIC que, al ocultarse, lleva la predicción `Nodule` bajo un umbral.

## Componentes

| Ruta | Responsabilidad |
|---|---|
| `src/data/` | Carga de JSRT, CSV de coordenadas, máscaras de nódulo y SLIC limitado a pulmón. |
| `src/occlusion/` | Oclusiones `zero`, `mean`, `gaussian_blur` e `inpaint` de Telea. |
| `src/models/` | Adaptador de TorchXRayVision DenseNet y oráculo sintético de prueba. |
| `src/explainers/` | RC2, exploración, CEGAR y soluciones diversas MDSOL. |
| `src/evaluation/` | Métricas de validez, minimalidad, localización GT y diversidad. |
| `src/visualization/` | Paneles por paciente, gráficos y tablas académicas. |

## Entorno e instalación

Use **Python 3.10–3.12**. PyTorch y TorchXRayVision aún no tienen soporte binario
uniforme para Python 3.14.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PyTorch requiere un índice específico para CUDA, instálelo primero siguiendo
[pytorch.org](https://pytorch.org/get-started/locally/) y después instale el resto.
La primera creación del modelo XRV puede descargar sus pesos.

## Datos JSRT

La disposición recomendada, también válida bajo `/kaggle/input/...`, es:

```text
data/jsrt/
├── images/                   # PNG/BMP/JPG/TIFF/GIF o JSRT RAW .IMG
├── lung_masks/               # máscara binaria, incluido GIF; mismo identificador
└── nodules.csv               # image_id,x,y,diameter; diámetro opcional
```

El cargador reconoce encabezados alternativos: `id`/`case`, `coord_x`, `coord_y`,
`radius`, `diameter_px`, etc. Las coordenadas están en píxeles fuente y se
transforman geométricamente a 224×224. Para descargar un CSV oficial autorizado use
`--nodule-url URL`; se guarda como `nodules_official.csv`. Revise la licencia de
JSRT y la edición de anotaciones antes de distribuir o publicar resultados.

### Ejecución completa en Kaggle

El cuaderno único [`kaggle_runner.ipynb`](kaggle_runner.ipynb) está preparado para
**Run All**. Clona esta versión del repositorio, instala
`requirements-kaggle.txt`, obtiene o reutiliza los dos inputs públicos, prepara las
anotaciones, ejecuta el benchmark, empaqueta los artefactos en
`/kaggle/working/tfm_maxsat_artifacts.zip` y publica los resultados en una rama
`kaggle-results/<marca-temporal>` del repositorio configurado.

Active **Internet** en las opciones del notebook. Alternativamente, adjunte ambos
datasets a la sesión para evitar descargas:

1. [JSRT 247 Image Lung Segmentation Mask Dataset](https://www.kaggle.com/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset)
   (`abduzzami/jsrt-247-image-lung-segmentation-mask-dataset`, Apache-2.0): contiene
   247 RX bajo `content/jsrt/cxr/` y las **máscaras pulmonares** bajo
   `content/jsrt/masks/`.
2. [Nodules in Chest X-rays (JSRT)](https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt)
   (`raddar/nodules-in-chest-xrays-jsrt`, licencia declarada por Kaggle: *Unknown*):
   contiene `jsrt_metadata.csv`, con los 154 casos `JPCLN`, su centro `x,y`, tamaño
   clínico `size` y estado diagnóstico. El cuaderno conserva SHA-256 y referencias
   de ambas fuentes en `kaggle_jsrt_manifest.json`.

La segunda fuente **no proporciona contornos manuales de nódulo a nivel de píxel**.
Por ello, el pipeline construye una máscara circular a partir de centro y diámetro
clínico (0,175 mm/píxel; referencia JSRT 2048×2048) y marca cada salida con
`ground_truth_representation=circle_from_coordinate_diameter`. IoU, Dice, cobertura
y precisión se deben interpretar frente a esta aproximación geométrica, no como
comparación con una segmentación clínica manual. Las máscaras de `masks/` del primer
dataset se utilizan únicamente para restringir SLIC al pulmón.

Para publicar automáticamente desde Kaggle, cree el secreto del notebook
`GITHUB_TOKEN` con un Fine-Grained Personal Access Token que tenga permiso
**Contents: Read and write** sobre `ttromas-arch/tfm-maxsat-xai`. El token nunca se
escribe en el notebook, resultados ni Git; sin ese secreto la ejecución llega hasta
los artefactos de `/kaggle/working` y termina con una instrucción explícita para
configurarlo. El cuaderno publica en una rama nueva, no sobrescribe `main`.

#### Referencias de datos y uso responsable

- **Base JSRT original:** J. Shiraishi *et al.*, “Development of a digital image
  database for chest radiographs with and without a lung nodule: receiver operating
  characteristic analysis of radiologists' detection of pulmonary nodules”, *AJR
  American Journal of Roentgenology*, 174(1), 71–74, 2000.
  [https://doi.org/10.2214/ajr.174.1.1740071](https://doi.org/10.2214/ajr.174.1.1740071).
- **RX y máscaras pulmonares usadas por el cuaderno:** Abduz Zami,
  [*JSRT 247 Image Lung Segmentation Mask Dataset*](https://www.kaggle.com/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset),
  Kaggle, versión 1 (2024-07-13), licencia declarada **Apache-2.0**.
- **Centros y tamaños de nódulos usados por el cuaderno:** raddar,
  [*Nodules in Chest X-rays (JSRT)*](https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt),
  Kaggle, versión 1 (2020-02-18). Kaggle declara su licencia como **Unknown**;
  compruebe los términos de la fuente original y cite la base JSRT primaria antes de
  redistribuir resultados, imágenes o derivados.

El manifiesto de cada ejecución registra los *slugs* de Kaggle, los conteos de IDs,
la representación del GT y SHA-256 de `jsrt_metadata.csv`. Esto permite detectar un
cambio silencioso de fuente y reproducir exactamente la transformación aplicada.

## Prueba extrema a extremo

```powershell
python scripts/smoke_test.py
python -m pytest -q
```

La prueba no necesita JSRT ni pesos: crea una RX sintética, ejecuta SLIC, las cuatro
oclusiones, RC2/CEGAR/MDSOL, métricas y una figura. Los resultados se guardan en
`outputs/smoke_test/`. Para validar el adaptador médico, posiblemente descargando
pesos, ejecute:

```powershell
python scripts/smoke_test.py --exercise-torchxrayvision
```

## Benchmark clínico

```powershell
python scripts/run_experiments.py `
  --jsrt-root data/jsrt `
  --output-dir outputs/jsrt_run `
  --num-superpixels 50 `
  --weight-mode cardinality `
  --threshold 0.20 `
  --max-images 100 `
  --diverse-solutions 3
```

El modelo predeterminado es `torchxrayvision` con `densenet121-res224-all`. Use
`--threshold-mode relative` para interpretar `--threshold` como fracción retenida
de la probabilidad original; `--weight-mode area` minimiza área en lugar de número
de regiones. Cada resultado se escribe incrementalmente en `results.csv`; un marcador
se escribe tras completar cada método, de modo que al reanudar no se omiten resultados
parciales. Los fallos aparecen en `failures.jsonl`. `--classifier synthetic` sirve
solo para infraestructura, **no produce resultados clínicos**.

## Formulación y límites

La variable `s_i` es verdadera cuando el superpíxel `i` se conserva. RC2 recibe la
cláusula blanda `(s_i,w_i)` por región (`w_i=1` o su área), de forma que falsificarla
equivale a pagar por ocultarla. Ante un candidato positivo del oráculo se añade el
corte CEGAR `∨_{i:I(s_i)=1} ¬s_i`, exigiendo apagar otra región activa. MDSOL bloquea
el modelo de activación completo de cada solución y agrega costes blandos por
reutilizar regiones apagadas.

El clasificador no es proposicional ni necesariamente monótono. La validez se
verifica con inferencia real y la minimalidad se informa respecto a la abstracción
CEGAR y el presupuesto de consultas; no es una garantía clínica ni causal.

## Artefactos

El CSV contiene validez, probabilidades, caída de confianza, cardinalidad, área,
IoU, Dice, cobertura, precisión, correlación Spearman con GT, tiempos RC2 y totales,
llamadas al oráculo, número de cláusulas y diversidad Jaccard. Se generan paneles de
RX/GT/SLIC/cuatro explicaciones, además de tablas Markdown/LaTeX y gráficos box y
violin por método de oclusión.