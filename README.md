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