# research-assistant

Proyecto final de **Procesamiento del Lenguaje Natural** (Máster en IA Aplicada, UC3M).

**Grupo:** Antonio Bello, Martín Hernández, Iker Lekuona y Eduardo Jorge Sanjurjo
**Aplicación C1:** Sistema de asistencia en investigación académica sobre papers de arXiv.

| Hito | Contenido | Notebook | Estado |
|---|---|---|---|
| 1 | Corpus, preprocesado con spaCy, consultas de evaluación y baselines dispersos (TF-IDF, BM25) | `milestone_1.ipynb` | Hecho |
| 2 | Sistema RAG: recuperación densa + LLM de pesos abiertos, comparado con el Hito 1 | — | Pendiente |
| 3 | Extensión: modelado de tópicos o LoRA/QLoRA | — | Pendiente |

## Resultados

54 consultas, 114 494 fragmentos, evaluación a nivel de documento:

| Método | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| TF-IDF + coseno | 0,389 | 0,481 | 0,281 |
| BM25 | **0,574** | **0,685** | **0,502** |

BM25 supera a TF-IDF en los cinco temas. Detalle, análisis de errores y limitaciones en [MILESTONE_1.md](MILESTONE_1.md).

## Cómo ejecutarlo

Todos los comandos se ejecutan desde la raíz del repositorio.

### 1. Entorno

Se recomienda Python 3.11 o 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Índice de fragmentos

`data/chunks/` no está en el repositorio: se genera con `src/corpus_builder.py` a partir de `data/corpus.jsonl`, que sí lo está.

* Si ya tenéis los textos extraídos en `data/txt/<tema>/<arxiv_id>.txt`, se construye sin conexión en pocos minutos:

  ```bash
  python src/corpus_builder.py --check   # comprueba que están los 4 000 TXT, sin escribir nada
  python src/corpus_builder.py           # genera data/chunks/
  ```

* Si no los tenéis, este comando descarga los PDF del bucket público de arXiv, extrae su texto y genera el índice. Tarda en torno a una hora:

  ```bash
  python src/corpus_builder.py --download
  ```

### 3. Notebook

Abrir `milestone_1.ipynb` y seleccionar el kernel de `.venv` en Jupyter o VS Code. Los parámetros de ejecución (`SMOKE_TEST`, `REGENERATE_QUERIES`) se explican al principio del notebook.

Las consultas guardadas en `data/queries/` se reutilizan, así que no se vuelven a generar. Los pasos largos se guardan en disco y se reutilizan; para recalcular uno, borrad su archivo (por ejemplo, `data/chunks/chunk_lemmas.jsonl` para repetir la lematización).

### 4. Regenerar las consultas (opcional)

Solo hace falta un LLM con `REGENERATE_QUERIES = True`, y regenerarlas invalida la revisión manual:

* **Ollama en local** (`LLM_BACKEND = 'ollama'`): la celda de arranque del notebook lo instala, lo arranca y descarga el modelo.
* **Servidor de la UC3M** (`LLM_BACKEND = 'uc3m'`): la clave se lee de la variable de entorno `UC3M_API_KEY` (o se pide por teclado). 

### 5. Google Colab

La primera celda de código del notebook instala las dependencias automáticamente. Hay que ajustar `PROJECT_DIR` (sección 0) a la carpeta de Drive que contiene el repositorio completo (`src/` y `data/`, con `data/chunks/` ya generado).

## Estructura del repositorio

```
research-assistant/
├── milestone_1.ipynb                  # notebook del Hito 1                              [git]
├── MILESTONE_1.md                     # resumen de una página del Hito 1                 [git]
├── README.md
├── requirements.txt
├── src/
│   └── corpus_builder.py              # arXiv -> PDF -> texto -> fragmentos              [git]
└── data/
    ├── corpus.jsonl                   # los 4 000 papers del corpus (metadatos)          [git]
    ├── queries/
    │   ├── candidate_queries.jsonl    # las 60 consultas candidatas del LLM              [git]
    │   ├── queries.jsonl              # consultas de evaluación finales (≥ 30)           [git]
    │   └── queries_review.csv         # documentación de la revisión de las consultas    [git]
    ├── chunks/
    │   ├── <tema>.jsonl               # todos los fragmentos válidos de cada paper       [no git]
    │   └── chunk_lemmas.jsonl         # caché de lemas de spaCy                          [no git]
    ├── txt/<tema>/<arxiv_id>.txt      # texto limpio de cada PDF                         [no git]
    ├── pdf/<tema>/<arxiv_id>.pdf      # PDF descargados                                  [no git]
    ├── candidates.jsonl               # candidatos de la API de arXiv                    [no git]
    └── smoke/                         # salidas del modo SMOKE_TEST                      [no git]
```

`<tema>` es `av`, `rl`, `nlp`, `nn` o `dl`. Los archivos `[no git]` se regeneran con `src/corpus_builder.py` o con el notebook.

## Cómo se construye el corpus (`src/corpus_builder.py`)

1. **Candidatos**: la API de arXiv, una petición por tema y por mes (enero 2025 – noviembre 2026).
2. **PDF a texto**: descarga desde `gs://arxiv-dataset` (o `arxiv.org`), extracción con PyMuPDF (40 primeras páginas) y limpieza (Unicode, guiones de fin de línea, sello de arXiv, URLs, correos, bibliografía).
3. **Selección**: 800 papers válidos por tema; cada paper va al primer tema que lo reclama.
4. **Fragmentos**: ventanas de 300 palabras con un 15 % de solape. Se descartan las que no son prosa (menos del 70 % de palabras alfabéticas) y se indexan **todos los fragmentos válidos** de cada paper (unos 114 500 en total).

Con `data/corpus.jsonl` ya presente, el modo por defecto conserva esa selección y solo repite el paso 4 desde `data/txt/`.

## Formato de los archivos principales

**`data/corpus.jsonl`**: un paper por línea, sólo metadatos (`doc_id`, `arxiv_id`, `title`, `abstract`, `published`, `categories`, `topic`, `n_words`, `n_chunks_ok`). El resumen (`abstract`) procede de la API de arXiv.

**`data/chunks/<tema>.jsonl`**: un fragmento por línea.

```json
{"chunk_id": "doc_2609.02056_c00", "doc_id": "doc_2609.02056", "chunk_idx": 0,
 "n_chunks_doc": 35, "topic": "nlp", "n_words": 300, "text": "HyGRAIL: Cost-Aware and ..."}
```

**`data/queries/queries.jsonl`**: el conjunto de test de todo el proyecto (se reutiliza en los Hitos 2 y 3), en el formato del enunciado más el fragmento que se dio al LLM para generar la consulta.

```json
{"qid": "q01", "query": "...", "relevant_ids": ["doc_2504.03160"], "source_chunk": "doc_2504.03160_c23"}
```

## Decisiones de diseño principales

* **Corpus**: 5 temas × 800 papers = 4 000 papers, con una ventana de fechas fija para que sea reproducible.
* **Unidad de indexación**: fragmentos de 300 palabras; el IDF y la longitud media de TF-IDF y BM25 se calculan sobre los fragmentos.
* **Evaluación a nivel de documento**: cada paper recibe la puntuación de su mejor fragmento, lo que permite comparar con el RAG del Hito 2.
* **BM25** con la librería `bm25s` (variante de Lucene, k1 = 1.5, b = 0.75). Sus parámetros no se ajustan sobre las consultas, porque son el conjunto de test.
