# research-assistant

Proyecto final de **Procesamiento del Lenguaje Natural** (Máster en IA Aplicada, UC3M) · **Aplicación C: sistema de asistencia en investigación académica** sobre papers de arXiv.

| Hito | Contenido | Notebook | Estado |
|---|---|---|---|
| 1 | Corpus, preprocesado con spaCy, consultas de evaluación y baselines dispersos (TF-IDF, BM25) | `milestone_1_limpio.ipynb` | En curso |
| 2 | Sistema RAG: recuperación densa + LLM de pesos abiertos, comparado con el Hito 1 | — | Pendiente |
| 3 | Extensión: modelado de tópicos o LoRA/QLoRA | — | Pendiente |

## Cómo ejecutarlo

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
```

* `corpus.jsonl` y `data/chunks/*.jsonl` están en el repositorio, así que **no hace falta descargar los PDFs**. `corpus_builder.py` sólo se ejecuta si falta alguno de esos archivos (≈ 1 h).
* La sección 3 usa el LLM del servidor de la UC3M. La clave se lee de la variable de entorno `UC3M_API_KEY` (o se pide por teclado). **No la subáis a git.**
* El notebook se ejecuta **en dos tandas**: hasta la sección 3.1; después se revisan las consultas a mano, se rellena `REVIEW` (3.2) y se ejecuta desde esa celda hasta el final.
* Los pasos largos (lemas de spaCy, consultas candidatas) se guardan en disco y se reutilizan. Para recalcular uno, borrad su archivo.
* En Google Colab, `corpus_builder.py` tiene que estar en la misma carpeta de Drive que el notebook.

## Estructura del repositorio

```
research-assistant/
├── milestone_1_limpio.ipynb   # notebook del Hito 1
├── corpus_builder.py          # construcción del corpus: arXiv -> PDFs -> texto -> fragmentos
├── requirements.txt
├── corpus.jsonl               # los 4 000 papers del corpus (metadatos)           [git]
├── queries.jsonl              # consultas de evaluación finales (≥ 30)            [git]
├── queries_review.csv         # documentación de la revisión de las consultas     [git]
├── MILESTONE_1.md             # resumen de una página del Hito 1                  [git]
└── data/
    ├── chunks/<tema>.jsonl    # los 10 000 fragmentos indexados                   [git]
    ├── candidates_queries.jsonl   # las 60 consultas candidatas del LLM           [git]
    ├── chunk_lemmas.jsonl         # caché de lemas de spaCy                       [no git]
    ├── candidates.jsonl           # candidatos de la API de arXiv                 [no git]
    ├── pdf/<tema>/<arxiv_id>.pdf  # PDFs descargados                              [no git]
    └── txt/<tema>/<arxiv_id>.txt  # texto limpio de cada PDF                      [no git]
```

`<tema>` es `av`, `rl`, `nlp`, `nn` o `dl`. Los archivos `[no git]` sólo hacen falta para reconstruir el corpus desde cero.

## Cómo se construye el corpus (`corpus_builder.py`)

1. **Candidatos**: la API de arXiv, una petición por tema y por mes (enero 2025 – noviembre 2026).
2. **PDF a texto**: descarga desde `gs://arxiv-dataset` (o `arxiv.org`), extracción con PyMuPDF (40 primeras páginas) y limpieza (Unicode, guiones de fin de línea, sello de arXiv, URLs, correos, bibliografía).
3. **Selección**: 800 papers válidos por tema; cada paper va al primer tema que lo reclama.
4. **Fragmentos**: ventanas de 300 palabras con un 15 % de solape, sin las que no son prosa. Cada paper aporta **2 o 3** al índice: siempre el primero (título y resumen) y el resto repartido por el texto.

## Formato de los archivos principales

**`corpus.jsonl`**: un paper por línea, sólo metadatos (`doc_id`, `arxiv_id`, `title`, `abstract`, `published`, `categories`, `topic`, `n_words`, `n_chunks_ok`).

**`data/chunks/<tema>.jsonl`**: un fragmento por línea.

```json
{"chunk_id": "doc_2609.02056_c00", "doc_id": "doc_2609.02056", "chunk_idx": 0,
 "n_chunks_doc": 35, "topic": "nlp", "n_words": 300, "text": "HyGRAIL: Cost-Aware and ..."}
```

**`queries.jsonl`**: el conjunto de test de todo el proyecto (se reutiliza en los Hitos 2 y 3), en el formato del enunciado más el fragmento de origen.

```json
{"qid": "q01", "query": "...", "relevant_ids": ["doc_2504.03160"], "source_chunk": "doc_2504.03160_c23"}
```

## Decisiones de diseño principales

* **Corpus**: 5 temas × 800 papers = 4 000 papers, con una ventana de fechas fija para que sea reproducible.
* **Unidad de indexación**: fragmentos de 300 palabras; el IDF y la longitud media de TF-IDF y BM25 se calculan sobre los fragmentos.
* **Evaluación a nivel de documento**: cada paper recibe la puntuación de su mejor fragmento, lo que permite comparar con el RAG del Hito 2.
* **BM25** propio (Okapi, IDF de Lucene, k1 = 1.5, b = 0.75). Sus parámetros no se ajustan sobre las consultas, porque son el conjunto de test.
