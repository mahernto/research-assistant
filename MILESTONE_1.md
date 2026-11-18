# Hito 1: Baseline de recuperación

**Grupo:** Antonio Bello, Martín Hernández, Iker Lekuona y Eduardo Jorge Sanjurjo

**Aplicación C1:** Sistema de asistencia de investigación académica sobre papers de arXiv.

**Fecha:** 18 de noviembre de 2026

**Repositorio:** [github.com/mahernto/research-assistant](https://github.com/mahernto/research-assistant)

Para ver el detalle del proyecto (código, celdas y salidas), consulta el notebook `milestone_1.ipynb`. Para ejecutarlo o ver detalles adicionales de la estructura del proyecto, consulta el [README.md](README.md).

## Qué se ha hecho

**Corpus** (`src/corpus_builder.py`): 4 000 artículos de arXiv (ene. 2025–nov. 2026), 800 por tema (vehículos autónomos, RL, PLN, redes neuronales, aprendizaje profundo), seleccionados por relevancia en la API de arXiv y barajados con `SEED=42`; un artículo compartido por varios temas se asigna al primero de la lista. Texto extraído de las 40 primeras páginas de cada PDF, convertido a TXT y limpiado (sin bibliografía, URLs ni correos). Se exigen ≥ 1 500 palabras y ≥ 3 fragmentos válidos por artículo. La selección queda congelada en `data/corpus.jsonl`; el script documenta cómo se generó y no se vuelve a ejecutar.

**Índice:** Fragmentos de 300 palabras con 15 % de solape; se descartan los fragmentos finales de menos de 100 palabras y los que no son prosa (menos del 70 % de tokens alfabéticos: tablas, fórmulas). Se indexan todos los fragmentos válidos: 114 494, con 25 chunks/paper de mediana. Cada documento tiene id `doc_<arxiv_id>` y cada fragmento `doc_<arxiv_id>_cNN` (el índice de ventana original se conserva, por lo que puede haber huecos). IDF y longitud media se calculan sobre fragmentos, no sobre documentos.

**Preprocesado:** Lematización con spaCy (`en_core_web_sm`), conservando solo tokens alfabéticos de categoría NOUN, VERB, ADJ o PROPN y sin stopwords de spaCy; n-gramas con Gensim `Phrases` (`min_count=5`, `threshold=20`); filtrado de términos demasiado raros o demasiado frecuentes (`no_below=5`, `no_above=0,5`, sobre fragmentos). Vocabulario: 341 238 → 97 858 términos. Las consultas siguen exactamente el mismo proceso.

**Consultas** (`data/queries/`): 60 candidatas *known-item* generadas con `llama3.1:8b` (Ollama-local o con la API de UC3M) en tres estilos (breve, conversacional, detallado). Los fragmentos de origen se eligen al azar (uno por documento, cuota equilibrada por tema, `SEED=42`) y el prompt pide explícitamente no copiar las palabras del texto. Hay controles automáticos (la LLM valora aceptación de query) y 12 revisadas a mano (6 descartadas por triviales o ambiguas; detalle en `queries_review.csv`). Quedan 54 definitivas en `queries.jsonl`, con `qid`, `query`, `relevant_ids` y `source_chunk`.

**Baselines:** Se usan los métodos de *recuperación dispersa* TF-IDF con coseno (Gensim) y BM25 (`bm25s`, variante Lucene, k₁=1,5, b=0,75), sobre el mismo vocabulario. Cada artículo recibe la puntuación de su mejor fragmento y aparece una sola vez en el ranking antes de cortar en K.

## Resultados

54 consultas, 114 494 fragmentos: Cada artículo recibe la puntuación de su mejor fragmento según TF-IDF o BM25, se ordenan por esa puntuación y se mide en qué posición queda el artículo correcto (Recall@K: ¿está entre los K primeros?; MRR: 1/posición en el ranking completo).

| Método | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| TF-IDF + coseno | 0,389 | 0,481 | 0,281 |
| BM25 | **0,574** | **0,685** | **0,502** |

BM25 gana a TF-IDF en los cinco temas, con una diferencia clara (+0,204 en Recall@10). PLN es el tema más fácil (0,917) y vehículos autónomos el más difícil (0,571, con solo 7 consultas tras los descartes manuales). Los parámetros k₁=1,5 y b=0,75 se fijaron de antemano y no se eligieron sobre las consultas; el barrido de k₁ (0,5–2,0) y b del notebook es solo un análisis de sensibilidad, y apenas cambia el resultado. BM25 encuentra el fragmento exacto de origen solo el 14,8 % de las veces, pero acierta el artículo el 68,5 %.

**Análisis de errores**: 16 de 54 consultas fallan con ambos métodos (el artículo relevante no entra en el top-10). En 14 de ellas, la mitad o más de los términos de la consulta no están en su fragmento de origen (solape 0,34 en fallos vs. 0,46 en aciertos), sobre todo en consultas conversacionales que evitan la jerga técnica (*hidden from view* vs. *occlusion*): es un desajuste de vocabulario. En los 2 casos restantes la consulta es genérica y otros artículos puntúan mejor.

## Problemas encontrados y decisiones de diseño

- **Índice completo vs. muestra:** La primera versión indexaba solo 2-3 fragmentos/artículo (10 000 en total) y podía perder información relevante. Se pasó a indexar todos los fragmentos válidos, dando lugar a 114 494.
- **BM25 con `bm25s`:** Sustituye la implementación propia. Usa la variante Lucene (la de `bm25s` por defecto); da el mismo ranking que la fórmula clásica, pues solo cambia por un factor constante que no afecta al orden.
- **Calidad de extracción y corpus:** Los PDF conservan ruido (orden de lectura, fórmulas, bibliografía no detectada) que se cuela en el vocabulario y distorsiona ligeramente los rankings. Además, se usan solo las 40 primeras páginas y los artículos se eligen por relevancia en arXiv, no como muestra aleatoria de arXiv.
- **Preprocesado:** Solo se conservan palabras con letras, así que términos como `GPT-4` o `3D` se pierden o se parten.
- **Juez más permisivo tras naturalizar las consultas:** Al hacer los prompts más humanos y menos de plantilla, el juez (el mismo modelo que genera) se volvió menos exigente y dio coherencia máxima a 52 de 60 candidatas. Es un efecto esperado del cambio, no un fallo del sistema.
- **Un único artículo relevante por consulta:** Solo se marca como relevante el artículo de origen, por diseño del test; evita el sesgo de ampliar la relevancia con los propios métodos evaluados. Implica que las métricas son una cota inferior, pues no reconocen otros artículos igual de válidos.
- **Desarrollo con `SMOKE_TEST` (100 artículos/tema):** Permite depurar el pipeline en minutos antes de lanzarlo sobre los 4 000 artículos.
- **Reproducibilidad:** `data/txt/` y `data/chunks/` no están en el repositorio por tamaño; el README explica cómo regenerarlos. El notebook entregado tiene todas las salidas visibles.

## Siguiente paso: *Embeddings* y Sistema RAG

TF-IDF y BM25 son léxicos: Solo puntúan coincidencias exactas de términos, y la mayoría de los fallos vienen de paráfrasis (*car*/*vehicle*, *hidden from view*/*occlusion*). Los embeddings codifican significado en vectores densos, acercando textos similares aunque no compartan palabras. El Hito 2 comparará recuperación densa (embeddings + base vectorial) con estos baselines, sobre las mismas consultas y los mismos fragmentos (mismos `chunk_id`), y usará esa recuperación como base de un sistema RAG: Los fragmentos mejor puntuados se pasarán como contexto a un LLM para que genere una respuesta directa a la pregunta del usuario, en vez de devolver solo una lista de artículos.

## Uso de herramientas de IA generativa

Por un lado, para la creación de *queries*, y para la valoración de su coherencia y especificidad, se ha utilizado `llama3.1:8b` con el uso de Ollama en local, o vía la API de UC3M. Además, se declara que durante la realización del Hito 1 se han usado herramientas de IA como apoyo en el desarrollo del código y la estructura del proyecto.