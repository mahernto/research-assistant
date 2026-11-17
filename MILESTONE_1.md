# Hito 1: Baseline de recuperación

**Grupo:** Antonio Bello, Martín Hernández, Iker Lekuona y Eduardo Jorge Sanjurjo

**Aplicación C1:** Sistema de asistencia de investigación académica sobre papers de arXiv.

**Fecha:** 18 de noviembre de 2026

**Repositorio:** [github.com/mahernto/research-assistant](https://github.com/mahernto/research-assistant)

Para ver el detalle del proyecto (código, celdas y salidas), consulta el notebook `milestone_1.ipynb`. Para ejecutarlo o ver detalles adicionales de la estructura del proyecto, consulta el [README.md](README.md).

## Qué se ha hecho

**Corpus** (`src/corpus_builder.py`): 4 000 artículos de arXiv (ene. 2025–nov. 2026), 800 por tema (vehículos autónomos, RL, PLN, redes neuronales, aprendizaje profundo). Texto extraído de los PDF, convertidos a TXT y limpiado.

**Índice:** Fragmentos de 300 palabras con 15 % de solape, se descartan las que no son prosa. Se indexan todos los fragmentos válidos: 114 494, con 25 chunks/paper de mediana.

**Preprocesado:** Lematización con spaCy (`en_core_web_sm`), n-gramas con Gensim `Phrases`, filtrado por frecuencia documental por exceso o mínimo de frecuencia. Vocabulario: 341 238 → 97 858 términos. Las consultas siguen el mismo proceso.

**Consultas** (`data/queries/`): 60 candidatas *known-item* generadas con `llama3.1:8b` (Ollama-local o con la API de UC3M) en tres estilos (breve, conversacional, detallado), con controles automáticos (la LLM valora aceptación de query) y 12 revisadas a mano (6 descartadas). Quedan 54 definitivas en `queries.jsonl`.

**Baselines:** Se usan los métodos de *recuperación dispersa* TF-IDF con coseno (Gensim) y BM25 (`bm25s`, Lucene, k₁=1,5, b=0,75), sobre el mismo vocabulario. Cada artículo recibe la puntuación de su mejor fragmento.

## Resultados

54 consultas, 114 494 fragmentos: Cada artículo recibe la puntuación de su mejor fragmento según TF-IDF o BM25, se ordenan por esa puntuación y se mide en qué posición queda el artículo correcto (Recall@K: ¿está entre los K primeros?; MRR: 1/posición en el ranking completo).

| Método | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| TF-IDF + coseno | 0,389 | 0,481 | 0,281 |
| BM25 | **0,574** | **0,685** | **0,502** |

BM25 gana a TF-IDF en los cinco temas, con una diferencia clara (+0,204 en Recall@10). PLN es el tema más fácil (0,917) y vehículos autónomos el más difícil (0,571, con solo 7 consultas tras los descartes manuales). Ajustar k₁ y b apenas cambia el resultado. BM25 encuentra el fragmento exacto de origen solo el 14,8 % de las veces, pero acierta el artículo el 68,5 %.

**Análisis de errores**: Se revisan las consultas que fallan con ambos métodos para buscar la causa común, siendo las que fallan 16 de 54 consultas. Para el análisis se toma en cuenta el fragmento de origen. En 14 casos, la mitad o más de los términos de la consulta no están en su fragmento de origen (solape 0,34 en fallos vs. 0,46 en aciertos), sobre todo en consultas conversacionales que evitan la jerga técnica (*hidden from view* vs. *occlusion*), ocurriendo un desajuste de vocabulario. En los 2 casos restantes la consulta es genérica y otros artículos puntúan mejor. 
## Problemas encontrados y decisiones de diseño

- **Índice completo vs. muestra:** La primera versión indexaba solo 2-3 fragmentos/artículo (10 000 en total) y podía perder información relevante. Se pasó a indexar todos los fragmentos válidos dando lugar a 114 494.
- **BM25 con `bm25s`:** Sustituye la implementación propia por esta librería. Usa la variante Lucene porque es la que implementa `bm25s` por defecto. Se comprobó que da el mismo ranking que la fórmula clásica de BM25, ya que solo cambia por un factor constante que no afecta al orden de los resultados.
- **Calidad de extracción:** Los PDF conservan ruido (orden de lectura desordenado, fórmulas, bibliografía no detectada). Ese ruido se cuela como términos en el vocabulario y puede hacer que TF-IDF/BM25 den peso a palabras irrelevantes, distorsionando ligeramente los rankings.
- **Preprocesado:** Solo se conservan palabras con letras, así que términos como `GPT-4` o `3D` se pierden o se parten.
- **Juez más permisivo tras naturalizar las consultas:** Al cambiar los prompts para que las consultas sonaran más humanas y menos de plantilla, el juez (el mismo modelo que las genera) se volvió menos exigente y dio coherencia máxima a 52 de 60 candidatas. Es un efecto esperado de ese cambio, no un fallo del sistema de generación.
- **Un único artículo relevante por consulta:** Cada consulta marca como relevante solo el artículo del que se generó, por diseño del test. Esto es una simplificación razonable, ya que evita el sesgo de ampliar la relevancia con los propios métodos evaluados. Pero esto implica que las métricas son una cota inferior, no reconocen otros artículos igual de válidos que el de origen o valoración máxima.
- **Desarrollo con `SMOKE_TEST` (100 artículos/tema)**: Permite depurar y probar todo el pipeline en minutos, antes de lanzarlo sobre los 4 000 artículos, que tarda mucho más y dificulta iterar rápido sobre errores.

## Siguiente paso: *Embeddings* y Sistema RAG

TF-IDF y BM25 son léxicos: Solo puntúan coincidencias exactas de términos, y la mayoría de los fallos vienen de paráfrasis (*car*/*vehicle*, *hidden from view*/*occlusion*). Los embeddings codifican significado en vectores densos, acercando textos similares aunque no compartan palabras. El Hito 2 comparará recuperación densa (embeddings + base vectorial) con estos baselines, sobre las mismas consultas, y usará esa recuperación como base de un sistema RAG: Los fragmentos mejor puntuados se pasarán como contexto a un LLM para que genere una respuesta directa a la pregunta del usuario, en vez de devolver solo una lista de artículos.

## Uso de herramientas de IA generativa

Por un lado, para la creación de *queries*, y para la valoración de su coherencia y especificidad, se ha utilizado `llama3.1:8b` con el uso de Ollama en local, o vía la API de UC3M. Además, se declara que durante la realización del Hito 1 se han usado herramientas de IA como apoyo en el desarrollo del código y la estructura del proyecto.
