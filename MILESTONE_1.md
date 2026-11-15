# Hito 1 · Baseline de recuperación dispersa (TF-IDF y BM25)

**Grupo:** Antonio Bello, Martín Hernández, Iker Lekuona y Eduardo Jorge Sanjurjo · **Aplicación C:** asistente de investigación académica · **Commit:** ⟨hash del commit en `main`⟩

## Qué se ha hecho

- **Corpus** (`src/corpus_builder.py`): 4 000 artículos de arXiv publicados entre enero de 2025 y noviembre de 2026, 800 por tema (vehículos autónomos, aprendizaje por refuerzo, PLN, redes neuronales y aprendizaje profundo). Texto completo extraído de los PDF (40 primeras páginas) y limpiado: sello de arXiv, URLs, correos, palabras partidas y bibliografía.
- **Índice:** ventanas de 300 palabras con un 15 % de solape; se descartan las que no son prosa y se indexan **todos los fragmentos válidos** (25 por artículo de mediana).
- **Preprocesado:** spaCy `en_core_web_sm` (lemas de sustantivos, verbos, adjetivos y nombres propios, sin palabras vacías generales ni específicas del dominio), n-gramas con Gensim `Phrases` y filtrado por frecuencia documental (al menos 5 fragmentos y como máximo el 50 %). Vocabulario: 341 238 términos, 97 858 tras el filtrado. Las consultas siguen exactamente el mismo proceso.
- **Consultas** (`data/queries/`): 60 candidatas *known-item* generadas con `llama3.1:8b` ⟨vía Ollama / servidor de la UC3M⟩ a partir del título, el resumen y un fragmento al azar de cada artículo, en cuatro estilos (breve, enunciado, pregunta y detallado). Controles automáticos de longitud, copia literal de cuatro palabras, términos compartidos y juicio del LLM; **12 revisadas a mano** (6 descartadas por copiar el título, ser genéricas o no corresponder al artículo). Quedan **54 consultas** en `queries.jsonl`. Dejamos la revisión en el archivo `queries_review.csv`.
- **Baselines:** TF-IDF con coseno (Gensim: frecuencia bruta × log₂(N/df), norma L2) y **BM25 Okapi propio** con IDF de Lucene, k₁ = 1,5 y b = 0,75, que cuenta una vez cada término distinto de la consulta. IDF y longitud media se calculan sobre fragmentos; cada artículo recibe la puntuación de su mejor fragmento.

## Resultados

Nivel artículo, 54 consultas, índice de 114 494 fragmentos; MRR con corte en 10 y en 100 resultados.

| Método | Recall@5 | Recall@10 | MRR@10 | MRR@100 |
|---|---:|---:|---:|---:|
| TF-IDF + coseno | 0,389 | 0,481 | 0,268 | 0,281 |
| BM25 (k₁ = 1,5, b = 0,75) | **0,574** | **0,685** | **0,495** | **0,502** |

- **BM25 supera a TF-IDF de forma consistente:** +0,204 en Recall@10 (IC 95 % bootstrap [+0,093, +0,333]) y +0,227 en MRR@10 ([+0,119, +0,340]). Es mejor en los cinco temas; PLN es el más fácil (Recall@10 de BM25: 0,917) y vehículos autónomos el más difícil (0,571, con solo 7 consultas).
- **Sensibilidad (exploratoria):** con k₁ entre 0,5 y 2 y b entre 0 y 1, Recall@10 de BM25 se mueve entre 0,648 y 0,685. No se ajustan sobre el test.
- **Índice completo frente al anterior** (10 000 fragmentos, mismas consultas): Recall@10 baja de 0,574 a 0,481 en TF-IDF y de 0,741 a 0,685 en BM25. Más fragmentos dan más cobertura, pero también más competidores, y con la agregación por el máximo los artículos largos tienen más oportunidades de colocar un fragmento arriba.
- **Fallos:** 16 consultas fallan en el top 10 con ambos métodos. En 14 de ellas, la mitad o más de los términos de la consulta no aparecen en su fragmento de origen (*vocabulary mismatch*: la consulta parafrasea, por ejemplo *hidden from view* frente a *occlusion*); en las otras 2, los términos aparecen pero otros artículos los usan con más peso (consultas genéricas o competidores plausibles).

## Problemas encontrados y decisiones de diseño

- **Índice completo frente a muestra de fragmentos.** La primera versión indexaba 10 000 fragmentos (2-3 por artículo), así que la información útil podía quedar fuera del índice. Se pasó a todos los fragmentos válidos para tratar de no perder información, especialmente pensando en las siguientes etapas del proyecto y no limitar las posibles ampliaciones. 
- **Consultas no regeneradas.** Se generaron con el índice anterior (19 de 60 a partir del primer fragmento, que contiene el resumen). Se conservan para mantener la revisión manual y comparar ambos índices con el mismo test; todos sus fragmentos de origen siguen en el índice.
- **Calidad de la extracción.** El texto de los PDF conserva problemas de orden de lectura, autores y afiliaciones, fórmulas y tablas aplanadas. El filtro de prosa y la limpieza los reducen, pero no los eliminan por completo. Además, parte de la bibliografía sigue en el índice cuando el encabezado *References* no se detecta (aparecen n-gramas como `vol_pp` o `international_conference`).
- **Juez poco discriminativo.** Generador y juez son el mismo modelo, que dio coherencia 5 a 52 de las 60 candidatas. La revisión manual se centró en las 12 marcadas por los controles, y algunas consultas aceptadas automáticamente siguen siendo genéricas o cercanas al título.
- **Relevancia incompleta.** Solo el artículo de origen cuenta como relevante. Sin embargo, otros artículos pueden responder bien a la consulta y penalizan las métricas.
- **Desarrollo con muestra (`SMOKE_TEST`).** El pipeline se depuró con 100 artículos por tema antes de ejecutarlo sobre los 4 000 (hemos seguido las recomendaciones del profesor).



## Uso de herramientas de IA generativa

- `llama3.1:8b` ⟨vía Ollama / servidor de la UC3M⟩: generación de las consultas candidatas y valoración de su coherencia y especificidad.
