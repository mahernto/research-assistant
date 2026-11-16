# Hito 1 · Baseline de recuperación dispersa (TF-IDF y BM25)

**Grupo:** Antonio Bello, Martín Hernández, Iker Lekuona y Eduardo Jorge Sanjurjo · **Aplicación C:** asistente de investigación académica · **Commit:** ⟨hash del commit en `main`⟩

## Qué se ha hecho

- **Corpus** (`src/corpus_builder.py`): 4 000 artículos de arXiv publicados entre enero de 2025 y noviembre de 2026, 800 por tema (vehículos autónomos, aprendizaje por refuerzo, PLN, redes neuronales y aprendizaje profundo). Texto completo extraído de los PDF (40 primeras páginas) y limpiado: sello de arXiv, URLs, correos, palabras partidas y bibliografía.
- **Índice:** ventanas de 300 palabras con un 15 % de solape (la última de cada artículo puede tener entre 100 y 300). Se descartan las que no son prosa (menos del 70 % de tokens alfabéticos) y se indexan **todos los fragmentos válidos**: 114 494, 25 por artículo de mediana.
- **Preprocesado:** spaCy `en_core_web_sm` (lemas de sustantivos, verbos, adjetivos y nombres propios, sin palabras vacías generales ni específicas del dominio), n-gramas con Gensim `Phrases` y filtrado por frecuencia documental (al menos 5 fragmentos y como máximo el 50 %). Vocabulario: 341 238 términos, 97 858 tras el filtrado. Las consultas siguen exactamente el mismo proceso, con el modelo de n-gramas y el diccionario aprendidos sobre el corpus.
- **Consultas** (`data/queries/`): 60 candidatas *known-item* generadas con `llama3.1:8b` ⟨vía Ollama / servidor de la UC3M⟩ a partir del título, el resumen y un fragmento al azar de cada artículo, en tres estilos: breve, conversacional (enunciado o pregunta, sin jerga técnica) y detallado. Controles automáticos de longitud, copia literal de cuatro palabras, términos compartidos y juicio del LLM (coherencia y especificidad, con un mínimo por estilo); **12 revisadas a mano** (6 descartadas por copiar el título, ser genéricas o no corresponder al artículo). Quedan **54 consultas** en `queries.jsonl` y la revisión en `queries_review.csv`.
- **Baselines:** TF-IDF con coseno (Gensim: frecuencia bruta × log₂(N/df), norma L2) y **BM25 con la librería `bm25s`** (variante de Lucene, k₁ = 1,5 y b = 0,75), sobre el mismo vocabulario filtrado y contando una vez cada término distinto de la consulta. IDF y longitud media se calculan sobre fragmentos; cada artículo recibe la puntuación de su mejor fragmento.

## Resultados

Nivel artículo, 54 consultas, índice de 114 494 fragmentos. El MRR se calcula sobre el ranking completo, sin corte.

| Método | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| TF-IDF + coseno | 0,389 | 0,481 | 0,281 |
| BM25 (k₁ = 1,5, b = 0,75) | **0,574** | **0,685** | **0,502** |

- **BM25 supera a TF-IDF de forma consistente:** +0,204 en Recall@10 (IC 95 % bootstrap [+0,093, +0,333]) y +0,221 en MRR ([+0,116, +0,330]). Es mejor en los cinco temas; PLN es el más fácil (Recall@10 de BM25: 0,917) y vehículos autónomos el más difícil (0,571, con solo 7 consultas).
- **Precisión de las diferencias:** con 54 consultas y un solo relevante por consulta, cada acierto vale 1/54 ≈ 1,9 puntos de Recall@K. La diferencia entre métodos (37 frente a 26 aciertos en el top 10) es clara; diferencias de uno o dos aciertos no lo son.
- **Sensibilidad (exploratoria):** con k₁ entre 0,5 y 2 y b entre 0 y 1, Recall@10 de BM25 se mueve entre 0,648 y 0,685 y el MRR entre 0,468 y 0,508, es decir, uno o dos aciertos. No se ajustan sobre el test.
- **Índice completo frente al anterior** (10 000 fragmentos, mismas consultas): Recall@10 baja de 0,574 a 0,481 en TF-IDF y de 0,741 a 0,685 en BM25. Más fragmentos dan más cobertura, pero también más competidores, y con la agregación por el máximo los artículos largos tienen más oportunidades de colocar un fragmento arriba.
- **Fragmento de origen:** BM25 recupera el fragmento exacto que originó la consulta en el top 10 solo en el 14,8 % de los casos (TF-IDF: 9,3 %), aunque recupera su artículo en el 68,5 %.

## Análisis de errores

- **16 consultas fallan en el top 10 con ambos métodos** (13 fallan con uno solo y 25 aciertan con los dos).
- **La causa principal es el desajuste de vocabulario.** En 14 de esos 16 casos, la mitad o más de los términos de la consulta no aparecen en su fragmento de origen. El solape medio de la consulta con ese fragmento es 0,34 en los fallos de BM25 frente a 0,46 en los aciertos. Afecta sobre todo a las consultas conversacionales, que por diseño evitan la jerga del artículo: por ejemplo, *change their behavior based on the input* para un artículo sobre *input-dependent neuromodulation*, o *hidden from view* frente a *occlusion*.
- **Consultas genéricas.** En los otros 2 casos los términos aparecen, pero otros artículos los usan con más peso; los competidores recuperados son a menudo respuestas razonables, que la evaluación con un único relevante cuenta como fallos.
- **Limitaciones del diagnóstico.** La pista automática compara con el fragmento de origen, no con todo el artículo, y cuenta como ausentes palabras de relleno del estilo conversacional (`look`, `research`, `way`, `need`), por lo que sobrestima algo el desajuste.

## Problemas encontrados y decisiones de diseño

- **Índice completo frente a muestra de fragmentos.** La primera versión indexaba 10 000 fragmentos (2-3 por artículo), así que la información útil podía quedar fuera del índice. Se pasó a todos los fragmentos válidos para no perder información, especialmente pensando en las siguientes etapas del proyecto.
- **Consultas no regeneradas.** Se generaron con el índice anterior (19 de 60 a partir del primer fragmento, que contiene el resumen). Se conservan para mantener la revisión manual y comparar ambos índices con el mismo test; todos sus fragmentos de origen siguen en el índice.
- **BM25 con `bm25s`.** Se sustituyó la implementación propia por la librería vista en clase. Se comprobó que produce el mismo ranking: su variante de Lucene solo omite el factor constante (k₁ + 1) del numerador. Se usa `get_scores` y no `retrieve`, porque la agregación por artículo necesita la puntuación de todos los fragmentos.
- **Calidad de la extracción.** El texto de los PDF conserva problemas de orden de lectura, autores y afiliaciones, fórmulas y tablas aplanadas. El filtro de prosa y la limpieza los reducen, pero no los eliminan por completo. Además, parte de la bibliografía sigue en el índice cuando el encabezado *References* no se detecta (aparecen n-gramas como `vol_pp` o `international_conference`).
- **Términos perdidos en el preprocesado.** Solo se conservan tokens alfabéticos, así que desaparecen términos con números como `GPT-4` o `3D`, y se forman n-gramas como `end_end` (*end-to-end*, al eliminar *to*).
- **Juez poco discriminativo.** Generador y juez son el mismo modelo, que dio coherencia 5 a 52 de las 60 candidatas. La revisión manual se centró en las 12 marcadas por los controles, y algunas consultas aceptadas automáticamente siguen siendo genéricas o cercanas al título.
- **Relevancia incompleta.** Solo el artículo de origen cuenta como relevante, aunque otros artículos puedan responder bien a la consulta; esto penaliza las métricas.
- **Desarrollo con muestra (`SMOKE_TEST`).** El pipeline se depuró con 100 artículos por tema (incluidos siempre los de las consultas) antes de ejecutarlo sobre los 4 000, siguiendo las recomendaciones del profesor.

## Siguiente paso: necesidad de embeddings

TF-IDF y BM25 son métodos léxicos: cada término es una dimensión independiente y solo puntúan las coincidencias exactas tras lematizar. Por eso no relacionan palabras distintas que significan lo mismo (*car* y *vehicle*, *hidden from view* y *occlusion*, *learn* y *train*), y son precisamente esas paráfrasis las que causan la mayoría de los fallos. Los embeddings representan cada texto como un vector denso que codifica su significado, de modo que textos con sentido parecido quedan cerca aunque no compartan palabras. La recuperación densa del Hito 2 (embeddings y base de datos vectorial sobre los mismos fragmentos) se comparará con estos baselines en las mismas consultas.

## Uso de herramientas de IA generativa

- `llama3.1:8b` ⟨vía Ollama / servidor de la UC3M⟩: generación de las consultas candidatas y valoración de su coherencia y especificidad.
