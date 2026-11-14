"""Construcción del corpus del Hito 1: arXiv -> PDFs -> texto limpio -> corpus.jsonl + data/chunks/<tema>.jsonl.

El notebook sólo llama a `build()` si faltan `corpus.jsonl` o los fragmentos; los dos están en el repositorio,
así que en una ejecución normal este módulo no descarga nada. También se puede lanzar a mano:

    python corpus_builder.py

Pasos (cada uno se salta si su resultado ya existe en disco):
  1. API de arXiv: candidatos por tema y mes             -> data/candidates.jsonl (checkpoint en .partial)
  2. PDF -> texto limpio, por paper (8 en paralelo)        -> data/pdf/<tema>/<id>.pdf, data/txt/<tema>/<id>.txt
  3. Selección de PAPERS_PER_TOPIC papers válidos por tema -> corpus.jsonl
     (si corpus.jsonl ya existe, sólo se descargan los textos que falten de esos papers)
  4. Fragmentación y selección de TARGET_CHUNKS fragmentos -> data/chunks/<tema>.jsonl
"""
import os, re, json, math, time, random, calendar, threading, unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from tqdm.auto import tqdm

# ================================ PARÁMETROS DEL CORPUS ================================
SEED = 42

# tema -> consulta a la API de arXiv. Cada paper se asigna a un único tema, el primero
# que lo reclama; los temas más específicos van primero y los más genéricos (deep learning) al final.
TOPICS = {
    'autonomous_vehicles':    '(abs:"autonomous driving" OR abs:"autonomous vehicle" OR abs:"autonomous vehicles" OR abs:"self-driving")',
    'reinforcement_learning': 'abs:"reinforcement learning" AND (cat:cs.LG OR cat:cs.AI)',
    'nlp':                    'cat:cs.CL',
    'neural_networks':        'cat:cs.NE AND (abs:"neural network" OR abs:"neural networks")',
    'deep_learning':          'cat:cs.LG AND abs:"deep learning"',
}
# Nombre corto de cada tema (carpetas y archivos de fragmentos)
TOPIC_DIRS = {'autonomous_vehicles': 'av', 'reinforcement_learning': 'rl', 'nlp': 'nlp',
              'neural_networks': 'nn', 'deep_learning': 'dl'}

PAPERS_PER_TOPIC = 800
TARGET_CHUNKS = 10_000            # nº total de fragmentos a indexar
DATE_START, DATE_END = '2025-01', '2026-11'   # ventana fija -> corpus reproducible
OVERSAMPLE = 1.5                  # candidatos extra por tema (duplicados entre temas, PDFs que fallan...)
API_DELAY = 3.0                   # arXiv: como máximo 1 petición cada 3 s
PDF_WORKERS = 8                   # descargas en paralelo desde el bucket público de arXiv
MAX_PDF_MB = 25
MAX_PAGES = 40                    # sólo se lee el principio de los PDFs larguísimos
MIN_DOC_WORDS = 1500              # menos palabras suele indicar un PDF mal extraído
MIN_GOOD_CHUNKS = 3               # un paper válido tiene al menos 3 fragmentos de texto "normal"

# Fragmentación
CHUNK_WORDS = 300                 # el enunciado pide 200-400 palabras
CHUNK_OVERLAP = 0.15              # ... y un solape del 10-20 %
MIN_CHUNK_WORDS = 100             # el resto final del paper, si es más corto, se descarta
MIN_ALPHA_RATIO = 0.70            # fracción mínima de "palabras" alfabéticas (filtra tablas, fórmulas, referencias)

assert set(TOPIC_DIRS) == set(TOPICS), 'TOPIC_DIRS y TOPICS deben tener los mismos temas'


# Utils
def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def chunks_path(chunks_dir, topic):
    return os.path.join(chunks_dir, TOPIC_DIRS[topic] + '.jsonl')


# =============================== 1. Candidatos: API de arXiv ===============================
API_URL = 'https://export.arxiv.org/api/query'
HEADERS = {'User-Agent': 'UC3M-NLP-course-project/1.0 (educational use)'}
NS = {'a': 'http://www.w3.org/2005/Atom', 'x': 'http://arxiv.org/schemas/atom'}
_last_api_call = [0.0]


def api_get(params, retries=6):
    """GET a la API de arXiv respetando >= API_DELAY s entre peticiones (términos de uso).
    Ante HTTP 429/5xx espera mucho más (respeta Retry-After si existe) y reintenta."""
    for attempt in range(retries):
        time.sleep(max(0.0, API_DELAY - (time.time() - _last_api_call[0])))
        _last_api_call[0] = time.time()
        wait = 15 * 2 ** attempt                                # 15, 30, 60, 120, 240, 480 s
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r.text
            ra = r.headers.get('Retry-After', '')
            if ra.isdigit():
                wait = max(wait, int(ra))
            print(f'  HTTP {r.status_code} (intento {attempt + 1}/{retries}); espero {wait} s')
        except requests.RequestException as e:
            print(f'  {type(e).__name__} (intento {attempt + 1}/{retries}); espero {wait} s')
        if attempt < retries - 1:
            time.sleep(wait)
    raise RuntimeError(f'La API de arXiv no responde: {params["search_query"]}')


def parse_feed(xml_text):
    """Atom XML de arXiv -> lista de dicts con los metadatos de cada paper."""
    papers = []
    for e in ET.fromstring(xml_text).findall('a:entry', NS):
        abs_url = e.findtext('a:id', '', NS)
        if '/abs/' not in abs_url:
            continue                                        # entrada de error de la API
        arxiv_id, _, version = abs_url.split('/abs/')[-1].rpartition('v')    # '2401.01234v2'
        primary = e.find('x:primary_category', NS)
        papers.append({
            'doc_id': f'doc_{arxiv_id}',
            'arxiv_id': arxiv_id,
            'version': int(version),
            'title': ' '.join(e.findtext('a:title', '', NS).split()),
            'abstract': ' '.join(e.findtext('a:summary', '', NS).split()),
            'published': e.findtext('a:published', '', NS)[:10],
            'primary_category': primary.get('term') if primary is not None else '',
            'categories': [c.get('term') for c in e.findall('a:category', NS)],
        })
    return papers


def month_windows(start, end):
    """'2025-01', '2026-10' -> pares (AAAAMMDDhhmm, AAAAMMDDhhmm) de cada mes, como pide submittedDate."""
    y, m = map(int, start.split('-'))
    y_end, m_end = map(int, end.split('-'))
    while (y, m) <= (y_end, m_end):
        last_day = calendar.monthrange(y, m)[1]
        yield f'{y}{m:02d}010000', f'{y}{m:02d}{last_day}2359'
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def collect_candidates(partial_file):
    """Una petición por tema y mes. Cada respuesta se guarda en `partial_file` para poder reanudar."""
    windows = list(month_windows(DATE_START, DATE_END))
    per_window = math.ceil(PAPERS_PER_TOPIC * OVERSAMPLE / len(windows))   # candidatos pedidos por tema y mes
    n_calls = len(TOPICS) * len(windows)
    print(f'{n_calls} peticiones a la API (≈ {n_calls * API_DELAY / 60:.0f} min), {per_window} candidatos por tema y mes')
    done = {}                                                              # (tema, inicio de ventana) -> papers
    if os.path.exists(partial_file):
        for rec in read_jsonl(partial_file):
            done[(rec['topic'], rec['start'])] = rec['papers']
        print(f'Reanudando: {len(done)} peticiones ya guardadas en {os.path.basename(partial_file)}')
    seen, rows = set(), []
    for topic, query in TOPICS.items():
        for start, end in tqdm(windows, desc=topic):
            if (topic, start) not in done:
                params = {'search_query': f'({query}) AND submittedDate:[{start} TO {end}]',
                          'start': 0, 'max_results': per_window,
                          'sortBy': 'relevance', 'sortOrder': 'descending'}
                done[(topic, start)] = parse_feed(api_get(params))
                with open(partial_file, 'a', encoding='utf-8') as f:         # guardar ya: si falla, se reanuda aquí
                    f.write(json.dumps({'topic': topic, 'start': start, 'papers': done[(topic, start)]},
                                       ensure_ascii=False) + '\n')
            for p in done[(topic, start)]:
                if (topic, p['doc_id']) not in seen:
                    seen.add((topic, p['doc_id']))
                    rows.append({**p, 'topic': topic})
    return rows


# =============================== 2. Conversión de PDF a texto ===============================
GCS_URL = 'https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf/{yymm}/{arxiv_id}v{version}.pdf'
ARXIV_PDF_URL = 'https://arxiv.org/pdf/{arxiv_id}v{version}'
_arxiv_lock = threading.Lock()
_count_lock = threading.Lock()
_last_arxiv_call = [0.0]
PDF_SOURCE = Counter()           # de dónde viene cada PDF descargado (informativo)
FAILED = {}                      # arxiv_id -> motivo del descarte


def _download(url, throttled=False):
    """Bytes de la URL si es un PDF válido y no demasiado grande; si no, None."""
    try:
        if throttled:                  # arxiv.org: una sola petición cada API_DELAY s
            with _arxiv_lock:
                time.sleep(max(0.0, API_DELAY - (time.time() - _last_arxiv_call[0])))
                r = requests.get(url, headers=HEADERS, timeout=90)
                _last_arxiv_call[0] = time.time()
        else:
            r = requests.get(url, headers=HEADERS, timeout=90)
    except requests.RequestException:
        return None
    if r.status_code == 200 and r.content[:5] == b'%PDF-' and len(r.content) <= MAX_PDF_MB * 1e6:
        return r.content
    return None


def fetch_pdf(paper, pdf_dir):
    """Descarga (o reutiliza) el PDF de un paper: primero el bucket gs://arxiv-dataset, si no arxiv.org.
    Devuelve su ruta, o None si no se ha podido."""
    topic_dir = os.path.join(pdf_dir, TOPIC_DIRS[paper['topic']])
    os.makedirs(topic_dir, exist_ok=True)
    path = os.path.join(topic_dir, paper['arxiv_id'] + '.pdf')
    if os.path.exists(path):
        return path
    data, source = None, None
    for v in dict.fromkeys([paper['version'], 1]):      # versión que indica la API y, si no está, la v1
        data = _download(GCS_URL.format(yymm=paper['arxiv_id'][:4], arxiv_id=paper['arxiv_id'], version=v))
        if data:
            source = 'bucket gs://arxiv-dataset'
            break
    if data is None:
        data = _download(ARXIV_PDF_URL.format(arxiv_id=paper['arxiv_id'], version=paper['version']), throttled=True)
        source = 'arxiv.org (lento)'
    if data is None:
        return None
    with open(path + '.tmp', 'wb') as f:
        f.write(data)
    os.replace(path + '.tmp', path)
    with _count_lock:
        PDF_SOURCE[source] += 1
    return path


def extract_text(pdf_path):
    """Texto de las primeras MAX_PAGES páginas del PDF (PyMuPDF)."""
    import pymupdf
    with pymupdf.open(pdf_path) as pdf:
        return '\n'.join(pdf[i].get_text('text', sort=True) for i in range(min(len(pdf), MAX_PAGES)))


ARXIV_STAMP = re.compile(r'arXiv:\s*\d{4}\.\d{4,5}(v\d+)?\s*\[[^\]]+\]\s*\d{1,2}\s+[A-Za-z]{3}\s+\d{4}')
URL_RE = re.compile(r'(https?://|www\.)\S+')
EMAIL_RE = re.compile(r'\S+@\S+\.\S+')
REF_HEADING = re.compile(r'\n\s*(?:\d+\.?\s+)?(?:references|bibliography)\s*\n', re.I)


def clean_pdf_text(raw):
    """Texto crudo de un PDF -> una sola línea de texto sin referencias, URLs, correos ni guiones de fin de línea."""
    text = unicodedata.normalize('NFKC', raw)
    text = ARXIV_STAMP.sub(' ', text)
    heads = [m.start() for m in REF_HEADING.finditer(text) if m.start() > 0.4 * len(text)]
    if heads:
        text = text[:heads[-1]]                          # fuera la bibliografía (y lo que venga después)
    text = re.sub(r'(\w)-\n(?=[a-z])', r'\1', text)      # rein-\nforcement -> reinforcement
    text = URL_RE.sub(' ', text)
    text = EMAIL_RE.sub(' ', text)
    return ' '.join(text.split())


def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP, min_words=MIN_CHUNK_WORDS):
    """Ventanas de `size` palabras con solape `overlap`. El resto final, si es muy corto, se descarta."""
    words = text.split()
    step = max(1, round(size * (1 - overlap)))
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if chunks and len(piece) < min_words:           # casi todo es solape con el fragmento anterior
            break
        chunks.append(' '.join(piece))
        if start + size >= len(words):
            break
    return chunks


WORD_RE = re.compile(r"^\(?[A-Za-z][A-Za-z'\-]*[.,;:!?)]*$")


def alpha_ratio(chunk):
    """Fracción de 'palabras' formadas sólo por letras (con puntuación final): baja en tablas, fórmulas y referencias."""
    words = chunk.split()
    return sum(bool(WORD_RE.match(w)) for w in words) / max(1, len(words))


def process_paper(paper, pdf_dir, text_dir):
    """PDF -> texto limpio (cacheado en text_dir). Devuelve el registro del paper o None si no sirve."""
    topic_dir = os.path.join(text_dir, TOPIC_DIRS[paper['topic']])
    os.makedirs(topic_dir, exist_ok=True)
    txt_path = os.path.join(topic_dir, paper['arxiv_id'] + '.txt')
    try:
        if os.path.exists(txt_path):
            with open(txt_path, encoding='utf-8') as f:
                text = f.read()
        else:
            pdf_path = fetch_pdf(paper, pdf_dir)
            if pdf_path is None:
                FAILED[paper['arxiv_id']] = 'descarga'
                return None
            text = clean_pdf_text(extract_text(pdf_path))
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
    except Exception as e:                               # PDF corrupto, cifrado...
        FAILED[paper['arxiv_id']] = f'{type(e).__name__}: {e}'
        return None
    n_words = len(text.split())
    n_good = sum(alpha_ratio(c) >= MIN_ALPHA_RATIO for c in chunk_text(text))
    if n_words < MIN_DOC_WORDS or n_good < MIN_GOOD_CHUNKS:
        FAILED[paper['arxiv_id']] = f'texto insuficiente ({n_words} palabras, {n_good} fragmentos válidos)'
        return None
    return {**paper, 'n_words': n_words, 'n_chunks_ok': n_good}


# =============================== 3. Selección de papers ===============================
def select_papers(candidates, pdf_dir, text_dir):
    """Por tema (de más a menos específico), candidatos en orden aleatorio hasta reunir PAPERS_PER_TOPIC válidos."""
    claimed, selected = set(), []
    work = lambda p: process_paper(p, pdf_dir, text_dir)
    for topic in TOPICS:
        pool = [p for p in candidates if p['topic'] == topic and p['doc_id'] not in claimed]
        random.Random(SEED).shuffle(pool)
        good, i = [], 0
        with ThreadPoolExecutor(PDF_WORKERS) as ex:
            while len(good) < PAPERS_PER_TOPIC and i < len(pool):
                batch = pool[i:i + (PAPERS_PER_TOPIC - len(good))]
                i += len(batch)
                good += [r for r in tqdm(ex.map(work, batch), total=len(batch), desc=topic) if r]
        good = good[:PAPERS_PER_TOPIC]
        claimed |= {p['doc_id'] for p in good}
        selected += good
        if len(good) < PAPERS_PER_TOPIC:
            print(f'{topic}: sólo {len(good)} papers válidos de {PAPERS_PER_TOPIC} '
                  '(subid OVERSAMPLE o ampliad la ventana de fechas)')
    return selected


# =============================== 4. Índice de fragmentos ===============================
def pick_positions(n, k):
    """k posiciones equiespaciadas de range(n) que siempre incluyen la primera."""
    if k >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k + 1)[:-1].round().astype(int).tolist()))


def build_chunks(docs, text_dir, target, seed):
    """Fragmenta cada paper y elige `target` fragmentos: 2 o 3 por paper, siempre el primero y el resto
    repartido a intervalos regulares."""
    rng = random.Random(seed)
    per_doc = {}                                  # doc_id -> [(posición en el paper, texto)] de fragmentos válidos
    for d in docs:
        with open(os.path.join(text_dir, TOPIC_DIRS[d['topic']], d['arxiv_id'] + '.txt'), encoding='utf-8') as f:
            pieces = chunk_text(f.read())
        per_doc[d['doc_id']] = [(i, c) for i, c in enumerate(pieces) if alpha_ratio(c) >= MIN_ALPHA_RATIO]

    base, extra = divmod(target, len(per_doc))
    bonus = set(rng.sample(list(per_doc), extra))                 # estos papers aportan un fragmento más
    chosen = {d: pick_positions(len(v), base + (d in bonus)) for d, v in per_doc.items()}
    # Cada paper tiene >= MIN_GOOD_CHUNKS fragmentos válidos, así que siempre se llega exactamente a `target`
    assert sum(len(v) for v in chosen.values()) == target, 'algún paper tiene menos fragmentos que su presupuesto'

    topic_of = {d['doc_id']: d['topic'] for d in docs}
    rows = []
    for doc_id, kept in per_doc.items():
        for j in chosen[doc_id]:
            idx, text = kept[j]
            rows.append({'chunk_id': f'{doc_id}_c{idx:02d}', 'doc_id': doc_id, 'chunk_idx': idx,
                         'n_chunks_doc': len(kept), 'topic': topic_of[doc_id],
                         'n_words': len(text.split()), 'text': text})
    return rows


# ==================================== pipeline ====================================
def build(project_dir='.'):
    """Ejecuta los pasos que falten y deja corpus.jsonl y data/chunks/<tema>.jsonl en disco."""
    data_dir = os.path.join(project_dir, 'data')
    pdf_dir, text_dir = os.path.join(data_dir, 'pdf'), os.path.join(data_dir, 'txt')
    chunks_dir = os.path.join(data_dir, 'chunks')
    candidates_file = os.path.join(data_dir, 'candidates.jsonl')
    manifest_file = os.path.join(project_dir, 'corpus.jsonl')
    os.makedirs(chunks_dir, exist_ok=True)

    if os.path.exists(manifest_file):
        # Corpus ya elegido: sólo hay que asegurarse de tener el texto de sus papers
        print('Reutilizando corpus.jsonl (se descargan los textos que falten)...')
        with ThreadPoolExecutor(PDF_WORKERS) as ex:
            docs = [r for r in tqdm(ex.map(lambda p: process_paper(p, pdf_dir, text_dir), read_jsonl(manifest_file)),
                                    desc='corpus') if r]
    else:
        if os.path.exists(candidates_file):
            candidates = read_jsonl(candidates_file)
        else:
            candidates = collect_candidates(candidates_file + '.partial')
            write_jsonl(candidates_file, candidates)
        print('Candidatos por tema:', dict(Counter(c['topic'] for c in candidates)))
        docs = select_papers(candidates, pdf_dir, text_dir)
        write_jsonl(manifest_file, docs)

    print(f'{len(docs)} papers · PDFs descargados: {dict(PDF_SOURCE) or "(ya estaban en disco)"} · '
          f'descartados: {len(FAILED)}', Counter(v.split(' (')[0].split(':')[0] for v in FAILED.values()).most_common(5))

    rows = build_chunks(docs, text_dir, TARGET_CHUNKS, SEED + 1)
    for t in TOPICS:
        write_jsonl(chunks_path(chunks_dir, t), [r for r in rows if r['topic'] == t])
    print(f'{len(rows)} fragmentos guardados en {chunks_dir}')


if __name__ == '__main__':
    build()
