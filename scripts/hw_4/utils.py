"""Shared utilities for BERTopic experiment framework.

Key design decisions:
- embedding_model=None ALWAYS; precomputed embeddings passed via fit_transform
- Single sample_size everywhere (12,000 docs)
- Same metric computation across all experiments
- Diversity uses sum(len(words)) denominator (not top_n_words * topic_count)
"""

import gc
import json
import logging
import os
import pickle
import re
import tempfile
import time
import urllib.request
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.backend._sklearn import SklearnEmbedder
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
from razdel import tokenize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import make_pipeline
from umap import UMAP

logging.getLogger("gensim").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# Regexes for preprocessing
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b", re.UNICODE)
SPACE_RE = re.compile(r"\s+", re.UNICODE)
WORD_RE = re.compile(r"^[a-zа-яё]+$", re.IGNORECASE)

# Lazy-initialized singletons
_MORPH = None
_RUSSIAN_STOPWORDS = None
_REPRESENTATION_BACKENDS = {}


def _get_morph():
    global _MORPH
    if _MORPH is None:
        _MORPH = MorphAnalyzer()
    return _MORPH


def get_stopwords():
    global _RUSSIAN_STOPWORDS
    if _RUSSIAN_STOPWORDS is None:
        nltk.download("stopwords", quiet=True)
        english_stopwords = set(stopwords.words("english"))
        latin_news_noise = {
            "afp",
            "associated",
            "bbc",
            "cnn",
            "daily",
            "news",
            "num",
            "of",
            "post",
            "press",
            "reuters",
            "ru",
        }
        _RUSSIAN_STOPWORDS = sorted(
            set(stopwords.words("russian"))
            | english_stopwords
            | latin_news_noise
            | {"__url__", "__num__", "это", "который", "которые", "очень"}
        )
    return _RUSSIAN_STOPWORDS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_URL = "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz"


def get_sample_cache_path(root: Path, sample_size: int, seed: int) -> Path:
    return root / "tmp" / "bertopic_search" / f"sample_n{sample_size}_seed{seed}.pkl"


def _atomic_replace_bytes(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            write_fn(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_pickle(df: pd.DataFrame, path: Path) -> None:
    _atomic_replace_bytes(path, lambda handle: pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL))


def atomic_write_npy(array: np.ndarray, path: Path) -> None:
    _atomic_replace_bytes(path, lambda handle: np.save(handle, array))


def atomic_write_json(payload: dict, path: Path) -> None:
    _atomic_replace_bytes(path, lambda handle: handle.write(json.dumps(payload, indent=2, ensure_ascii=False).encode()))


def normalize_text(text: str) -> str:
    """Lowercase, replace URLs and numbers with placeholders, collapse whitespace."""
    text = text.lower()
    text = URL_RE.sub(" __url__ ", text)
    text = NUM_RE.sub(" __num__ ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def lemmatize_text(text: str) -> str:
    """Normalize then lemmatize Russian text using pymorphy3."""
    morph = _get_morph()
    normalized = normalize_text(text)
    parts = []
    for item in tokenize(normalized):
        token = item.text
        if WORD_RE.fullmatch(token) is None:
            parts.append(token)
            continue
        parts.append(morph.parse(token)[0].normal_form)
    return " ".join(parts)


def load_sample(root: Path, sample_size: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """Load or create the requested sample. Cache keys are versioned by size and seed."""
    from corus import load_lenta

    cache_path = get_sample_cache_path(root, sample_size, seed)
    data_path = root / "data" / "lenta-ru-news.csv.gz"

    if cache_path.exists():
        return pd.read_pickle(cache_path).copy().reset_index(drop=True)

    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, data_path)

    rows: list[dict[str, str]] = []
    for record in load_lenta(data_path):
        title = (record.title or "").strip()
        text = (record.text or "").strip()
        topic = (record.topic or "").strip()
        if not text or not topic:
            continue
        rows.append({"title": title, "text": text, "topic": topic})

    df = (
        pd.DataFrame(rows)
        .sample(n=sample_size, random_state=seed)
        .reset_index(drop=True)
    )
    df["title_only_light"] = df["title"].fillna("").map(normalize_text)
    df["text_only_light"] = df["text"].map(normalize_text)
    df["title_text_light"] = (df["title_only_light"] + " " + df["text_only_light"]).str.strip()
    atomic_write_pickle(df, cache_path)
    return df.reset_index(drop=True)


def preprocess_texts(
    df: pd.DataFrame,
    root: Path | None = None,
    sample_size: int = 12_000,
    seed: int = 42,
    require_lemma: bool = False,
) -> pd.DataFrame:
    """Ensure requested preprocessing columns exist and persist them back to the sample cache."""
    if "title_only_light" not in df.columns:
        df["title_only_light"] = df["title"].fillna("").map(normalize_text)
    if "text_only_light" not in df.columns:
        df["text_only_light"] = df["text"].map(normalize_text)
    if "title_text_light" not in df.columns:
        df["title_text_light"] = (df["title_only_light"] + " " + df["text_only_light"]).str.strip()

    if require_lemma:
        changed = False
        if "text_only_lemma" not in df.columns:
            log.info("Computing text_only_lemma column (one-time cost)...")
            df["text_only_lemma"] = df["text"].map(lemmatize_text)
            changed = True
        if "title_only_lemma" not in df.columns:
            log.info("Computing title_only_lemma column (one-time cost)...")
            df["title_only_lemma"] = df["title"].fillna("").map(lemmatize_text)
            changed = True
        if "title_text_lemma" not in df.columns:
            df["title_text_lemma"] = (df["title_only_lemma"] + " " + df["text_only_lemma"]).str.strip()
            changed = True

        if changed and root is not None:
            cache_path = get_sample_cache_path(root, sample_size, seed)
            atomic_write_pickle(df, cache_path)
            log.info("Cached lemma column back to %s", cache_path.name)

    return df


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _encode_sentence_transformer(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device_override: str | None = None,
) -> np.ndarray:
    """Encode texts with a sentence-transformer model. Retries with smaller batch on GPU OOM."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = device_override or (
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = SentenceTransformer(model_name, device=device)
    try:
        effective_batch = min(batch_size, 16) if device == "mps" else batch_size
        embeddings = np.asarray(
            model.encode(
                texts,
                batch_size=effective_batch,
                show_progress_bar=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        if not np.isfinite(embeddings).all():
            raise ValueError(f"Non-finite embeddings produced by {model_name}")
        return embeddings
    except Exception as exc:
        release_torch_memory(device=device)
        should_retry = device != "cpu" and ("Non-finite" in str(exc) or "out of memory" in str(exc).lower())
        if not should_retry:
            raise

        retry_batch = min(batch_size, 4)
        log.warning("Encoder %s failed on %s (%s). Retrying batch=%s", model_name, device, exc, retry_batch)
        retry_model = SentenceTransformer(model_name, device=device)
        try:
            return np.asarray(
                retry_model.encode(
                    texts,
                    batch_size=retry_batch,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        finally:
            release_torch_memory(device=device)


def _encode_tfidf_svd(
    texts: list[str],
    n_components: int,
    tfidf_ngram_range: list[int],
    tfidf_min_df: int,
    tfidf_max_df: float,
    tfidf_max_features: int,
    tfidf_sublinear_tf: bool,
    seed: int,
) -> np.ndarray:
    """Build TF-IDF matrix and project it with TruncatedSVD."""
    stop_words = get_stopwords()
    tfidf = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=tuple(tfidf_ngram_range),
        min_df=tfidf_min_df,
        max_df=tfidf_max_df,
        max_features=tfidf_max_features,
        token_pattern=r"(?u)\b[\wёЁ-]+\b",
        sublinear_tf=tfidf_sublinear_tf,
    )
    matrix = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    return svd.fit_transform(matrix).astype(np.float32)


def release_torch_memory(device: str | None = None) -> None:
    """Force Python GC and flush MPS/CUDA caches."""
    gc.collect()

    try:
        import torch
    except ImportError:
        return

    resolved_device = device or (
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )

    if resolved_device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif resolved_device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_embeddings(
    texts: list[str],
    encoder_key: str,
    encoder_cfg: dict,
    cache_dir: Path,
    seed: int = 42,
    text_column: str = "text_only_light",
) -> np.ndarray:
    """Compute or load cached embeddings. Key fix: no backend encoder confound."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_stem = f"{encoder_key}_{text_column}_{len(texts)}"
    if encoder_cfg["type"] == "tfidf_svd":
        cache_stem = f"{cache_stem}_seed{seed}"
    cache_path = cache_dir / f"{cache_stem}.npy"

    if cache_path.exists():
        embeddings = np.load(cache_path)
        if np.isfinite(embeddings).all() and embeddings.shape[0] == len(texts):
            log.info(f"Loaded cached embeddings: {cache_path.name}")
            return embeddings
        log.warning(f"Corrupted cache, recomputing: {cache_path.name}")
        cache_path.unlink()

    enc_type = encoder_cfg["type"]
    log.info(f"Computing embeddings: {encoder_key} ({enc_type}) for {len(texts)} texts")

    if enc_type == "sentence_transformer":
        embeddings = _encode_sentence_transformer(
            texts,
            encoder_cfg["model_name"],
            encoder_cfg["batch_size"],
            device_override=encoder_cfg.get("device"),
        )
    elif enc_type == "tfidf_svd":
        embeddings = _encode_tfidf_svd(
            texts,
            encoder_cfg["n_components"],
            encoder_cfg["tfidf_ngram_range"],
            encoder_cfg["tfidf_min_df"],
            encoder_cfg["tfidf_max_df"],
            encoder_cfg["tfidf_max_features"],
            encoder_cfg["tfidf_sublinear_tf"],
            seed,
        )
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")

    if not np.isfinite(embeddings).all():
        nan_count = int(np.isnan(embeddings).sum())
        inf_count = int(np.isinf(embeddings).sum())
        raise ValueError(
            f"Refusing to cache non-finite embeddings for {encoder_key}: nan_count={nan_count}, inf_count={inf_count}"
        )

    atomic_write_npy(embeddings, cache_path)
    log.info(f"Cached embeddings: {cache_path.name} shape={embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_dim_reduction(cfg: dict, seed: int = 42):
    """Build dimensionality reduction model from config."""
    if cfg["type"] == "umap":
        return UMAP(
            n_neighbors=cfg["n_neighbors"],
            n_components=cfg["n_components"],
            min_dist=cfg.get("min_dist", 0.0),
            metric=cfg.get("metric", "cosine"),
            random_state=seed,
            transform_seed=seed,
        )
    elif cfg["type"] == "pca":
        return PCA(n_components=cfg["n_components"], random_state=seed)
    raise ValueError(f"Unknown dim_reduction type: {cfg['type']}")


def build_clustering(cfg: dict, seed: int = 42):
    """Build clustering model from config."""
    if cfg["type"] == "hdbscan":
        import hdbscan

        return hdbscan.HDBSCAN(
            min_cluster_size=cfg["min_cluster_size"],
            min_samples=cfg.get("min_samples", None),
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
    elif cfg["type"] == "kmeans":
        return MiniBatchKMeans(
            n_clusters=cfg["n_clusters"],
            random_state=seed,
            batch_size=cfg.get("batch_size", 1024),
            n_init=cfg.get("n_init", "auto"),
        )
    raise ValueError(f"Unknown clustering type: {cfg['type']}")


def build_vectorizer(cfg: dict) -> CountVectorizer:
    """Build CountVectorizer from config."""
    stop_words = get_stopwords()
    return CountVectorizer(
        stop_words=stop_words,
        ngram_range=tuple(cfg.get("ngram_range", [1, 1])),
        min_df=cfg.get("min_df", 1),
        max_df=cfg.get("max_df", 0.95),
        token_pattern=r"(?u)\b[\wёЁ-]+\b",
    )


def load_representation_backend(
    encoder_cfg: dict,
    seed: int,
    text_key: str,
    texts: list[str] | None = None,
):
    from sentence_transformers import SentenceTransformer

    enc_type = encoder_cfg["type"]
    if enc_type == "sentence_transformer":
        model_name = encoder_cfg["model_name"]
        device = encoder_cfg.get("device", "cpu")
        cache_key = ("sentence_transformer", model_name, device)
        if cache_key not in _REPRESENTATION_BACKENDS:
            _REPRESENTATION_BACKENDS[cache_key] = SentenceTransformer(model_name, device=device)
        return _REPRESENTATION_BACKENDS[cache_key]

    if enc_type == "tfidf_svd":
        if texts is None:
            raise ValueError("texts are required for tfidf_svd representation backends")
        cache_key = (
            "tfidf_svd",
            text_key,
            encoder_cfg["n_components"],
            tuple(encoder_cfg["tfidf_ngram_range"]),
            encoder_cfg["tfidf_min_df"],
            encoder_cfg["tfidf_max_df"],
            encoder_cfg["tfidf_max_features"],
            encoder_cfg["tfidf_sublinear_tf"],
            seed,
        )
        if cache_key not in _REPRESENTATION_BACKENDS:
            pipe = make_pipeline(
                TfidfVectorizer(
                    stop_words=get_stopwords(),
                    ngram_range=tuple(encoder_cfg["tfidf_ngram_range"]),
                    min_df=encoder_cfg["tfidf_min_df"],
                    max_df=encoder_cfg["tfidf_max_df"],
                    max_features=encoder_cfg["tfidf_max_features"],
                    token_pattern=r"(?u)\b[\wёЁ-]+\b",
                    sublinear_tf=encoder_cfg["tfidf_sublinear_tf"],
                ),
                TruncatedSVD(n_components=encoder_cfg["n_components"], random_state=seed),
            )
            pipe.fit(texts)
            _REPRESENTATION_BACKENDS[cache_key] = SklearnEmbedder(pipe)
        return _REPRESENTATION_BACKENDS[cache_key]

    raise ValueError(f"Unsupported encoder type for representation backend: {enc_type}")


def build_representation(cfg: dict):
    """Build representation model from config. Returns None for plain c-TF-IDF."""
    repr_type = cfg.get("type", "none")
    if repr_type == "none":
        return None
    if repr_type == "keybert":
        return KeyBERTInspired()
    if repr_type == "mmr":
        return MaximalMarginalRelevance(diversity=cfg.get("diversity", 0.3))
    raise ValueError(f"Unknown representation type: {repr_type}")


def build_topic_model(
    texts: list[str],
    embeddings: np.ndarray,
    encoder_cfg: dict,
    dim_cfg: dict,
    clust_cfg: dict,
    vec_cfg: dict,
    repr_cfg: dict,
    top_n_words: int = 10,
    seed: int = 42,
    text_key: str = "text_only_light",
):
    """Build and fit BERTopic."""
    repr_type = repr_cfg.get("type", "none")
    representation_backend = None
    if repr_type in {"keybert", "mmr"}:
        representation_backend = load_representation_backend(
            encoder_cfg=encoder_cfg,
            seed=seed,
            text_key=text_key,
            texts=texts,
        )

    def _make_model(current_vec_cfg: dict) -> BERTopic:
        return BERTopic(
            embedding_model=representation_backend,
            umap_model=build_dim_reduction(dim_cfg, seed),
            hdbscan_model=build_clustering(clust_cfg, seed),
            vectorizer_model=build_vectorizer(current_vec_cfg),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=repr_cfg.get("reduce_frequent_words", True)),
            representation_model=build_representation(repr_cfg),
            top_n_words=top_n_words,
            calculate_probabilities=False,
            verbose=False,
        )

    model = _make_model(vec_cfg)

    t0 = time.time()
    topics, _ = model.fit_transform(texts, embeddings=embeddings)
    elapsed = time.time() - t0

    return model, topics, elapsed


def cleanup_topic_model(model: BERTopic) -> None:
    """Release GPU-held resources after a BERTopic run."""
    representation_model = getattr(model, "representation_model", None)
    if representation_model is not None and hasattr(representation_model, "close"):
        representation_model.close()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    model: BERTopic,
    texts: list[str],
    topics: list[int],
    top_n_words: int = 10,
    true_labels: list[str] | None = None,
) -> dict:
    """Compute all metrics"""
    topic_ids = sorted(t for t in set(topics) if t != -1)
    topic_count = len(topic_ids)

    # Coverage
    coverage = float(np.mean(np.asarray(topics) != -1))

    all_topic_words = []
    for tid in topic_ids:
        words = [w for w, _ in model.get_topic(tid)[:top_n_words]]
        if words:
            all_topic_words.append(words)

    if topic_count == 0 or not all_topic_words:
        return {
            "topic_count": topic_count,
            "coverage": coverage,
            "topic_diversity": 0.0,
            "coherence_umass": float("nan"),
            "nmi_topic": float("nan"),
            "ari_topic": float("nan"),
        }

    # Topic diversity: unique words / total words (corrected denominator)
    unique_words = len({w for words in all_topic_words for w in words})
    total_words = sum(len(words) for words in all_topic_words)
    topic_diversity = unique_words / total_words if total_words > 0 else 0.0

    # Coherence: use model's own vectorizer for consistent tokenization
    analyzer = model.vectorizer_model.build_analyzer()
    tokenized_docs = [analyzer(text) for text in texts]
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # Filter topic words through dictionary
    filtered_topic_words = []
    for words in all_topic_words:
        filtered = [w for w in words if w in dictionary.token2id]
        if len(filtered) >= 2:
            filtered_topic_words.append(filtered)

    if not filtered_topic_words:
        coherence_val = float("nan")
    else:
        cm = CoherenceModel(
            topics=filtered_topic_words,
            corpus=corpus,
            dictionary=dictionary,
            coherence="u_mass",
        )
        coherence_val = float(cm.get_coherence())

    if true_labels is None:
        nmi_val = float("nan")
        ari_val = float("nan")
    else:
        nmi_val = float(normalized_mutual_info_score(true_labels, topics))
        ari_val = float(adjusted_rand_score(true_labels, topics))

    return {
        "topic_count": topic_count,
        "coverage": coverage,
        "topic_diversity": topic_diversity,
        "coherence_umass": coherence_val,
        "nmi_topic": nmi_val,
        "ari_topic": ari_val,
    }


def compute_selection_score(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add admissibility and selection_score columns to results DataFrame.

    Admissible: 12 <= topic_count <= 40 AND coverage >= 0.65 AND finite coherence.
    Score: 0.5 * norm(coherence) + 0.3 * norm(diversity) + 0.2 * norm(coverage)
    Normalization: min-max within the full DataFrame.
    """
    target_min = cfg.target_topic_min
    target_max = cfg.target_topic_max
    min_coverage = cfg.min_coverage
    weights = cfg.score_weights

    w_coh = weights.get("coherence", 0.5)
    w_div = weights.get("diversity", 0.3)
    w_cov = weights.get("coverage", 0.2)

    df = df.copy()

    df["admissible"] = (
        (df["topic_count"] >= target_min)
        & (df["topic_count"] <= target_max)
        & (df["coverage"] >= min_coverage)
        & df["coherence_umass"].apply(np.isfinite)
    )

    for col in ["coherence_umass", "topic_diversity", "coverage"]:
        if col not in df.columns:
            df[f"{col}_norm"] = 0.0
            continue
        series = df[col].astype(float)
        mn, mx = series.min(), series.max()
        if np.isclose(mn, mx):
            df[f"{col}_norm"] = 1.0
        else:
            df[f"{col}_norm"] = ((series - mn) / (mx - mn)).fillna(0.0)

    df["selection_score"] = (
        w_coh * df["coherence_umass_norm"] + w_div * df["topic_diversity_norm"] + w_cov * df["coverage_norm"]
    )

    return df
