"""BERTopic experiment runner.

Usage:
    uv run python scripts/run_experiment.py phase=single
    uv run python scripts/run_experiment.py phase=structure
    uv run python scripts/run_experiment.py phase=all

"""

import hashlib
import json
import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    atomic_write_json,
    build_embeddings,
    build_topic_model,
    cleanup_topic_model,
    compute_metrics,
    compute_selection_score,
    load_sample,
    preprocess_texts,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONF_DIR = Path(__file__).resolve().parent / "conf"
COMPONENT_KEYS = ["embed_text", "vec_text", "encoder", "dim_reduction", "clustering", "vectorizer", "representation"]


def _resolve_component_cfg(cfg: DictConfig, key: str, selected_key: str | None = None) -> dict:
    """Resolve a component config block to a plain dict. Uses cfg.<key> unless selected_key overrides."""
    component_key = selected_key if selected_key is not None else getattr(cfg, key)
    val = cfg.components[key][component_key]
    return OmegaConf.to_container(val, resolve=True)


def _set_selection(cfg: DictConfig, **overrides) -> DictConfig:
    """Override top-level selection fields on a struct-mode OmegaConf config."""
    OmegaConf.set_struct(cfg, False)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    OmegaConf.set_struct(cfg, True)
    return cfg


def _make_trial_cfg(cfg: DictConfig, row: dict | pd.Series | None = None) -> DictConfig:
    """Clone cfg into an independent trial config. If row is given, apply component/seed overrides from it."""
    trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if row is None:
        return trial_cfg

    row_dict = dict(row)

    component_configs = row_dict.get("_component_configs")
    if component_configs and isinstance(component_configs, str):
        import json as _json

        component_configs = _json.loads(component_configs)
    if isinstance(component_configs, dict):
        for param_key, variant_dict in component_configs.items():
            variant_key = str(row_dict.get(param_key, ""))
            if variant_key and variant_key not in trial_cfg.components.get(param_key, {}):
                _inject_variant(trial_cfg, param_key, variant_key, variant_dict)

    overrides = {key: str(row_dict[key]) for key in COMPONENT_KEYS if key in row_dict and pd.notna(row_dict[key])}
    if overrides:
        _set_selection(trial_cfg, **overrides)
    if "seed" in row_dict and pd.notna(row_dict["seed"]):
        _set_selection(trial_cfg, seed=int(row_dict["seed"]))
    return trial_cfg


def _inject_variant(cfg: DictConfig, parameter: str, variant_key: str, variant_cfg: dict) -> DictConfig:
    """Register a new component variant under cfg.components and select it."""
    OmegaConf.set_struct(cfg, False)
    cfg.components[parameter][variant_key] = OmegaConf.create(variant_cfg)
    setattr(cfg, parameter, variant_key)
    OmegaConf.set_struct(cfg, True)
    return cfg


def _load_phase_config(name: str) -> dict:
    """Load a phase definition YAML from scripts/conf/."""
    return yaml.safe_load((CONF_DIR / name).read_text())


def _dir_from_cfg(cfg: DictConfig, key: str) -> Path:
    """Resolve a paths.<key> value to an absolute directory, creating it if needed."""
    path = ROOT / cfg.paths[key]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _score_df(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Add admissibility and selection_score columns. Passes through empty/unscorable DataFrames."""
    if df.empty or "coherence_umass" not in df.columns:
        return df
    return compute_selection_score(df, cfg)


def select_top_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return the top-k rows by selection_score, preferring admissible rows."""
    if df.empty:
        return df.copy()

    working = df.copy()
    if "selection_score" not in working.columns and "coherence_umass" in working.columns:
        raise ValueError("select_top_k expects a scored dataframe")

    admissible = working[working["admissible"]] if "admissible" in working.columns else pd.DataFrame()
    chosen = admissible if not admissible.empty else working
    return chosen.sort_values("selection_score", ascending=False).head(k).reset_index(drop=True)


def _existing_valid_map(df: pd.DataFrame) -> dict[str, dict]:
    if df.empty or "experiment_id" not in df.columns:
        return {}
    valid = {}
    for row in df.to_dict("records"):
        exp_id = row.get("experiment_id")
        if exp_id and not _is_failed_record(row):
            valid[exp_id] = row
    return valid


def run_single(cfg: DictConfig) -> dict:
    """Run one BERTopic configuration end-to-end. Returns a flat metrics dict."""
    random_seed = int(cfg.seed)
    np.random.seed(random_seed)

    embed_col = cfg.embed_text
    vec_col = cfg.vec_text
    enc_key = cfg.encoder
    dim_key = cfg.dim_reduction
    clust_key = cfg.clustering
    vec_key = cfg.vectorizer
    repr_key = cfg.representation

    enc_cfg = _resolve_component_cfg(cfg, "encoder")
    dim_cfg = _resolve_component_cfg(cfg, "dim_reduction")
    clust_cfg = _resolve_component_cfg(cfg, "clustering")
    vec_cfg = _resolve_component_cfg(cfg, "vectorizer")
    repr_cfg = _resolve_component_cfg(cfg, "representation")

    require_lemma = "lemma" in embed_col or "lemma" in vec_col
    df = load_sample(ROOT, cfg.sample_size, random_seed)
    df = preprocess_texts(df, ROOT, cfg.sample_size, random_seed, require_lemma=require_lemma)

    texts_for_emb = df[embed_col].tolist()
    texts_for_vec = df[vec_col].tolist()
    true_labels = df["topic"].tolist()

    embeddings = build_embeddings(
        texts=texts_for_emb,
        encoder_key=enc_key,
        encoder_cfg=enc_cfg,
        cache_dir=_dir_from_cfg(cfg, "embeddings_dir"),
        seed=random_seed,
        text_column=embed_col,
    )

    model = None
    try:
        model, topics, elapsed = build_topic_model(
            texts=texts_for_vec,
            embeddings=embeddings,
            encoder_cfg=enc_cfg,
            dim_cfg=dim_cfg,
            clust_cfg=clust_cfg,
            vec_cfg=vec_cfg,
            repr_cfg=repr_cfg,
            top_n_words=cfg.top_n_words,
            seed=random_seed,
            text_key=vec_col,
        )

        metrics = compute_metrics(model, texts_for_vec, topics, cfg.top_n_words, true_labels=true_labels)

        result = {
            "seed": random_seed,
            "experiment_id": f"single_{embed_col}_{vec_col}_{enc_key}_{dim_key}_{clust_key}",
            "embed_text": embed_col,
            "vec_text": vec_col,
            "encoder": enc_key,
            "dim_reduction": dim_key,
            "clustering": clust_key,
            "vectorizer": vec_key,
            "representation": repr_key,
            "elapsed_sec": round(elapsed, 2),
            **metrics,
        }

        non_outlier = sorted(t for t in set(topics) if t != -1)
        if non_outlier:
            words_0 = [w for w, _ in model.get_topic(non_outlier[0])[: cfg.top_n_words]]
            result["sample_topic_words"] = ", ".join(words_0)

        return result
    finally:
        if model is not None:
            cleanup_topic_model(model)


def run_structure(cfg: DictConfig, results) -> pd.DataFrame:
    """Full factorial grid over embed_text x vec_text x encoder x clustering x dim_reduction."""
    structure_cfg = _load_phase_config("grid.yaml")

    embed_texts = structure_cfg.get("embed_text", [cfg.embed_text])
    vec_texts = structure_cfg.get("vec_text", [cfg.vec_text])
    encoders = structure_cfg.get("encoder", [cfg.encoder])
    clusterings = structure_cfg.get("clustering", [cfg.clustering])
    dim_reductions = structure_cfg.get("dim_reduction", [cfg.dim_reduction])
    fixed_vec = structure_cfg.get("vectorizer", cfg.vectorizer)
    fixed_repr = structure_cfg.get("representation", cfg.representation)

    total = len(embed_texts) * len(vec_texts) * len(encoders) * len(clusterings) * len(dim_reductions)
    log.info(
        "Structure search: %s experiments (%s emb x %s vec x %s enc x %s clust x %s dim)",
        total,
        len(embed_texts),
        len(vec_texts),
        len(encoders),
        len(clusterings),
        len(dim_reductions),
    )

    existing_df = results.load("structure", "experiments")
    existing_map = _existing_valid_map(existing_df)
    pending = []
    for i, (emb, vec, enc, clust, dim) in enumerate(
        product(embed_texts, vec_texts, encoders, clusterings, dim_reductions), 1
    ):
        exp_id = f"struct_{emb}_{vec}_{enc}_{clust}_{dim}"
        if exp_id in existing_map:
            continue
        trial_cfg = _make_trial_cfg(cfg)
        _set_selection(
            trial_cfg,
            embed_text=emb,
            vec_text=vec,
            encoder=enc,
            clustering=clust,
            dim_reduction=dim,
            vectorizer=fixed_vec,
            representation=fixed_repr,
        )
        pending.append((i, emb, vec, enc, clust, dim, trial_cfg, fixed_vec, fixed_repr))

    log.info("Structure: reusing %s cached rows, running %s rows", len(existing_map), len(pending))
    for i, emb, vec, enc, clust, dim, trial_cfg, fixed_vec, fixed_repr in pending:
        exp_id = f"struct_{emb}_{vec}_{enc}_{clust}_{dim}"
        log.info("[%s/%s] %s", i, total, f"{emb}|{vec} x {enc} x {clust} x {dim}")
        try:
            row = run_single(trial_cfg)
            row["phase"] = "structure"
            row["experiment_id"] = exp_id
            results.upsert_row("structure", "experiments", row)
        except Exception as exc:
            log.error("  FAILED: %s", exc)
            results.upsert_row(
                "structure",
                "experiments",
                {
                    "phase": "structure",
                    "experiment_id": exp_id,
                    "error": str(exc),
                    "embed_text": emb,
                    "vec_text": vec,
                    "encoder": enc,
                    "clustering": clust,
                    "dim_reduction": dim,
                    "vectorizer": fixed_vec,
                    "representation": fixed_repr,
                },
            )

    return results.load("structure", "experiments")


def run_peripherals(cfg: DictConfig, baselines: pd.DataFrame, results) -> pd.DataFrame:
    """Probe vectorizer x representation combinations on each structure finalist."""
    peri_cfg = _load_phase_config("peripherals.yaml")
    vectorizers = peri_cfg.get("vectorizer", [cfg.vectorizer])
    representations = peri_cfg.get("representation", [cfg.representation])

    total = len(baselines) * len(vectorizers) * len(representations)
    log.info("Peripheral search: %s experiments (%s finalists)", total, len(baselines))

    existing_df = results.load("peripherals", "experiments")
    existing_map = _existing_valid_map(existing_df)
    count = 0
    for baseline in baselines.to_dict("records"):
        base_cfg = _make_trial_cfg(cfg, baseline)
        for vec_key, repr_key in product(vectorizers, representations):
            count += 1
            exp_id = f"peri_{baseline['experiment_id']}_{vec_key}_{repr_key}"
            if exp_id in existing_map:
                continue
            trial_cfg = _make_trial_cfg(base_cfg)
            _set_selection(trial_cfg, vectorizer=vec_key, representation=repr_key)
            log.info("[%s/%s] %s -> %s x %s", count, total, baseline["experiment_id"], vec_key, repr_key)
            try:
                row = run_single(trial_cfg)
                row["phase"] = "peripherals"
                row["parent_experiment_id"] = baseline["experiment_id"]
                row["experiment_id"] = exp_id
                results.upsert_row("peripherals", "experiments", row)
            except Exception as exc:
                log.error("  FAILED: %s", exc)
                results.upsert_row(
                    "peripherals",
                    "experiments",
                    {
                        "phase": "peripherals",
                        "parent_experiment_id": baseline["experiment_id"],
                        "experiment_id": exp_id,
                        "error": str(exc),
                        **{key: baseline.get(key) for key in COMPONENT_KEYS},
                        "vectorizer": vec_key,
                        "representation": repr_key,
                    },
                )

    log.info("Peripherals: reusing %s cached rows", len(existing_map))
    return results.load("peripherals", "experiments")


def _group_applies(group: dict, baseline_cfg: DictConfig) -> bool:
    """Check whether a tuning group's apply_if filter matches the current baseline."""
    apply_if = group.get("apply_if", {})
    if not apply_if:
        return True
    for key, expected in apply_if.items():
        component, type_key = key.rsplit("_type", 1)
        actual = _resolve_component_cfg(baseline_cfg, component).get("type", "")
        if actual != expected:
            return False
    return True


def run_tuning(
    cfg: DictConfig,
    baselines: pd.DataFrame,
    results,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline-anchored hyperparameter tuning per finalist. Returns (probes, combined)."""
    tune_cfg = _load_phase_config("tuning.yaml")

    existing_probe_map = _existing_valid_map(results.load("tuning", "probes"))
    existing_combined_map = _existing_valid_map(results.load("tuning", "combined"))

    for baseline in baselines.to_dict("records"):
        baseline_cfg = _make_trial_cfg(cfg, baseline)
        chosen_variants: dict[str, tuple[str, dict]] = {}

        log.info("Tuning around finalist %s", baseline["experiment_id"])
        for group in tune_cfg.get("tuning_groups", []):
            if not _group_applies(group, baseline_cfg):
                continue

            parameter = group["parameter"]
            group_name = group["name"]
            group_rows = [
                {
                    **baseline,
                    "phase": "tuning_probe",
                    "tuning_group": group_name,
                    "variant_key": "__baseline__",
                    "is_group_baseline": True,
                    "experiment_id": f"tuneprobe_{baseline['experiment_id']}_{group_name}_baseline",
                }
            ]

            for variant in group.get("variants", []):
                probe_exp_id = f"tuneprobe_{baseline['experiment_id']}_{group_name}_{variant['key']}"
                if probe_exp_id in existing_probe_map:
                    group_rows.append(existing_probe_map[probe_exp_id])
                    continue
                trial_cfg = _make_trial_cfg(baseline_cfg)
                trial_cfg = _inject_variant(trial_cfg, parameter, variant["key"], variant["config"])
                try:
                    row = run_single(trial_cfg)
                    row["phase"] = "tuning_probe"
                    row["parent_experiment_id"] = baseline["experiment_id"]
                    row["tuning_group"] = group_name
                    row["variant_key"] = variant["key"]
                    row["is_group_baseline"] = False
                    row["experiment_id"] = probe_exp_id
                    group_rows.append(row)
                    results.upsert_row("tuning", "probes", row)
                except Exception as exc:
                    log.error("  FAILED %s/%s: %s", baseline["experiment_id"], variant["key"], exc)
                    failed_row = {
                        "phase": "tuning_probe",
                        "parent_experiment_id": baseline["experiment_id"],
                        "tuning_group": group_name,
                        "variant_key": variant["key"],
                        "is_group_baseline": False,
                        "experiment_id": probe_exp_id,
                        "error": str(exc),
                        **{key: baseline.get(key) for key in COMPONENT_KEYS},
                    }
                    group_rows.append(failed_row)
                    results.upsert_row("tuning", "probes", failed_row)

            group_df = _score_df(pd.DataFrame(group_rows), cfg)
            if not group_df.empty:
                group_df["group_selection_score"] = group_df["selection_score"]
                group_df["group_admissible"] = group_df["admissible"]

                best_group_row = select_top_k(group_df, 1).iloc[0]
                if not bool(best_group_row.get("is_group_baseline", False)):
                    winner_cfg = next(
                        variant["config"]
                        for variant in group["variants"]
                        if variant["key"] == best_group_row["variant_key"]
                    )
                    chosen_variants[parameter] = (str(best_group_row["variant_key"]), winner_cfg)

        baselinecarry_id = f"baselinecarry_{baseline['experiment_id']}"
        results.upsert_row(
            "tuning",
            "combined",
            {
                **baseline,
                "phase": "tuning_combined",
                "parent_experiment_id": baseline["experiment_id"],
                "experiment_id": baselinecarry_id,
                "candidate_kind": "baseline",
                "applied_variants": "baseline",
            },
        )

        combined_cfg = _make_trial_cfg(baseline_cfg)
        applied_keys = []
        injected_configs = {}
        for parameter, (variant_key, variant_cfg) in chosen_variants.items():
            combined_cfg = _inject_variant(combined_cfg, parameter, variant_key, variant_cfg)
            applied_keys.append(f"{parameter}={variant_key}")
            injected_configs[parameter] = variant_cfg

        if applied_keys:
            tuned_id = f"tuned_{baseline['experiment_id']}"
            if tuned_id in existing_combined_map:
                continue
            try:
                combined_row = run_single(combined_cfg)
            except Exception as exc:
                combined_row = {
                    "phase": "tuning_combined",
                    "parent_experiment_id": baseline["experiment_id"],
                    "experiment_id": tuned_id,
                    "candidate_kind": "tuned",
                    "applied_variants": "; ".join(applied_keys),
                    "_component_configs": json.dumps(injected_configs),
                    "error": str(exc),
                    **{key: baseline.get(key) for key in COMPONENT_KEYS},
                }
            else:
                combined_row["phase"] = "tuning_combined"
                combined_row["parent_experiment_id"] = baseline["experiment_id"]
                combined_row["experiment_id"] = tuned_id
                combined_row["candidate_kind"] = "tuned"
                combined_row["applied_variants"] = "; ".join(applied_keys)
                combined_row["_component_configs"] = json.dumps(injected_configs)
            results.upsert_row("tuning", "combined", combined_row)

    probe_df = _score_df(results.load("tuning", "probes"), cfg)
    combined_df = _score_df(results.load("tuning", "combined"), cfg)
    return probe_df, combined_df


def run_confirmation(
    cfg: DictConfig,
    finalists: pd.DataFrame,
    results,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run finalists across multiple seeds. Returns (raw runs, aggregated summary)."""
    seeds = list(cfg.selection.confirmation_seeds)

    total = len(finalists) * len(seeds)
    log.info("Confirmation: %s runs (%s finalists x %s seeds)", total, len(finalists), len(seeds))

    existing_map = _existing_valid_map(results.load("confirmation", "runs"))
    count = 0
    for finalist in finalists.to_dict("records"):
        for seed in seeds:
            count += 1
            exp_id = f"confirm_{finalist['experiment_id']}_seed{seed}"
            if exp_id in existing_map:
                continue
            trial_cfg = _make_trial_cfg(cfg, finalist)
            _set_selection(trial_cfg, seed=int(seed))
            log.info("[%s/%s] %s @ seed=%s", count, total, finalist["experiment_id"], seed)
            try:
                row = run_single(trial_cfg)
                row["phase"] = "confirm"
                row["parent_experiment_id"] = finalist["experiment_id"]
                row["experiment_id"] = exp_id
            except Exception as exc:
                row = {
                    "phase": "confirm",
                    "parent_experiment_id": finalist["experiment_id"],
                    "experiment_id": exp_id,
                    "error": str(exc),
                    **{key: finalist.get(key) for key in COMPONENT_KEYS},
                    "seed": int(seed),
                }
            results.upsert_row("confirmation", "runs", row)

    runs_df = results.load("confirmation", "runs")
    if runs_df.empty:
        return runs_df, runs_df

    summary_df = (
        runs_df.groupby("parent_experiment_id", as_index=False)
        .agg(
            embed_text=("embed_text", "first"),
            vec_text=("vec_text", "first"),
            encoder=("encoder", "first"),
            dim_reduction=("dim_reduction", "first"),
            clustering=("clustering", "first"),
            vectorizer=("vectorizer", "first"),
            representation=("representation", "first"),
            mean_topic_count=("topic_count", "mean"),
            std_topic_count=("topic_count", "std"),
            mean_coverage=("coverage", "mean"),
            std_coverage=("coverage", "std"),
            mean_topic_diversity=("topic_diversity", "mean"),
            std_topic_diversity=("topic_diversity", "std"),
            mean_coherence_umass=("coherence_umass", "mean"),
            std_coherence_umass=("coherence_umass", "std"),
            mean_nmi_topic=("nmi_topic", "mean"),
            std_nmi_topic=("nmi_topic", "std"),
            mean_ari_topic=("ari_topic", "mean"),
            std_ari_topic=("ari_topic", "std"),
            mean_elapsed_sec=("elapsed_sec", "mean"),
        )
        .rename(columns={"parent_experiment_id": "experiment_id"})
    )

    scored_summary = summary_df.rename(
        columns={
            "mean_topic_count": "topic_count",
            "mean_coverage": "coverage",
            "mean_topic_diversity": "topic_diversity",
            "mean_coherence_umass": "coherence_umass",
        }
    )
    scored_summary = _score_df(scored_summary, cfg)
    return runs_df, scored_summary


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-safe list of dicts."""
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """Load a list of dicts back into a DataFrame."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _replace_records(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """Replace rows by experiment_id, preserving untouched rows and order where possible."""
    if not existing:
        return incoming
    if not incoming:
        return existing
    if "experiment_id" not in incoming[0]:
        return incoming

    incoming_by_id = {r["experiment_id"]: r for r in incoming if "experiment_id" in r}
    merged = []
    seen_ids = set()
    for row in existing:
        exp_id = row.get("experiment_id")
        if exp_id in incoming_by_id:
            merged.append(incoming_by_id[exp_id])
            seen_ids.add(exp_id)
        else:
            merged.append(row)
    for exp_id, row in incoming_by_id.items():
        if exp_id not in seen_ids:
            merged.append(row)
    return merged


class Results:
    """Fingerprinted results store with atomic saves and row-level upserts."""

    def __init__(self, path: Path, fingerprint: str):
        self.path = path
        self.fingerprint = fingerprint
        self.data: dict[str, dict[str, list]] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                meta = raw.get("meta", {})
                if meta.get("fingerprint") == fingerprint:
                    for k, v in raw.items():
                        if k != "meta":
                            self.data[k] = v
                    log.info("Loaded %s phases from %s", len(self.data), path.name)
                else:
                    stale_fingerprint = meta.get("fingerprint") or "legacy"
                    stale_path = path.with_name(f"{path.stem}.{stale_fingerprint}{path.suffix}")
                    if not stale_path.exists():
                        path.replace(stale_path)
                        log.warning("Moved stale cache to %s", stale_path.name)
                    else:
                        log.warning("Ignoring %s because its fingerprint does not match the current config", path.name)
            except (json.JSONDecodeError, KeyError):
                log.warning("Could not parse %s, starting fresh", path.name)

    def replace(self, phase: str, key: str, df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        """Replace a full table."""
        if phase not in self.data:
            self.data[phase] = {}
        self.data[phase][key] = _df_to_records(df)
        if save:
            self.save()
        return df

    def upsert_row(self, phase: str, key: str, row: dict) -> None:
        """Replace or append a single experiment row and persist immediately."""
        if phase not in self.data:
            self.data[phase] = {}
        existing = self.data[phase].get(key, [])
        self.data[phase][key] = _replace_records(existing, [row])
        self.save()

    def load(self, phase: str, key: str) -> pd.DataFrame:
        """Load a previously stored DataFrame by phase/key."""
        return _records_to_df(self.data.get(phase, {}).get(key, []))

    def save(self) -> None:
        """Write everything to one JSON file atomically."""
        payload = {
            "meta": {
                "fingerprint": self.fingerprint,
                "updated_at": datetime.now().isoformat(),
            },
            **self.data,
        }
        atomic_write_json(payload, self.path)
        log.info("Saved results to %s", self.path)


def _is_failed_record(row: dict) -> bool:
    if row.get("error") not in (None, ""):
        return True
    for metric in ["topic_count", "coverage", "topic_diversity", "coherence_umass"]:
        value = row.get(metric)
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
    return False


def _run_phase(name: str, cfg: DictConfig, results: Results) -> pd.DataFrame | None:
    """Run a single phase, accumulate results. Returns finalists DataFrame or None."""
    if name == "structure":
        df = _score_df(run_structure(cfg, results), cfg)
        results.replace("structure", "experiments", df)
        finalists = select_top_k(df, int(cfg.selection.structure_finalists_k))
        results.replace("structure", "finalists", finalists)
        return finalists

    if name == "peripherals":
        structure_df = results.load("structure", "finalists")
        df = _score_df(run_peripherals(cfg, structure_df, results), cfg)
        results.replace("peripherals", "experiments", df)
        finalists = select_top_k(df, int(cfg.selection.peripherals_finalists_k))
        results.replace("peripherals", "finalists", finalists)
        return finalists

    if name == "tuning":
        peripheral_df = results.load("peripherals", "finalists")
        probe_df, combined_df = run_tuning(cfg, peripheral_df, results)
        probe_df = _score_df(probe_df, cfg)
        combined_df = _score_df(combined_df, cfg)
        results.replace("tuning", "probes", probe_df)
        results.replace("tuning", "combined", combined_df)
        finalists = select_top_k(combined_df, int(cfg.selection.confirmation_finalists_k))
        results.replace("tuning", "finalists", finalists)
        return finalists

    if name == "confirm":
        finalists_df = results.load("tuning", "finalists")
        runs_df, summary_df = run_confirmation(cfg, finalists_df, results)
        results.replace("confirmation", "runs", runs_df)
        results.replace("confirmation", "summary", summary_df)
        return summary_df

    return None


def _config_fingerprint(cfg: DictConfig) -> str:
    payload = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "grid": _load_phase_config("grid.yaml"),
        "peripherals": _load_phase_config("peripherals.yaml"),
        "tuning": _load_phase_config("tuning.yaml"),
    }
    payload["config"].pop("phase", None)
    payload["config"].pop("hydra", None)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Entry point. Dispatches to the requested phase (single, structure, peripherals, tuning, confirm, all)."""
    results_dir = _dir_from_cfg(cfg, "results_dir")
    results = Results(results_dir / "results.json", fingerprint=_config_fingerprint(cfg))
    phase = cfg.get("phase", "single")

    if phase == "single":
        df = _score_df(pd.DataFrame([run_single(cfg)]), cfg)
        results.replace("single", "experiments", df)
        return

    if phase == "all":
        phases = ["structure", "peripherals", "tuning", "confirm"]
    else:
        phases = [phase]

    for p in phases:
        _run_phase(p, cfg, results)


if __name__ == "__main__":
    main()
