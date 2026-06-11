import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RESULTS_ROOT = Path("retrieval_framework/results")


COMPONENT_RUNS = [
    {
        "stage": "R0",
        "label": "Dense only",
        "kind": "retrieval_only",
        "path": "nq-dense-llama3-Top5-M10x10-adv-LM_targeted-5-retrieval-only.summary.json",
        "components": "dense",
        "purpose": "Original dense baseline.",
    },
    {
        "stage": "R1",
        "label": "BM25 only",
        "kind": "retrieval_only",
        "path": "nq-bm25-llama3-Top5-M10x10-adv-LM_targeted-5-retrieval-only.summary.json",
        "components": "bm25",
        "purpose": "Sparse keyword baseline.",
    },
    {
        "stage": "R2",
        "label": "Paper hybrid alpha=0.5",
        "kind": "retrieval_only",
        "path": "nq-paper_hybrid-llama3-Top5-M10x10-alpha0p5-adv-LM_targeted-5-retrieval-only.summary.json",
        "components": "dense+bm25 weighted hybrid",
        "purpose": "Paper-style hybrid baseline.",
    },
    {
        "stage": "R3",
        "label": "Secure ensemble, no hardening",
        "kind": "retrieval_only",
        "path": "nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-adv-LM_targeted-5-retrieval-only.summary.json",
        "components": "dense+bm25 consensus rrf",
        "purpose": "Consensus RRF without evidence hardening.",
    },
    {
        "stage": "R4",
        "label": "Secure ensemble + cluster cap",
        "kind": "retrieval_only",
        "path": "nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap1-adv-LM_targeted-5-retrieval-only.summary.json",
        "components": "consensus rrf + retriever-level cluster cap",
        "purpose": "Checks whether duplicate caps alone help.",
    },
    {
        "stage": "R5",
        "label": "Evidence hardening v1",
        "kind": "retrieval_only",
        "path": "evidence_focused/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-adv-LM_targeted-5-retrieval-only.summary.json",
        "components": "cluster cap + answer support + contradiction + query echo",
        "purpose": "First focused evidence-hardening path.",
    },
    {
        "stage": "R6",
        "label": "Head-focused filter",
        "kind": "retrieval_only",
        "path": "head_focused/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-hf3-hs2-cons-adv-LM_targeted-5-retrieval-only.summary.json",
        "components": "evidence hardening + top3 conflict handling",
        "purpose": "Tests head-focused answer-support filter.",
    },
    {
        "stage": "R7",
        "label": "QA-only answer extraction v3",
        "kind": "retrieval_only",
        "path": "evidence_qa_only_v3/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-adv-LM_targeted-5-retrieval-only.summary.json",
        "components": "QA extractor replaces heuristic answer extraction",
        "purpose": "More general answer extraction using the small QA model.",
    },
    {
        "stage": "R8",
        "label": "QA-only robust scoring",
        "kind": "retrieval_only",
        "path": "evidence_qa_only_robust/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-adv-LM_targeted-5-retrieval-only.summary.json",
        "components": "QA extractor + robust answer scoring",
        "purpose": "Adds robust answer-level scoring without LLM generation.",
    },
    {
        "stage": "R9",
        "label": "QA robust + top1 dominance",
        "kind": "retrieval_only",
        "path": "evidence_qa_only_robust_top1/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-adv-LM_targeted-5-retrieval-only.summary.json",
        "components": "QA robust scoring + top1 dominance defense",
        "purpose": "Current strongest retrieval-only component stack before weak-attack grounding.",
    },
    {
        "stage": "R10",
        "label": "Evidence focused v3",
        "kind": "retrieval_only",
        "path": "evidence_focused_v3/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-adv-LM_targeted-5-retrieval-only.summary.json",
        "components": "focused heuristic evidence hardening v3",
        "purpose": "Latest focused heuristic retrieval-only run.",
    },
    {
        "stage": "G0",
        "label": "Evidence focused v3 + LLM",
        "kind": "llm",
        "path": "evidence_focused_v3/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-adv-LM_targeted-5.summary.json",
        "components": "focused heuristic evidence hardening v3 + llama3",
        "purpose": "LLM ASR for focused v3.",
    },
    {
        "stage": "G1",
        "label": "QA-only v3 + LLM",
        "kind": "llm",
        "path": "evidence_qa_only_v3_llm/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-adv-LM_targeted-5.summary.json",
        "components": "QA extractor + llama3",
        "purpose": "LLM ASR for QA-only v3.",
    },
    {
        "stage": "G2",
        "label": "QA robust + stable LLM",
        "kind": "llm",
        "path": "evidence_qa_only_robust_llm_stable/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-adv-LM_targeted-5.summary.json",
        "components": "QA robust scoring + stable generation",
        "purpose": "LLM ASR after generation-stability fix.",
    },
    {
        "stage": "G3",
        "label": "QA robust top1 + stable LLM",
        "kind": "llm",
        "path": "evidence_qa_only_robust_top1_llm_stable/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-adv-LM_targeted-5.summary.json",
        "components": "QA robust + top1 dominance + stable generation",
        "purpose": "Strongest LLM run currently available.",
    },
]


MISSING_RECOMMENDATIONS = [
    {
        "item": "Clean utility on NQ",
        "status": "missing",
        "why": "Current component rows mostly use poisoned NQ. Need clean accuracy/EM/F1 to show the defense is not over-filtering.",
    },
    {
        "item": "Clean utility on HotpotQA",
        "status": "missing",
        "why": "Needed to test multi-hop evidence behavior and answer-support diversity on clean data.",
    },
    {
        "item": "Unified one-module-at-a-time run",
        "status": "partial",
        "why": "Existing results are useful but were produced across different versioned configs. A strict paper table should toggle one component per row under one config.",
    },
    {
        "item": "LLM ablation for every component",
        "status": "partial",
        "why": "Several rows are retrieval-only. LLM ASR is only available for selected final variants.",
    },
    {
        "item": "Weak attack component ablation",
        "status": "missing",
        "why": "Weak attack results compare defended vs undefended, but not individual defense components.",
    },
    {
        "item": "Adaptive strong attack",
        "status": "missing",
        "why": "Current weak attacks are not optimized against the defense; PoisonedRAG and defense-aware attacks should remain the strong-attack test.",
    },
]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        print(f"Skipping invalid JSON {path}: {exc}")
        return None


def nested_get(data: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def fmt_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def feature_flags(config: Dict[str, Any]) -> Dict[str, Any]:
    retriever = config.get("retriever", {})
    hardening = config.get("evidence_hardening", {})
    answer_support = hardening.get("answer_support", {})
    return {
        "retriever": retriever.get("type", ""),
        "fusion": retriever.get("fusion", ""),
        "retriever_cluster_cap": retriever.get("cluster_cap", ""),
        "evidence_hardening": bool(hardening.get("enabled")),
        "eh_candidate_depth": hardening.get("candidate_depth", ""),
        "eh_cluster": bool(nested_get(hardening, ["cluster", "enabled"], False)),
        "eh_cluster_cap": nested_get(hardening, ["cluster", "cap"], ""),
        "answer_support": bool(answer_support.get("enabled", False)),
        "answer_extractor": answer_support.get("extractor", "heuristic" if answer_support.get("enabled") else ""),
        "answer_scoring_mode": answer_support.get("answer_scoring_mode")
        or answer_support.get("scoring_mode")
        or answer_support.get("qa_scoring_mode", ""),
        "qa_model_name": answer_support.get("qa_model_name", ""),
        "qa_include_heuristic": answer_support.get("qa_include_heuristic", ""),
        "contradiction": bool(nested_get(hardening, ["contradiction", "enabled"], False)),
        "head_filter": bool(nested_get(hardening, ["head_filter", "enabled"], False)),
        "top1_dominance": bool(nested_get(hardening, ["top1_dominance", "enabled"], False)),
        "query_echo": bool(nested_get(hardening, ["query_echo", "enabled"], False)),
        "rank_guard": bool(nested_get(hardening, ["rank_guard", "enabled"], False)),
        "grounding": bool(answer_support.get("grounding_enabled", False)),
        "skip_llm": bool(config.get("skip_llm", False)),
    }


def row_from_summary(run: Dict[str, str], root: Path) -> Dict[str, Any]:
    path = root / run["path"]
    payload = load_json(path)
    if payload is None:
        return {
            **run,
            "exists": False,
            "path": str(path),
        }

    config = payload.get("config", {})
    summary = payload.get("summary", {})
    flags = feature_flags(config)
    row: Dict[str, Any] = {
        **run,
        "exists": True,
        "path": str(path),
        "dataset": summary.get("dataset") or config.get("dataset", ""),
        "num_queries": summary.get("num_queries", ""),
        "top_k": summary.get("top_k") or config.get("top_k", ""),
        "retrieval_precision_mean": summary.get("retrieval_precision_mean", ""),
        "retrieval_recall_mean": summary.get("retrieval_recall_mean", ""),
        "retrieval_f1_mean": summary.get("retrieval_f1_mean", ""),
        "contamination_at_1_mean": summary.get("contamination_at_1_mean", ""),
        "contamination_at_3_mean": summary.get("contamination_at_3_mean", ""),
        "contamination_at_5_mean": summary.get("contamination_at_5_mean", ""),
        "asr_mean": summary.get("asr_mean", ""),
        "hardening_filtered_by_cluster_count_mean": summary.get(
            "hardening_filtered_by_cluster_count_mean", ""
        ),
        "hardening_filtered_by_answer_count_mean": summary.get(
            "hardening_filtered_by_answer_count_mean", ""
        ),
        "hardening_conflict_rate": summary.get("hardening_conflict_rate", ""),
        "head_filter_trigger_rate": summary.get("head_filter_trigger_rate", ""),
        "top1_dominance_trigger_rate": summary.get("top1_dominance_trigger_rate", ""),
        "answer_level_conflict_rate": summary.get("answer_level_conflict_rate", ""),
    }
    row.update(flags)
    return row


def collect_inventory(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*.summary.json")):
        payload = load_json(path)
        if payload is None:
            continue
        config = payload.get("config", {})
        summary = payload.get("summary", {})
        flags = feature_flags(config)
        row = {
            "run_name": path.name,
            "relative_path": str(path.relative_to(root)),
            "dataset": summary.get("dataset") or config.get("dataset", ""),
            "num_queries": summary.get("num_queries", ""),
            "retrieval_precision_mean": summary.get("retrieval_precision_mean", ""),
            "contamination_at_1_mean": summary.get("contamination_at_1_mean", ""),
            "contamination_at_5_mean": summary.get("contamination_at_5_mean", ""),
            "asr_mean": summary.get("asr_mean", ""),
        }
        row.update(flags)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, component_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Component Ablation Inventory",
        "",
        "This report consolidates existing experiment summaries. It does not rerun experiments.",
        "",
        "## Existing Component Rows",
        "",
        "| stage | label | kind | n | C@1 | C@5 | ASR | components | status |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in component_rows:
        status = "available" if row.get("exists") else "missing file"
        lines.append(
            "| {stage} | {label} | {kind} | {n} | {c1} | {c5} | {asr} | {components} | {status} |".format(
                stage=row.get("stage", ""),
                label=row.get("label", ""),
                kind=row.get("kind", ""),
                n=row.get("num_queries", ""),
                c1=fmt_float(row.get("contamination_at_1_mean")),
                c5=fmt_float(row.get("contamination_at_5_mean")),
                asr=fmt_float(row.get("asr_mean")),
                components=row.get("components", ""),
                status=status,
            )
        )

    lines.extend(
        [
            "",
            "## Missing Or Partial Items",
            "",
            "| item | status | why |",
            "|---|---|---|",
        ]
    )
    for item in MISSING_RECOMMENDATIONS:
        lines.append(f"| {item['item']} | {item['status']} | {item['why']} |")

    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- The table above mixes retrieval-only and LLM rows on purpose; use `kind` to separate them.",
            "- Existing rows are useful for an inventory, but not all of them are strict one-component toggles.",
            "- For a final paper ablation, keep one dataset, one attack, one top-k, one retriever stack, and toggle one defense component at a time.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified component-ablation report.")
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    parser.add_argument(
        "--output_dir",
        default="retrieval_framework/results/component_ablation_report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    out = Path(args.output_dir)

    component_rows = [row_from_summary(run, root) for run in COMPONENT_RUNS]
    inventory_rows = collect_inventory(root)

    component_fields = [
        "stage",
        "label",
        "kind",
        "exists",
        "dataset",
        "num_queries",
        "top_k",
        "retrieval_precision_mean",
        "retrieval_recall_mean",
        "retrieval_f1_mean",
        "contamination_at_1_mean",
        "contamination_at_3_mean",
        "contamination_at_5_mean",
        "asr_mean",
        "retriever",
        "fusion",
        "retriever_cluster_cap",
        "evidence_hardening",
        "eh_candidate_depth",
        "eh_cluster",
        "eh_cluster_cap",
        "answer_support",
        "answer_extractor",
        "answer_scoring_mode",
        "qa_model_name",
        "qa_include_heuristic",
        "contradiction",
        "head_filter",
        "top1_dominance",
        "query_echo",
        "rank_guard",
        "grounding",
        "skip_llm",
        "hardening_filtered_by_cluster_count_mean",
        "hardening_filtered_by_answer_count_mean",
        "hardening_conflict_rate",
        "head_filter_trigger_rate",
        "top1_dominance_trigger_rate",
        "answer_level_conflict_rate",
        "components",
        "purpose",
        "path",
    ]
    write_csv(out / "component_ablation_existing.csv", component_rows, component_fields)
    write_csv(out / "all_summary_inventory.csv", inventory_rows)
    write_csv(out / "missing_items.csv", MISSING_RECOMMENDATIONS, ["item", "status", "why"])
    write_markdown(out / "component_ablation_report.md", component_rows)

    available = sum(1 for row in component_rows if row.get("exists"))
    print(f"Wrote component report to {out}")
    print(f"Component rows available: {available}/{len(component_rows)}")
    print(f"Inventory rows: {len(inventory_rows)}")


if __name__ == "__main__":
    main()
