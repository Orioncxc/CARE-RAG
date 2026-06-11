from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "retrieval_framework" / "results" / "icde_paper_experiments"


def load_json(path: str | Path) -> Any:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summary_payload(path: str | Path) -> Dict[str, Any]:
    payload = load_json(path)
    return payload.get("summary", payload)


def pct(value: Any) -> Any:
    if value is None or value == "":
        return ""
    return round(float(value) * 100, 2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_raw(label: str, source: str | Path, manifest: List[Dict[str, str]]) -> None:
    src = ROOT / source
    if not src.exists():
        manifest.append({"label": label, "source": str(source), "copied_to": "", "status": "missing"})
        return
    safe_name = str(source).replace("/", "__")
    dst = OUT / "raw" / label / safe_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "label": label,
            "source": str(source),
            "copied_to": str(dst.relative_to(ROOT)),
            "status": "copied",
        }
    )


def table_training_provenance() -> List[Dict[str, Any]]:
    metadata = load_json(
        "retrieval_framework/results/candidate_proposal_hotpot_deberta_5000/"
        "labeled_examples_metadata.json"
    )
    history = load_json(
        "retrieval_framework/results/candidate_proposal_hotpot_deberta_5000/training_history.json"
    )
    return [
        {
            "component": "candidate proposal model",
            "base_model": "deepset/deberta-v3-base-squad2",
            "training_dataset": metadata.get("dataset_name"),
            "training_split": metadata.get("split"),
            "training_examples": metadata.get("num_examples"),
            "training_corpus_path": metadata.get("source_data_dir"),
            "evaluation_dataset": "nq",
            "evaluation_split": "test",
            "evaluation_queries": 100,
            "corpus_overlap_policy": "disjoint dataset directories: HotpotQA train for proposal training; NQ test/poisoned corpus for evaluation",
            "epochs": len(history),
            "final_train_loss": round(history[-1].get("train_loss", 0.0), 4) if history else "",
            "final_dev_loss": round(history[-1].get("dev_loss", 0.0), 4) if history else "",
        }
    ]


def table_small_model_reliability() -> List[Dict[str, Any]]:
    payload = load_json(
        "retrieval_framework/results/icde_paper_experiments/"
        "small_model_reliability_100q/qa_extractor_precision_summary.json"
    )
    rows = []
    for item in payload["summaries"]:
        model = item["model"]
        if model.endswith("candidate_proposal_hotpot_deberta_5000/model"):
            name = "Hotpot-trained proposal DeBERTa"
        elif "minilm" in model:
            name = "MiniLM SQuAD2"
        elif "roberta-base" in model:
            name = "RoBERTa SQuAD2"
        elif "deberta-v3" in model:
            name = "DeBERTa SQuAD2"
        else:
            name = model
        rows.append(
            {
                "model": name,
                "top1_correct_%": pct(item["top1_primary_correct_rate"]),
                "top1_poison_%": pct(item["top1_primary_incorrect_rate"]),
                "top1_other_%": pct(item["top1_primary_other_rate"]),
                "top5_gold_candidate_recall_%": pct(item["top5_any_correct_rate"]),
                "top5_poison_candidate_recall_%": pct(item["top5_any_incorrect_rate"]),
                "mention_correct_%": pct(item["mention_correct_rate"]),
                "mention_poison_%": pct(item["mention_incorrect_rate"]),
                "mentions": item["mention_total"],
            }
        )
    return rows


def candidate_rows(path: str, model: str, depth: int) -> Iterable[Dict[str, Any]]:
    for item in load_json(path):
        yield {
            "retrieved_depth": depth,
            "proposal_model": model,
            "candidate_pool": item["pool"],
            "avg_pool_size": round(item["avg_pool_size"], 2),
            "gold_R@20_%": pct(item.get("gold_recall@20")),
            "gold_R@50_%": pct(item.get("gold_recall@50")),
            "gold_R@100_%": pct(item.get("gold_recall@100")),
            "gold_R@200_%": pct(item.get("gold_recall@200")),
            "poison_R@20_%": pct(item.get("poison_recall@20")),
            "poison_R@100_%": pct(item.get("poison_recall@100")),
            "any_R@100_%": pct(item.get("any_recall@100")),
        }


def table_candidate_recall() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(
        candidate_rows(
            "retrieval_framework/results/candidate_pool_recall_top5_deberta_recall100/"
            "candidate_pool_recall_summary.json",
            "base DeBERTa",
            5,
        )
    )
    rows.extend(
        candidate_rows(
            "retrieval_framework/results/candidate_pool_recall_top5_hotpot_proposal_deberta/"
            "candidate_pool_recall_summary.json",
            "Hotpot-trained proposal DeBERTa",
            5,
        )
    )
    rows.extend(
        candidate_rows(
            "retrieval_framework/results/candidate_pool_recall_top30_base_deberta/"
            "candidate_pool_recall_summary.json",
            "base DeBERTa",
            30,
        )
    )
    rows.extend(
        candidate_rows(
            "retrieval_framework/results/candidate_pool_recall_top30_hotpot_proposal_deberta/"
            "candidate_pool_recall_summary.json",
            "Hotpot-trained proposal DeBERTa",
            30,
        )
    )
    return rows


def table_retrieval_hardening() -> List[Dict[str, Any]]:
    specs = [
        (
            "Dense retriever, no hardening",
            "retrieval_framework/results/nq-dense-llama3-Top5-M10x10-adv-LM_targeted-5-retrieval-only.summary.json",
        ),
        (
            "BM25 retriever, no hardening",
            "retrieval_framework/results/nq-bm25-llama3-Top5-M10x10-adv-LM_targeted-5-retrieval-only.summary.json",
        ),
        (
            "Secure ensemble, no hardening",
            "retrieval_framework/results/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-adv-LM_targeted-5-retrieval-only.summary.json",
        ),
        (
            "QA hardening with MiniLM",
            "retrieval_framework/results/evidence_qa_only_robust/nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-adv-LM_targeted-5-retrieval-only.summary.json",
        ),
        (
            "Full hardening with Hotpot-trained proposal",
            "retrieval_framework/results/candidate_proposal_hotpot_deberta_hardening_100q/proposal_hotpot_deberta_retrieval_100q.summary.json",
        ),
    ]
    rows = []
    for name, path in specs:
        summary = summary_payload(path)
        rows.append(
            {
                "method": name,
                "queries": summary.get("num_queries"),
                "contam@1_%": pct(summary.get("contamination_at_1_mean")),
                "contam@3_%": pct(summary.get("contamination_at_3_mean")),
                "contam@5_%": pct(summary.get("contamination_at_5_mean")),
                "candidate_supported_answers": summary.get(
                    "answer_level_candidate_supported_answer_count_mean", ""
                ),
                "no_strong_answer_%": pct(summary.get("answer_level_no_strong_answer_rate")),
                "multi_supported_conflict_%": pct(
                    summary.get("answer_level_multi_supported_conflict_rate")
                ),
            }
        )
    return rows


def table_component_ablation() -> List[Dict[str, Any]]:
    names = {
        "u00-secure-no-hardening": "secure ensemble only",
        "u01-cluster-only": "+ cluster cap",
        "u02-plus-qa-answer-support": "+ candidate answer support",
        "u03-plus-robust-answer-scoring": "+ robust answer scoring",
        "u04-plus-contradiction-query-echo": "+ contradiction/query-echo guards",
        "u05-plus-topic-grounding": "+ topic grounding",
        "u06-plus-top1-dominance": "+ top-1 dominance gate",
    }
    rows = []
    for run, label in names.items():
        path = (
            f"retrieval_framework/results/missing_experiments/unified_component_ablation_50k/"
            f"{run}.summary.json"
        )
        summary = summary_payload(path)
        rows.append(
            {
                "variant": label,
                "contam@1_%": pct(summary.get("contamination_at_1_mean")),
                "contam@3_%": pct(summary.get("contamination_at_3_mean")),
                "contam@5_%": pct(summary.get("contamination_at_5_mean")),
                "candidate_supported_answers": summary.get(
                    "answer_level_candidate_supported_answer_count_mean", ""
                ),
            }
        )
    return rows


def table_final_llm_asr() -> List[Dict[str, Any]]:
    rows = []
    qa_swap = load_json(
        "retrieval_framework/results/qa_model_swap_100q/"
        "qa_model_swap_llama3_generation_with_minilm_summary.json"
    )
    name_map = {
        "minilm_original_100q": "QA-top1 hardening, MiniLM",
        "qa_deberta_v3_base_100q": "QA-top1 hardening, DeBERTa",
        "qa_roberta_base_100q": "QA-top1 hardening, RoBERTa",
    }
    for item in qa_swap:
        rows.append(
            {
                "method": name_map.get(item["variant"], item["variant"]),
                "queries": item["n"],
                "ASR_%": pct(item["asr_poison_rate"]),
                "correct_%": pct(item["correct_rate"]),
                "unknown_%": pct(item["unknown_rate"]),
                "other_%": pct(item["other_rate"]),
                "note": "same Llama3 judging heuristic",
            }
        )
    proposal = load_json(
        "retrieval_framework/results/candidate_proposal_hotpot_deberta_llama3_100q/"
        "proposal_hotpot_deberta_llama3_eval_summary.json"
    )
    rows.append(
        {
            "method": "CARE-RAG full, Hotpot-trained proposal",
            "queries": proposal["n"],
            "ASR_%": pct(proposal["asr_rate"]),
            "correct_%": pct(proposal["correct_rate"]),
            "unknown_%": pct(proposal["unknown_rate"]),
            "other_%": pct(proposal["other_rate"]),
            "note": "strict ASR: poisoned target mention counts as attack success",
        }
    )
    rag2rag = summary_payload(
        "retrieval_framework/results/rag2rag_baseline/nq-dense-rag2rag-llama3-100q.summary.json"
    )
    rows.append(
        {
            "method": "RAG2RAG baseline",
            "queries": rag2rag.get("num_queries"),
            "ASR_%": pct(rag2rag.get("asr_mean")),
            "correct_%": pct(rag2rag.get("correct_mean")),
            "unknown_%": pct(rag2rag.get("unknown_mean")),
            "other_%": "",
            "note": "not directly comparable: very high abstention/suspicious rate",
        }
    )
    paper_hybrid = summary_payload(
        "retrieval_framework/results/nq-paper_hybrid-llama3-Top5-M10x10-alpha0p7-adv-LM_targeted-5.summary.json"
    )
    rows.append(
        {
            "method": "Poisoned RAG attack baseline, paper_hybrid alpha=0.7",
            "queries": paper_hybrid.get("num_queries"),
            "ASR_%": pct(paper_hybrid.get("asr_mean")),
            "correct_%": "",
            "unknown_%": "",
            "other_%": "",
            "note": "attack-only baseline summary provides ASR",
        }
    )
    return rows


def table_clean_utility() -> List[Dict[str, Any]]:
    path = ROOT / "retrieval_framework/results/icde_paper_experiments/clean_utility_summary.csv"
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            run_name = item["run_name"]
            if run_name == "proposal_hotpot_deberta_llama3_clean_100q":
                method = "CARE-RAG full, Hotpot-trained proposal"
            elif "dense-clean" in run_name:
                method = "Dense clean RAG"
            elif "v3-clean" in run_name:
                method = "Prior hardening clean RAG"
            else:
                method = run_name
            rows.append(
                {
                    "method": method,
                    "queries": item["num_queries"],
                    "retrieval_hit@1_%": pct(item["retrieval_hit_at_1"]),
                    "retrieval_hit@3_%": pct(item["retrieval_hit_at_3"]),
                    "retrieval_hit@5_%": pct(item["retrieval_hit_at_5"]),
                    "answer_contains_accuracy_%": pct(item["answer_contains_accuracy"]),
                    "nonempty_output_%": pct(item["nonempty_output_rate"]),
                }
            )
    return rows


def table_care_rag_ablation() -> List[Dict[str, Any]]:
    path = OUT / "ablations/retrieval_only/ablation_summary.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_report(tables: Dict[str, List[Dict[str, Any]]]) -> None:
    clean_row = next(
        row for row in tables["clean_utility"] if row["method"].startswith("CARE-RAG")
    )
    report = f"""# ICDE Experiment Package: CARE-RAG

This directory collects the experiments needed to support the current method:

1. Small QA/proposal models are unreliable as final answer predictors.
2. Candidate proposal should be evaluated by recall, not top-1 exact accuracy.
3. Multi-source answer candidates plus answer-level support sharply reduce poisoned evidence in the final retrieved context.
4. The final local Llama3 ASR is reduced on the 100-query poisoned NQ evaluation.
5. Clean no-attack utility remains comparable to prior clean RAG runs.

## Key Results

- HotpotQA-trained proposal model top-1 correct rate: {tables['small_model_reliability'][-1]['top1_correct_%']}%.
- HotpotQA-trained proposal model top-5 gold candidate recall: {tables['small_model_reliability'][-1]['top5_gold_candidate_recall_%']}%.
- Top-30 multi-source candidate pool gold R@100: {next(r for r in tables['candidate_recall'] if r['retrieved_depth'] == 30 and r['proposal_model'].startswith('Hotpot') and r['candidate_pool'] == 'multi_source')['gold_R@100_%']}%.
- Full hardening contam@5: {tables['retrieval_hardening'][-1]['contam@5_%']}%.
- Full Llama3 ASR: {next(r for r in tables['final_llm_asr'] if r['method'].startswith('CARE-RAG'))['ASR_%']}%.
- Clean answer-containing accuracy: {clean_row['answer_contains_accuracy_%']}%.

## Tables

- `tables/table1_training_provenance.csv`
- `tables/table2_small_model_reliability.csv`
- `tables/table3_candidate_recall.csv`
- `tables/table4_retrieval_hardening.csv`
- `tables/table5_component_ablation_retrieval.csv`
- `tables/table6_final_llm_asr.csv`
- `tables/table7_clean_utility.csv`
- `tables/table8_care_rag_ablation_retrieval.csv`

## Interpretation For Paper

The small model should not be described as a final QA oracle. Its top-1 accuracy is too low for that claim. The supported claim is that it is a noisy candidate proposal module. The method becomes robust because candidate generation is separated from answer verification: candidate proposal optimizes recall, while evidence hardening performs answer-level support aggregation, conflict checks, query-echo filtering, and margin/top-1 dominance gating before the local Llama3 receives context.

"""
    (OUT / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    for generated_dir in (OUT / "tables", OUT / "raw"):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
        generated_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "training_provenance": table_training_provenance(),
        "small_model_reliability": table_small_model_reliability(),
        "candidate_recall": table_candidate_recall(),
        "retrieval_hardening": table_retrieval_hardening(),
        "component_ablation": table_component_ablation(),
        "final_llm_asr": table_final_llm_asr(),
        "clean_utility": table_clean_utility(),
        "care_rag_ablation": table_care_rag_ablation(),
    }

    table_files = {
        "training_provenance": "table1_training_provenance.csv",
        "small_model_reliability": "table2_small_model_reliability.csv",
        "candidate_recall": "table3_candidate_recall.csv",
        "retrieval_hardening": "table4_retrieval_hardening.csv",
        "component_ablation": "table5_component_ablation_retrieval.csv",
        "final_llm_asr": "table6_final_llm_asr.csv",
        "clean_utility": "table7_clean_utility.csv",
        "care_rag_ablation": "table8_care_rag_ablation_retrieval.csv",
    }
    for key, filename in table_files.items():
        write_csv(OUT / "tables" / filename, tables[key])

    manifest: List[Dict[str, str]] = []
    raw_sources = {
        "training_model": [
            "retrieval_framework/results/candidate_proposal_hotpot_deberta_5000/"
            "labeled_examples_metadata.json",
            "retrieval_framework/results/candidate_proposal_hotpot_deberta_5000/"
            "training_history.json",
        ],
        "small_model_reliability": [
            "retrieval_framework/results/icde_paper_experiments/small_model_reliability_100q/"
            "qa_extractor_precision_summary.json",
            "retrieval_framework/results/icde_paper_experiments/small_model_reliability_100q/"
            "qa_extractor_precision_summary.csv",
        ],
        "candidate_recall": [
            "retrieval_framework/results/candidate_pool_recall_top5_deberta_recall100/"
            "candidate_pool_recall_summary.json",
            "retrieval_framework/results/candidate_pool_recall_top5_hotpot_proposal_deberta/"
            "candidate_pool_recall_summary.json",
            "retrieval_framework/results/candidate_pool_recall_top30_base_deberta/"
            "candidate_pool_recall_summary.json",
            "retrieval_framework/results/candidate_pool_recall_top30_hotpot_proposal_deberta/"
            "candidate_pool_recall_summary.json",
        ],
        "retrieval_hardening": [
            "retrieval_framework/results/candidate_proposal_hotpot_deberta_hardening_100q/"
            "proposal_hotpot_deberta_retrieval_100q.summary.json",
            "retrieval_framework/results/candidate_proposal_hotpot_deberta_retrieval_comparison.csv",
        ],
        "final_llm_asr": [
            "retrieval_framework/results/candidate_proposal_hotpot_deberta_llama3_100q/"
            "proposal_hotpot_deberta_llama3_100q.summary.json",
            "retrieval_framework/results/candidate_proposal_hotpot_deberta_llama3_100q/"
            "proposal_hotpot_deberta_llama3_eval_summary.json",
            "retrieval_framework/results/candidate_proposal_hotpot_deberta_llama3_100q/"
            "proposal_hotpot_deberta_llama3_eval_rows.csv",
            "retrieval_framework/results/qa_model_swap_100q/"
            "qa_model_swap_llama3_generation_with_minilm_summary.json",
        ],
        "component_ablation": [
            "retrieval_framework/results/missing_experiments/unified_component_ablation_50k/"
            "summary_table.csv",
        ],
        "care_rag_ablation": [
            "retrieval_framework/results/icde_paper_experiments/ablations/retrieval_only/"
            "ablation_summary.csv",
            "retrieval_framework/results/icde_paper_experiments/ablations/retrieval_only/"
            "ablation_summary.json",
        ],
        "clean_utility": [
            "retrieval_framework/results/icde_paper_experiments/clean_utility_summary.csv",
            "retrieval_framework/results/icde_paper_experiments/clean_utility_hotpot_proposal_100q/"
            "proposal_hotpot_deberta_llama3_clean_100q.summary.json",
        ],
    }
    for label, sources in raw_sources.items():
        for source in sources:
            copy_raw(label, source, manifest)
    write_csv(OUT / "manifest.csv", manifest)
    write_report(tables)
    print(f"Wrote ICDE experiment package to {OUT}")


if __name__ == "__main__":
    main()
