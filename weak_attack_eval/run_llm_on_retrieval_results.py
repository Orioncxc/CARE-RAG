from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from retrieval_framework.stable_generation import StableLLMGenerator
from src.prompts import wrap_prompt


REPRESENTATIVE_ATTACKS = [
    "answer_only_misinformation",
    "query_copy_misinformation",
    "multi_doc_consensus",
    "hybrid_keyword_dense_misinformation",
]


def clean_str(value: Any) -> str:
    text = str(value).strip()
    if len(text) > 1 and text[-1] == ".":
        text = text[:-1]
    return text.lower()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def iter_query_rows(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for iter_payload in payload.get("results", []):
        for rows in iter_payload.values():
            for row in rows:
                yield row


def result_path(results_dir: str, condition: str, attack: str) -> str:
    name = f"nq-{attack}-{condition}-100q.json"
    return os.path.join(results_dir, condition, attack, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM ASR evaluation over saved weak-attack retrieval results."
    )
    parser.add_argument(
        "--retrieval_results_dir",
        type=str,
        default="weak_attack_eval/results/defense_eval_100",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weak_attack_eval/results/llm_representative",
    )
    parser.add_argument(
        "--attacks",
        type=str,
        default=",".join(REPRESENTATIVE_ATTACKS),
        help="Comma-separated attack names.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="undefended,defended",
        help="Comma-separated conditions.",
    )
    parser.add_argument("--max_queries", type=int, default=10)
    parser.add_argument("--model_config_path", type=str, default="model_configs/llama3_config.json")
    parser.add_argument("--model_name", type=str, default="llama3")
    return parser.parse_args()


def selected_items(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_one_file(
    *,
    llm_generator: StableLLMGenerator,
    path: str,
    attack: str,
    condition: str,
    model_name: str,
    max_queries: Optional[int],
    output_dir: str,
) -> Dict[str, Any]:
    payload = load_json(path)
    rows = list(iter_query_rows(payload))
    if max_queries is not None and max_queries > 0:
        rows = rows[:max_queries]

    outputs: List[Dict[str, Any]] = []
    asr_hits = 0
    generation_failed = 0
    generation_empty = 0
    for idx, row in enumerate(rows, start=1):
        question = row["question"]
        contexts = [item.get("text", "") for item in row.get("retrieved", [])]
        prompt = wrap_prompt(question, contexts, prompt_id=4)
        generation = llm_generator.query(prompt)
        response = generation["response"]
        diagnostic = generation["diagnostic"]
        incorrect = row.get("incorrect_answer", "")
        asr_hit = clean_str(incorrect) in clean_str(response)
        asr_hits += int(asr_hit)
        generation_failed += int(bool(diagnostic.get("failed")))
        generation_empty += int(bool(diagnostic.get("empty_output")))
        print(
            f"[{attack}/{condition}] {idx}/{len(rows)} "
            f"asr_hit={int(asr_hit)} qid={row.get('id')}"
        )
        outputs.append(
            {
                "id": row.get("id"),
                "question": question,
                "answer": row.get("answer"),
                "incorrect_answer": incorrect,
                "target_adv_hits": row.get("target_adv_hits"),
                "first_adv_rank": row.get("first_adv_rank"),
                "input_prompt": prompt,
                "output_poison": response,
                "asr_hit": asr_hit,
                "generation": diagnostic,
                "retrieved": row.get("retrieved", []),
            }
        )

    total = len(rows)
    summary = {
        "attack_name": attack,
        "condition": condition,
        "model_name": model_name,
        "num_queries": total,
        "asr_hits": asr_hits,
        "asr_mean": asr_hits / total if total else 0.0,
        "generation_failed_count": generation_failed,
        "generation_empty_output_count": generation_empty,
        "source_retrieval_path": path,
    }
    out_name = f"{attack}-{condition}-{model_name}-llm-maxq{total}"
    dump_json(
        os.path.join(output_dir, f"{out_name}.json"),
        {
            "summary": summary,
            "outputs": outputs,
        },
    )
    dump_json(os.path.join(output_dir, f"{out_name}.summary.json"), summary)
    return summary


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "attack_name",
        "condition",
        "model_name",
        "num_queries",
        "asr_hits",
        "asr_mean",
        "generation_failed_count",
        "generation_empty_output_count",
        "source_retrieval_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_comparison_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "attack_name",
        "num_queries",
        "undefended_asr",
        "defended_asr",
        "delta_asr",
    ]
    by_attack: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_attack.setdefault(row["attack_name"], {})[row["condition"]] = row
    comparison_rows: List[Dict[str, Any]] = []
    for attack in sorted(by_attack):
        pair = by_attack[attack]
        if "undefended" not in pair or "defended" not in pair:
            continue
        undefended = float(pair["undefended"].get("asr_mean", 0.0))
        defended = float(pair["defended"].get("asr_mean", 0.0))
        comparison_rows.append(
            {
                "attack_name": attack,
                "num_queries": pair["defended"].get("num_queries", ""),
                "undefended_asr": undefended,
                "defended_asr": defended,
                "delta_asr": undefended - defended,
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from src.models import create_model

    llm = create_model(args.model_config_path)
    llm_generator = StableLLMGenerator(
        llm,
        {
            "enabled": True,
            "fallback_greedy": True,
            "retry_empty_with_greedy": False,
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "force_greedy": False,
            "suppress_transformers_warnings": True,
        },
    )

    summaries: List[Dict[str, Any]] = []
    for attack in selected_items(args.attacks):
        for condition in selected_items(args.conditions):
            path = result_path(args.retrieval_results_dir, condition, attack)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            summary = run_one_file(
                llm_generator=llm_generator,
                path=path,
                attack=attack,
                condition=condition,
                model_name=args.model_name,
                max_queries=args.max_queries,
                output_dir=args.output_dir,
            )
            summaries.append(summary)
            write_summary_csv(os.path.join(args.output_dir, "llm_summary_table.csv"), summaries)
            write_comparison_csv(os.path.join(args.output_dir, "llm_defense_comparison.csv"), summaries)

    print(f"Saved LLM summary to {os.path.join(args.output_dir, 'llm_summary_table.csv')}")
    print(
        "Saved LLM comparison to "
        f"{os.path.join(args.output_dir, 'llm_defense_comparison.csv')}"
    )


if __name__ == "__main__":
    main()

