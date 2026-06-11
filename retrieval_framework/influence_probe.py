import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional


def clean_str(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) > 1 and text[-1] == ".":
        text = text[:-1]
    return text.lower()


def iter_query_rows(results: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for block in results:
        if not isinstance(block, dict) or not block:
            continue
        rows = next(iter(block.values()))
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield row


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def default_output_path(result_path: str) -> str:
    if result_path.endswith(".json"):
        return result_path[: -len(".json")] + ".influence.json"
    return result_path + ".influence.json"


def run_probe(
    result_path: str,
    model_config_path: str,
    output_path: Optional[str],
    max_queries: int,
    max_docs: int,
    only_target_success: bool,
) -> Dict[str, Any]:
    from src.models import create_model
    from src.prompts import wrap_prompt

    payload = load_json(result_path)
    rows = list(iter_query_rows(payload.get("results", [])))
    llm = create_model(model_config_path)

    probed_rows: List[Dict[str, Any]] = []
    for row in rows:
        retrieved = row.get("retrieved", [])
        if not retrieved:
            continue
        incorrect = row.get("incorrect_answer", "")
        existing_output = row.get("output_poison")
        if only_target_success and (
            existing_output is None or clean_str(incorrect) not in clean_str(existing_output)
        ):
            continue

        question = row["question"]
        full_contexts = [item.get("text", "") for item in retrieved]
        if existing_output is None:
            existing_output = llm.query(wrap_prompt(question, full_contexts, prompt_id=4))

        doc_effects: List[Dict[str, Any]] = []
        for idx, item in enumerate(retrieved[:max_docs]):
            reduced_contexts = [
                candidate.get("text", "")
                for j, candidate in enumerate(retrieved)
                if j != idx
            ]
            response = llm.query(wrap_prompt(question, reduced_contexts, prompt_id=4))
            doc_effects.append(
                {
                    "removed_doc_id": item.get("doc_id"),
                    "removed_rank": item.get("rank"),
                    "removed_is_adv": item.get("metadata", {}).get("is_adv", False),
                    "response": response,
                    "output_changed": clean_str(response) != clean_str(existing_output),
                    "target_answer_removed": clean_str(incorrect) in clean_str(existing_output)
                    and clean_str(incorrect) not in clean_str(response),
                    "correct_answer_present": clean_str(row.get("answer", ""))
                    in clean_str(response),
                }
            )

        probed_rows.append(
            {
                "id": row.get("id"),
                "question": question,
                "baseline_output": existing_output,
                "incorrect_answer": incorrect,
                "answer": row.get("answer"),
                "doc_effects": doc_effects,
            }
        )
        if max_queries > 0 and len(probed_rows) >= max_queries:
            break

    summary = {
        "num_queries": len(probed_rows),
        "num_doc_effects": sum(len(row["doc_effects"]) for row in probed_rows),
        "target_answer_removed_count": sum(
            effect["target_answer_removed"]
            for row in probed_rows
            for effect in row["doc_effects"]
        ),
        "output_changed_count": sum(
            effect["output_changed"] for row in probed_rows for effect in row["doc_effects"]
        ),
    }
    output = {
        "source_result_path": result_path,
        "model_config_path": model_config_path,
        "summary": summary,
        "results": probed_rows,
    }
    output_path = output_path or default_output_path(result_path)
    dump_json(output_path, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote influence probe to {output_path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leave-one-out document influence probes for a result file."
    )
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--model_config_path", default="model_configs/llama3_config.json")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--max_queries", type=int, default=10)
    parser.add_argument("--max_docs", type=int, default=5)
    parser.add_argument("--only_target_success", type=str, default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_probe(
        result_path=args.result_path,
        model_config_path=args.model_config_path,
        output_path=args.output_path,
        max_queries=args.max_queries,
        max_docs=args.max_docs,
        only_target_success=args.only_target_success.lower() in {"true", "1", "yes", "y"},
    )


if __name__ == "__main__":
    main()

