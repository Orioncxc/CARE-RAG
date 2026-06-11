from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_framework.train_domain_qa_extractor import (  # noqa: E402
    compact_context,
    find_span,
    read_jsonl,
    read_qrels,
    train_model,
)


def answer_from_query(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    answer = metadata.get("answer") if isinstance(metadata, dict) else None
    return str(answer or "").strip()


def build_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "labeled_examples.json"
    if cache_path.exists() and not args.rebuild:
        examples = json.loads(cache_path.read_text())
        print(f"Loaded cached proposal examples: {len(examples)} from {cache_path}")
        return examples[: args.num_labels]

    queries = read_jsonl(data_dir / "queries.jsonl")
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    qrels = read_qrels(data_dir / "qrels" / f"{args.split}.tsv")

    qids = list(qrels.keys())
    random.Random(args.seed).shuffle(qids)

    examples: List[Dict[str, Any]] = []
    attempts = 0
    skipped_no_answer = 0
    skipped_no_span = 0
    for qid in qids:
        if len(examples) >= args.num_labels:
            break
        query = queries.get(qid)
        if not query:
            continue
        answer = answer_from_query(query)
        if not answer or answer.lower() in {"yes", "no", "noanswer", "unknown"}:
            skipped_no_answer += 1
            continue
        docs = list(qrels.get(qid, []))
        random.Random(args.seed + len(examples)).shuffle(docs)
        added_for_query = 0
        for doc_id in docs:
            if len(examples) >= args.num_labels:
                break
            if args.max_docs_per_query > 0 and added_for_query >= args.max_docs_per_query:
                break
            doc = corpus.get(doc_id)
            if not doc:
                continue
            attempts += 1
            context = compact_context(
                f"{doc.get('title', '')}. {doc.get('text', '')}",
                args.context_chars,
            )
            span = find_span(context, answer)
            if span is None:
                skipped_no_span += 1
                continue
            start, end, exact = span
            examples.append(
                {
                    "id": f"{qid}::{doc_id}",
                    "query_id": qid,
                    "doc_id": doc_id,
                    "question": query["text"],
                    "context": context,
                    "answer_text": exact,
                    "answer_start": start,
                    "answer_end": end,
                    "labeler_output": answer,
                    "source": f"{args.dataset_name}_{args.split}_exact_answer",
                }
            )
            added_for_query += 1
            if len(examples) % 500 == 0:
                print(f"built {len(examples)}/{args.num_labels} after {attempts} qrel attempts")

    cache_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False))
    metadata = {
        "num_examples": len(examples),
        "attempts": attempts,
        "skipped_no_answer": skipped_no_answer,
        "skipped_no_span": skipped_no_span,
        "seed": args.seed,
        "source_data_dir": args.data_dir,
        "split": args.split,
        "dataset_name": args.dataset_name,
    }
    (output_dir / "labeled_examples_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="datasets/hotpotqa")
    parser.add_argument("--dataset_name", default="hotpotqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="retrieval_framework/results/candidate_proposal_hotpot_deberta_5000")
    parser.add_argument("--base_model", default="deepset/deberta-v3-base-squad2")
    parser.add_argument("--num_labels", type=int, default=5000)
    parser.add_argument("--dev_size", type=int, default=500)
    parser.add_argument("--max_docs_per_query", type=int, default=2)
    parser.add_argument("--context_chars", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    examples = build_examples(args)
    if len(examples) < 100:
        raise RuntimeError(f"Too few proposal examples: {len(examples)}")
    train_model(args, examples)


if __name__ == "__main__":
    main()
