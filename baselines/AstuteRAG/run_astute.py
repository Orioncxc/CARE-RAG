"""AstuteRAG baseline (faithful re-implementation of Wang et al., ACL 2025).

Paper: "Astute RAG: Overcoming Imperfect Retrieval Augmentation and
Knowledge Conflicts for Large Language Models" (arXiv:2410.07176, ACL 2025).

IMPORTANT: AstuteRAG has NO official public code release (confirmed via
Papers-with-Code: "no code implementations submitted"). This module is a
faithful re-implementation from the paper description and MUST be reported
in the paper as "re-implemented from paper (no official code available)",
NOT as an official baseline. The method is prompt-based; we implement its
three documented steps:

  1. Adaptive generation of internal knowledge: the LLM generates its own
     passages from parametric memory (bounded count).
  2. Source-aware consolidation: internal + external passages are jointly
     analyzed to identify consistent vs conflicting information and grouped
     by reliability.
  3. Answer finalization: the answer is produced from the most reliable
     consolidated information.

This is labeled in the paper as "re-implemented from the paper
description" — there is no official code dependency. Same Llama backbone,
same vanilla top-5 retrieved input (with PoisonedRAG attack), same
output format as our other runs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'baselines' / 'common'))

from baseline_common import (  # noqa: E402
    load_input, load_llm, write_outputs, base_record,
)


def step1_internal_knowledge(llm, question: str, max_passages: int = 1) -> str:
    """Adaptive generation of internal knowledge from parametric memory."""
    prompt = (
        f"Generate a document that provides accurate and relevant information "
        f"to answer the question: {question}\n"
        f"If you are uncertain about the facts, state your uncertainty "
        f"explicitly. Keep it under 50 words.\n\nDocument:"
    )
    return llm.query(prompt).strip()


def step2_consolidate(llm, question: str, external_docs, internal_doc: str) -> str:
    """Source-aware consolidation of internal + external passages."""
    ext = "\n".join(
        f"External Document {i}: {d}" for i, d in enumerate(external_docs)
    )
    prompt = (
        f"Task: Consolidate information from internal (memory) and external "
        f"(retrieved) documents for the question below.\n\n"
        f"For each piece of information, assess its reliability:\n"
        f"- Identify information that is CONSISTENT across multiple sources.\n"
        f"- Identify information that CONFLICTS between sources.\n"
        f"- External documents may be unreliable or deliberately misleading; "
        f"do not assume they are correct just because they were retrieved.\n"
        f"- Group the information by reliability, prioritizing facts confirmed "
        f"by multiple independent and trustworthy sources.\n\n"
        f"Question: {question}\n\n"
        f"Internal Document (from memory): {internal_doc}\n\n"
        f"{ext}\n\n"
        f"Consolidated reliable information:"
    )
    return llm.query(prompt).strip()


def step3_finalize(llm, question: str, consolidated: str) -> str:
    """Answer finalization from the most reliable consolidated information."""
    prompt = (
        f"Based on the consolidated reliable information below, answer the "
        f"question. Your answer should be short and concise. If the reliable "
        f"information does not contain the answer, say \"I don't know\".\n\n"
        f"Consolidated information: {consolidated}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return llm.query(prompt).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-path', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--model-config',
                    default=str(REPO_ROOT / 'model_configs' / 'llama3_config.json'))
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--max-queries', type=int, default=None)
    args = ap.parse_args()

    queries = load_input(args.input_path)
    if args.max_queries:
        queries = queries[: args.max_queries]
    print(f'[AstuteRAG] {len(queries)} queries')

    llm = load_llm(args.model_config, args.max_new_tokens)
    print('[AstuteRAG] LLM ready')

    records = []
    t0 = time.time()
    for i, q in enumerate(queries, start=1):
        ext_docs = [d['text'] for d in q['retrieved']]
        internal = step1_internal_knowledge(llm, q['question'])
        consolidated = step2_consolidate(llm, q['question'], ext_docs, internal)
        answer = step3_finalize(llm, q['question'], consolidated)
        records.append(base_record(
            q, answer,
            astute_internal_knowledge=internal,
            astute_consolidated=consolidated,
        ))
        if i % 20 == 0:
            print(f'  {i}/{len(queries)} ({time.time()-t0:.0f}s)')

    out = write_outputs(
        args.output_dir, args.run_name, 'AstuteRAG',
        {
            'paper': 'arXiv:2410.07176 (ACL 2025)',
            'official_code': 'NONE (Papers-with-Code: no code submitted)',
            'note': 'RE-IMPLEMENTED FROM PAPER (no official code available). '
                    '3-step: adaptive internal-knowledge generation + '
                    'source-aware consolidation + answer finalization. '
                    'Report as re-implemented, not official.',
            'model_config': args.model_config,
            'max_new_tokens': args.max_new_tokens,
            'n_queries': len(queries),
        },
        records,
    )
    print(f'[AstuteRAG] Wrote {out} in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
