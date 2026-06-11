"""RAGDefender using the OFFICIAL package (SecAI-Lab/RAGDefender, ACSAC'25).

We import the official `ragdefender.RAGDefender` UNMODIFIED (by path, no
re-implementation), run its `defend(query, R)` post-retrieval filter on our
PoisonedRAG 100q input, then feed the surviving passages to our Llama with the
project's standard RAG prompt for answer generation.

Faithfulness contract:
  - Defense (poison detection + filtering): official `RAGDefender.defend`,
    unmodified, imported from baselines/RAGDefender-official/.
  - task_type per the official guidance: single_hop for NQ/MSMARCO,
    multi_hop for HotpotQA.
  - Answer generation: our Llama with the same `wrap_prompt` template used by
    every method in this project (fair, identical generation step).
  - Same retrieval, same attack data, same model as all our runs.

Official repo commit: `git -C baselines/RAGDefender-official rev-parse HEAD`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OFFICIAL = REPO / 'baselines' / 'RAGDefender-official'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(OFFICIAL))                       # import official pkg by path
sys.path.insert(0, str(REPO / 'baselines' / 'common'))

from ragdefender import RAGDefender  # official, unmodified  # noqa: E402
from baseline_common import load_input, load_llm, write_outputs, base_record  # noqa: E402
from src.prompts import wrap_prompt  # project standard RAG prompt  # noqa: E402

# Official guidance: NQ & MS MARCO -> single_hop; HotpotQA -> multi_hop.
TASK_TYPE = {'nq': 'single_hop', 'hotpotqa': 'multi_hop', 'msmarco': 'single_hop'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-path', required=True)
    ap.add_argument('--dataset', required=True,
                    choices=['nq', 'hotpotqa', 'msmarco'])
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--model-config',
                    default=str(REPO / 'model_configs' / 'llama3_config.json'))
    ap.add_argument('--max-new-tokens', type=int, default=150)
    ap.add_argument('--embedder-device', default='auto',
                    help='device for RAGDefender embedder (auto/cuda/cpu)')
    ap.add_argument('--max-queries', type=int, default=None)
    args = ap.parse_args()

    queries = load_input(args.input_path)
    if args.max_queries:
        queries = queries[: args.max_queries]
    print(f'[RAGDefender-official] {len(queries)} queries '
          f'(task_type={TASK_TYPE[args.dataset]})')

    # --- official defender (unmodified) ---
    defender = RAGDefender(
        task_type=TASK_TYPE[args.dataset],
        device=args.embedder_device,
    )

    # --- our LLM for the (identical-to-all-methods) generation step ---
    llm = load_llm(args.model_config, args.max_new_tokens)
    print('[RAGDefender-official] LLM ready')

    records = []
    t0 = time.time()
    total_removed = 0
    for i, q in enumerate(queries, start=1):
        R = [d['text'] for d in q['retrieved']]
        safe, removed = defender.defend(query=q['question'], R=R,
                                        return_indices=True)
        total_removed += len(removed)
        # If the defender removed everything, fall back to empty context (the
        # LLM will answer from parametric memory or say "I don't know").
        contexts = safe if safe else []
        prompt = wrap_prompt(q['question'], contexts, prompt_id=4)
        answer = llm.query(prompt)
        records.append(base_record(
            q, answer,
            ragdefender_kept=len(safe),
            ragdefender_removed_indices=sorted(removed),
        ))
        if i % 20 == 0:
            print(f'  {i}/{len(queries)} ({time.time()-t0:.0f}s) '
                  f'avg_removed={total_removed/i:.2f}')

    out = write_outputs(
        args.output_dir, args.run_name, 'RAGDefender (official)',
        {
            'paper': 'arXiv:2511.01268 (ACSAC 2025)',
            'official_repo': 'github.com/SecAI-Lab/RAGDefender',
            'official_code': 'ragdefender.RAGDefender.defend (unmodified)',
            'task_type': TASK_TYPE[args.dataset],
            'note': 'official post-retrieval filter; answer generation uses '
                    'project-standard wrap_prompt with our Llama (same as all '
                    'methods)',
            'model_config': args.model_config,
            'max_new_tokens': args.max_new_tokens,
            'n_queries': len(queries),
            'avg_removed_per_query': total_removed / len(queries),
        },
        records,
    )
    print(f'[RAGDefender-official] Wrote {out} in {time.time()-t0:.0f}s '
          f'| avg removed/query = {total_removed/len(queries):.2f}')


if __name__ == '__main__':
    main()
