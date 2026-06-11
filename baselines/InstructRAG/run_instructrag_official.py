"""InstructRAG-ICL using the OFFICIAL repo code (weizhepei/InstructRAG).

We import the official prompt construction (`format_prompt`, `build_contexts`,
`normalize_question`) and the official `rag.json` prompt_dict UNMODIFIED from
baselines/InstructRAG-official/, then run them on our PoisonedRAG 100q input
with our Llama backbone.

InstructRAG-ICL is training-free and needs few-shot demonstrations that are
*self-synthesized* by the model (official `do_rationale_generation` path, using
the official `rationale_generation_instruction`). We synthesize K demos from a
small held-out slice of the dataset's own queries (their retrieved docs + gold
answers) exactly as the official pipeline does, then evaluate on the 100
poisoned queries.

Faithfulness contract:
  - Prompt construction, instructions, and prompt_dict: official, unmodified.
  - Demonstrations: self-synthesized via the official rationale-generation
    instruction (the InstructRAG mechanism), using our Llama.
  - LLM backend swapped from vLLM (CUDA-only) to our transformers Llama, with
    a raw-prompt generate path so the official chat-formatted prompt is fed
    verbatim (no double chat-templating).
  - Same retrieval, same attack data, same model family as all our runs.

Official repo commit: see `git -C baselines/InstructRAG-official rev-parse HEAD`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OFFICIAL = REPO / 'baselines' / 'InstructRAG-official'
sys.path.insert(0, str(REPO))
# Put official src/ on the path so its internal `import log_utils, common_utils`
# (sibling-module style) resolves exactly as in the official repo.
sys.path.insert(0, str(OFFICIAL / 'src'))
sys.path.insert(0, str(REPO / 'baselines' / 'common'))

import torch  # noqa: E402
from baseline_common import load_input, write_outputs, base_record  # noqa: E402

# --- OFFICIAL InstructRAG code (unmodified) ---
import data_utils as official_data_utils  # noqa: E402
import common_utils as official_common_utils  # noqa: E402

# Map our dataset keys to the official dataset_name used for the
# rationale-generation postfix in rag.json.
DATASET_NAME_MAP = {
    'nq': 'NaturalQuestions',
    'hotpotqa': '2WikiMultiHopQA',   # both are multi-hop; closest official postfix
    'msmarco': 'NaturalQuestions',   # short-answer open-domain; closest postfix
}


def to_instructrag_example(q: dict, n_docs: int = 5) -> dict:
    """Convert our retrieval record to the InstructRAG example schema."""
    ctxs = []
    for d in q['retrieved'][:n_docs]:
        ctxs.append({
            'title': d.get('metadata', {}).get('title', '') if isinstance(
                d.get('metadata'), dict) else '',
            'text': d.get('text', ''),
            'score': float(d.get('score', 0.0)),
        })
    # build_contexts indexes ctxs[0] and ctxs[1] scores; ensure >=2 entries
    while len(ctxs) < 2:
        ctxs.append({'title': '', 'text': '', 'score': 0.0})
    return {
        'question': q['question'],
        'answers': [q['answer']],
        'ctxs': ctxs,
    }


def raw_generate(llm, prompt_str: str, max_new_tokens: int) -> str:
    """Generate from a pre-formatted prompt WITHOUT re-applying chat template.

    The official format_prompt already emits the full Llama-3 chat-formatted
    string (with <|start_header_id|> etc.). We tokenize it directly and decode
    only the newly generated tokens.
    """
    tok = llm.tokenizer
    model = llm.model
    device = next(model.parameters()).device
    inputs = tok(prompt_str, return_tensors='pt', add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(tok.eos_token_id)
    eot = tok.convert_tokens_to_ids('<|eot_id|>')
    if isinstance(eot, int) and eot not in eos_ids:
        eos_ids.append(eot)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=eos_ids or None,
        )
    new_tokens = out[0][inputs['input_ids'].shape[-1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def synthesize_demos(llm, demo_examples, dataset_name, prompt_dict,
                     n_docs, max_new_tokens):
    """Official self-synthesis: generate a rationale per demo example using the
    official rationale-generation instruction (do_rationale_generation=True)."""
    demos = []
    for ex in demo_examples:
        ex = dict(ex)
        ex['answers'] = ex.get('answers', [''])
        # official format_prompt with rationale-generation flag
        prompt = official_data_utils.format_prompt(
            dataset_name=dataset_name,
            example=ex,
            n_docs=n_docs,
            prompt_dict=prompt_dict,
            tokenizer=llm.tokenizer,
            do_rationale_generation=True,
            demos=[],
        )
        rationale = raw_generate(llm, prompt, max_new_tokens)
        demos.append({'question': ex['question'], 'rationale': rationale})
    return demos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-path', required=True)
    ap.add_argument('--dataset', required=True, choices=['nq', 'hotpotqa', 'msmarco'])
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--model-config',
                    default=str(REPO / 'model_configs' / 'llama3_config.json'))
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--n-docs', type=int, default=5)
    ap.add_argument('--n-demos', type=int, default=4)
    ap.add_argument('--max-queries', type=int, default=None)
    args = ap.parse_args()

    dataset_name = DATASET_NAME_MAP[args.dataset]
    prompt_dict = official_common_utils.jload(str(OFFICIAL / 'src' / 'rag.json'))

    queries = load_input(args.input_path)
    print(f'[InstructRAG-official] loaded {len(queries)} queries')

    # Demo source = last n_demos queries (held out from eval slice when capped).
    demo_src = queries[-args.n_demos:]
    if args.max_queries:
        eval_queries = queries[: args.max_queries]
    else:
        eval_queries = queries

    from baseline_common import load_llm
    llm = load_llm(args.model_config, args.max_new_tokens)
    print('[InstructRAG-official] LLM ready; synthesizing demos...')

    demo_examples = [to_instructrag_example(q, args.n_docs) for q in demo_src]
    t0 = time.time()
    demos = synthesize_demos(
        llm, demo_examples, dataset_name, prompt_dict,
        args.n_docs, args.max_new_tokens,
    )
    print(f'[InstructRAG-official] synthesized {len(demos)} demos '
          f'({time.time()-t0:.0f}s)')

    records = []
    t0 = time.time()
    for i, q in enumerate(eval_queries, start=1):
        ex = to_instructrag_example(q, args.n_docs)
        prompt = official_data_utils.format_prompt(
            dataset_name=dataset_name,
            example=ex,
            n_docs=args.n_docs,
            prompt_dict=prompt_dict,
            tokenizer=llm.tokenizer,
            do_rationale_generation=False,
            demos=demos,
        )
        full = raw_generate(llm, prompt, args.max_new_tokens)
        records.append(base_record(q, full, instructrag_full_output=full))
        if i % 20 == 0:
            print(f'  {i}/{len(eval_queries)} ({time.time()-t0:.0f}s)')

    out = write_outputs(
        args.output_dir, args.run_name, 'InstructRAG-ICL (official)',
        {
            'paper': 'arXiv:2406.13629 (ICLR 2025)',
            'official_repo': 'github.com/weizhepei/InstructRAG',
            'official_code': 'src/data_utils.format_prompt + rag.json (unmodified)',
            'note': 'InstructRAG-ICL training-free; demos self-synthesized via '
                    'official rationale_generation_instruction with our Llama; '
                    'vLLM backend replaced with transformers (raw-prompt path)',
            'dataset_name_used': dataset_name,
            'n_demos': len(demos),
            'model_config': args.model_config,
            'max_new_tokens': args.max_new_tokens,
            'n_queries': len(eval_queries),
        },
        records,
    )
    print(f'[InstructRAG-official] Wrote {out} in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
