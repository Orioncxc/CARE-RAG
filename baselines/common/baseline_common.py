"""Shared utilities for prompt-based RAG defense baselines.

Used by AstuteRAG and InstructRAG runners. Loads the same Llama model,
reads the same vanilla top-5 retrieved inputs (with PoisonedRAG attack)
that we feed to SeCon-RAG, and writes outputs in our standard per-query
format for downstream labeling with paper_v2/analysis/label_outputs.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_input(path: str) -> List[Dict[str, Any]]:
    """Load vanilla top-5 retrieval output (our framework format)."""
    with open(path) as f:
        data = json.load(f)
    all_queries = []
    for batch in data['results']:
        for it_key, queries in batch.items():
            for q in queries:
                all_queries.append(q)
    return all_queries


def load_llm(model_config_path: str, max_new_tokens: int):
    """Load the target LLM via the project's model factory.

    Sets max_output_tokens so prompt-based defenses with verbose reasoning
    are not truncated. device='auto' in the config picks CUDA on the 3090,
    MPS on Mac, CPU otherwise.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from src.models import create_model
    llm = create_model(model_config_path)
    if hasattr(llm, 'max_output_tokens'):
        llm.max_output_tokens = max_new_tokens
    return llm


def write_outputs(
    out_dir: str,
    run_name: str,
    method: str,
    config_meta: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> str:
    """Write per-query records in the standard format for labeling."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_json = os.path.join(out_dir, f'{run_name}.json')
    with open(out_json, 'w') as f:
        json.dump({
            'config': {'method': method, **config_meta},
            'results': records,
        }, f, indent=2)
    return out_json


def base_record(q: Dict[str, Any], output_text: str, **extra) -> Dict[str, Any]:
    """Build a per-query record with the fields label_outputs.py expects."""
    rec = {
        'id': q['id'],
        'question': q['question'],
        'answer': q['answer'],
        'incorrect_answer': q['incorrect_answer'],
        'target_adv_hits': q.get('target_adv_hits', 0),
        'retrieved': q['retrieved'],
        'output_poison': output_text,
    }
    rec.update(extra)
    return rec
