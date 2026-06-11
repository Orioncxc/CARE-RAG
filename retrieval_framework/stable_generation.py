import re
import unicodedata
from typing import Any, Dict, Optional


class StableLLMGenerator:
    """Run LLM generation with a deterministic fallback for unstable sampling."""

    def __init__(self, llm: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.llm = llm
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.fallback_greedy = bool(self.config.get("fallback_greedy", True))
        self.retry_empty_with_greedy = bool(
            self.config.get("retry_empty_with_greedy", True)
        )
        self.retry_unstable_with_greedy = bool(
            self.config.get("retry_unstable_with_greedy", True)
        )
        self.detect_gibberish = bool(
            self.config.get("detect_gibberish", True)
        )
        self.remove_invalid_values = bool(
            self.config.get("remove_invalid_values", True)
        )
        self.renormalize_logits = bool(self.config.get("renormalize_logits", True))
        self.force_greedy = bool(self.config.get("force_greedy", True))
        self.suppress_transformers_warnings = bool(
            self.config.get("suppress_transformers_warnings", True)
        )
        self.gibberish_min_chars = int(self.config.get("gibberish_min_chars", 80))
        self.max_new_tokens = self.config.get("max_new_tokens")
        self.trim_repeated_suffix = bool(
            self.config.get("trim_repeated_suffix", True)
        )
        self.repeated_suffix_min_run = int(
            self.config.get("repeated_suffix_min_run", 8)
        )
        self.repetition_penalty = float(self.config.get("repetition_penalty", 1.0))
        self.no_repeat_ngram_size = int(self.config.get("no_repeat_ngram_size", 0))

    def query(self, prompt: str) -> Dict[str, Any]:
        if not self.enabled or not self._supports_direct_generation():
            response = self.llm.query(prompt)
            return {
                "response": response,
                "diagnostic": {
                    "stable_generation_enabled": False,
                    "attempts": 1,
                    "fallback_used": False,
                    "error": None,
                    "failed": False,
                    "empty_output": response == "",
                    "gibberish_output": self._looks_gibberish(response),
                },
            }

        first_do_sample = self._should_sample() and not self.force_greedy
        first = self._try_generate(prompt, do_sample=first_do_sample)
        first_quality = self._quality(first.get("response", ""))
        should_retry_empty = (
            first["ok"]
            and first_quality["empty_output"]
            and self.retry_empty_with_greedy
            and first_do_sample
        )
        should_retry_gibberish = (
            first["ok"]
            and first_quality["gibberish_output"]
            and self.retry_unstable_with_greedy
            and first_do_sample
        )
        if first["ok"] and not should_retry_empty and not should_retry_gibberish:
            return {
                "response": first["response"],
                "diagnostic": self._diagnostic(first, attempts=1),
            }

        if not self.fallback_greedy:
            return {
                "response": first.get("response", ""),
                "diagnostic": self._diagnostic(first, attempts=1),
            }

        second = self._try_generate(prompt, do_sample=False)
        diagnostic = self._diagnostic(second, attempts=2)
        diagnostic["fallback_used"] = True
        if not first["ok"]:
            diagnostic["first_error"] = first["error"]
        elif first_quality["empty_output"]:
            diagnostic["first_empty_output"] = True
        elif first_quality["gibberish_output"]:
            diagnostic["first_gibberish_output"] = True
            diagnostic["first_gibberish_reasons"] = first_quality["gibberish_reasons"]
        return {"response": second.get("response", ""), "diagnostic": diagnostic}

    def _supports_direct_generation(self) -> bool:
        return (
            hasattr(self.llm, "model")
            and hasattr(self.llm, "tokenizer")
            and hasattr(self.llm, "_Llama__build_model_inputs")
            and hasattr(self.llm, "_Llama__eos_token_ids")
        )

    def _should_sample(self) -> bool:
        return bool(getattr(self.llm, "do_sample", True)) and float(
            getattr(self.llm, "temperature", 0.0)
        ) > 0.0

    def _try_generate(self, prompt: str, do_sample: bool) -> Dict[str, Any]:
        hf_logging = None
        old_verbosity = None
        try:
            import torch
            if self.suppress_transformers_warnings:
                from transformers.utils import logging as hf_logging

                old_verbosity = hf_logging.get_verbosity()
                hf_logging.set_verbosity_error()

            build_inputs = getattr(self.llm, "_Llama__build_model_inputs")
            eos_token_ids = getattr(self.llm, "_Llama__eos_token_ids")
            model_inputs = build_inputs(prompt)
            max_new_tokens = (
                int(self.max_new_tokens)
                if self.max_new_tokens is not None
                else int(getattr(self.llm, "max_output_tokens", 150))
            )
            generate_kwargs = {
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "eos_token_id": eos_token_ids(),
                "pad_token_id": self.llm.tokenizer.pad_token_id,
                "remove_invalid_values": self.remove_invalid_values,
                "renormalize_logits": self.renormalize_logits,
            }
            if self.repetition_penalty > 1.0:
                generate_kwargs["repetition_penalty"] = self.repetition_penalty
            if self.no_repeat_ngram_size > 0:
                generate_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
            if do_sample:
                generate_kwargs["temperature"] = float(
                    getattr(self.llm, "temperature", 0.1)
                )
                generate_kwargs["top_p"] = float(getattr(self.llm, "top_p", 0.9))
                generate_kwargs["top_k"] = int(getattr(self.llm, "top_k", 50))

            with torch.inference_mode():
                outputs = self.llm.model.generate(**model_inputs, **generate_kwargs)
            input_len = model_inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]
            response = self.llm.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            response, repair = self._repair_response(response)
            return {
                "ok": True,
                "response": response,
                "repair": repair,
                "do_sample": do_sample,
                "error": None,
            }
        except Exception as exc:
            return {
                "ok": False,
                "response": "",
                "do_sample": do_sample,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        finally:
            if hf_logging is not None and old_verbosity is not None:
                hf_logging.set_verbosity(old_verbosity)

    def _diagnostic(self, result: Dict[str, Any], attempts: int) -> Dict[str, Any]:
        quality = self._quality(result.get("response", ""))
        return {
            "stable_generation_enabled": True,
            "attempts": attempts,
            "fallback_used": False,
            "do_sample": result.get("do_sample"),
            "error": result.get("error"),
            "repair": result.get("repair"),
            "failed": not bool(result.get("ok")),
            "empty_output": quality["empty_output"],
            "gibberish_output": quality["gibberish_output"],
            "gibberish_reasons": quality["gibberish_reasons"],
            "remove_invalid_values": self.remove_invalid_values,
            "renormalize_logits": self.renormalize_logits,
            "retry_empty_with_greedy": self.retry_empty_with_greedy,
            "retry_unstable_with_greedy": self.retry_unstable_with_greedy,
            "force_greedy": self.force_greedy,
            "suppress_transformers_warnings": self.suppress_transformers_warnings,
        }

    def _repair_response(self, response: str) -> tuple[str, Optional[Dict[str, Any]]]:
        if not self.trim_repeated_suffix:
            return response, None

        stripped = response.strip()
        if not stripped:
            return stripped, None

        pattern = re.compile(
            r"(?P<prefix>.*?)(?P<char>[^\w\s])(?P=char){"
            + str(max(1, self.repeated_suffix_min_run - 1))
            + r",}\s*$",
            re.DOTALL,
        )
        match = pattern.match(stripped)
        if not match:
            symbol_tail = self._symbol_tail_match(stripped)
            if symbol_tail is None:
                return stripped, None
            repaired = (symbol_tail.group("prefix") + symbol_tail.group("tail")[0]).strip()
            return repaired, {
                "type": "trim_symbol_tail",
                "original_length": len(stripped),
                "repaired_length": len(repaired),
                "tail_prefix": symbol_tail.group("tail")[:16],
            }

        repaired = (match.group("prefix") + match.group("char")).strip()
        return repaired, {
            "type": "trim_repeated_suffix",
            "original_length": len(stripped),
            "repaired_length": len(repaired),
            "repeated_char": match.group("char"),
        }

    def _quality(self, response: str) -> Dict[str, Any]:
        text = response or ""
        reasons = self._gibberish_reasons(text) if self.detect_gibberish else []
        return {
            "empty_output": text == "",
            "gibberish_output": bool(reasons),
            "gibberish_reasons": reasons,
        }

    def _looks_gibberish(self, response: str) -> bool:
        if not self.detect_gibberish:
            return False
        return bool(self._gibberish_reasons(response or ""))

    def _gibberish_reasons(self, text: str) -> list[str]:
        stripped = text.strip()
        reasons: list[str] = []

        repeated_suffix = self._repeated_suffix_match(stripped)
        if repeated_suffix is not None:
            reasons.append("repeated_symbol_suffix")
        if self._symbol_tail_match(stripped) is not None:
            reasons.append("symbol_tail")
        early_code_marker_count = self._code_marker_count(stripped)
        if early_code_marker_count >= 1:
            reasons.append("code_like_fragments")

        if len(stripped) < self.gibberish_min_chars:
            return reasons

        chars = [ch for ch in stripped if not ch.isspace()]
        if not chars:
            return reasons

        non_ascii_ratio = sum(ord(ch) > 127 for ch in chars) / len(chars)
        symbol_ratio = sum(self._is_symbol_or_control(ch) for ch in chars) / len(chars)
        punctuation_ratio = sum(self._is_punctuation(ch) for ch in chars) / len(chars)
        code_marker_count = self._code_marker_count(stripped)
        script_count = len(self._non_latin_scripts(stripped))
        longest_no_space = max((len(part) for part in re.split(r"\s+", stripped)), default=0)

        if non_ascii_ratio > 0.18 and script_count >= 2:
            reasons.append("mixed_non_latin_scripts")
        if symbol_ratio > 0.22:
            reasons.append("high_symbol_density")
        if punctuation_ratio > 0.35:
            reasons.append("high_punctuation_density")
        if code_marker_count >= 3:
            reasons.append("code_like_fragments")
        elif self._code_marker_count(stripped) >= 1 and punctuation_ratio > 0.12:
            reasons.append("url_or_control_fragment")
        if longest_no_space > 90:
            reasons.append("very_long_unbroken_token")

        return reasons

    def _is_symbol_or_control(self, ch: str) -> bool:
        category = unicodedata.category(ch)
        return category.startswith("S") or category.startswith("C")

    def _is_punctuation(self, ch: str) -> bool:
        return unicodedata.category(ch).startswith("P")

    def _rough_punctuation_ratio(self, text: str) -> float:
        chars = [ch for ch in text if not ch.isspace()]
        if not chars:
            return 0.0
        return sum(self._is_punctuation(ch) for ch in chars) / len(chars)

    def _repeated_suffix_match(self, text: str) -> Optional[re.Match[str]]:
        if not text:
            return None
        pattern = re.compile(
            r"([^\w\s])\1{"
            + str(max(1, self.repeated_suffix_min_run - 1))
            + r",}\s*$",
            re.DOTALL,
        )
        return pattern.search(text)

    def _symbol_tail_match(self, text: str) -> Optional[re.Match[str]]:
        if not text:
            return None
        pattern = re.compile(r"(?P<prefix>.*?)(?P<tail>[^\w\s]{3,}.*)$", re.DOTALL)
        match = pattern.search(text)
        if match is None:
            return None

        tail = match.group("tail")
        if len(tail) < self.repeated_suffix_min_run:
            return None

        chars = [ch for ch in tail if not ch.isspace()]
        if not chars:
            return None

        symbol_or_punct = sum(
            self._is_symbol_or_control(ch) or self._is_punctuation(ch)
            for ch in chars
        )
        alnum = sum(ch.isalnum() for ch in chars)
        if symbol_or_punct / len(chars) >= 0.5 and alnum / len(chars) <= 0.4:
            return match
        return None

    def _non_latin_scripts(self, text: str) -> set[str]:
        scripts: set[str] = set()
        for ch in text:
            name = unicodedata.name(ch, "")
            if any(script in name for script in ["CJK", "HIRAGANA", "KATAKANA"]):
                scripts.add("cjk")
            elif "HANGUL" in name:
                scripts.add("hangul")
            elif "ARABIC" in name:
                scripts.add("arabic")
            elif "CYRILLIC" in name:
                scripts.add("cyrillic")
            elif "THAI" in name:
                scripts.add("thai")
            elif "GREEK" in name:
                scripts.add("greek")
            elif "DEVANAGARI" in name:
                scripts.add("devanagari")
            elif "HEBREW" in name:
                scripts.add("hebrew")
        return scripts

    def _code_marker_count(self, text: str) -> int:
        markers = [
            r"GetComponent",
            r"NSIndex",
            r"NSString",
            r"SQLException",
            r"Throwable",
            r"Controller",
            r"APIView",
            r"nullptr",
            r"::",
            r"</",
            r"://",
            r"&#\d+",
            r"PLUGIN",
            r"usercontent",
            r"formatassistant",
            r"INTERRUPTION",
            r"\{",
            r"\}",
            r"\);",
            r"_[A-Za-z]{3,}",
        ]
        return sum(1 for marker in markers if re.search(marker, text))
