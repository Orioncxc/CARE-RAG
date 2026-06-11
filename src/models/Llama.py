import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .Model import Model


class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        params = config["params"]
        self.max_output_tokens = int(params["max_output_tokens"])
        self.device = self.__resolve_device(params.get("device", "auto"))
        self.top_p = float(params.get("top_p", 0.9))
        self.top_k = int(params.get("top_k", 50))
        self.do_sample = self.__str_to_bool(params.get("do_sample", "True"))
        self.use_chat_template = self.__str_to_bool(params.get("use_chat_template", "True"))
        self.trust_remote_code = self.__str_to_bool(params.get("trust_remote_code", "False"))
        self.torch_dtype = self.__resolve_dtype(params.get("torch_dtype", "float16"))

        model_name_or_path = config["model_info"].get("local_path", self.name)
        hf_token = self.__get_hf_token(config)
        tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if hf_token:
            tokenizer_kwargs["token"] = hf_token
            model_kwargs["token"] = hf_token

        self.tokenizer = self.__load_tokenizer(model_name_or_path, tokenizer_kwargs, hf_token)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = params.get("device_map", None)
        if device_map:
            # Multi-GPU sharding for long-prompt / large-vocab OOM cases.
            # max_memory_per_gpu caps each GPU's model footprint, forcing the
            # model to spread across N GPUs even when it fits on one — leaving
            # headroom for KV-cache and large logits tensors (e.g. InstructRAG
            # on Qwen2.5: 15k-token prompt × 152k vocab = 9.5GB logits).
            # Set "max_memory_per_gpu": "Xgib" in the model config params.
            model_kwargs["device_map"] = device_map
            max_mem_str = params.get("max_memory_per_gpu", None)
            if max_mem_str:
                import torch
                n_gpus = torch.cuda.device_count()
                model_kwargs["max_memory"] = {i: max_mem_str for i in range(n_gpus)}
                print(
                    f"[Llama] device_map={device_map} visible_cuda_gpus={n_gpus} "
                    f"max_memory_per_gpu={max_mem_str}"
                )
            self.model = self.__load_model(model_name_or_path, model_kwargs, hf_token)
            if hasattr(self.model, "hf_device_map"):
                print(f"[Llama] hf_device_map={self.model.hf_device_map}")
        else:
            self.model = self.__load_model(model_name_or_path, model_kwargs, hf_token).to(self.device)
        self.model.eval()

    def __str_to_bool(self, s):
        if type(s) == bool:
            return s
        if type(s) == str:
            if s.lower() == "true":
                return True
            if s.lower() == "false":
                return False
        raise ValueError(f"{s} is not a valid boolean")

    def __resolve_dtype(self, dtype):
        if dtype == "auto":
            return "auto"
        if dtype == "float16":
            target_dtype = torch.float16
        elif dtype == "bfloat16":
            target_dtype = torch.bfloat16
        elif dtype == "float32":
            target_dtype = torch.float32
        else:
            raise ValueError(f"{dtype} is not a valid torch dtype setting")

        if self.device == "cpu" and target_dtype != torch.float32:
            return torch.float32
        if self.device == "mps" and target_dtype == torch.bfloat16:
            return torch.float16
        return target_dtype

    def __resolve_device(self, device):
        if type(device) != str:
            return "cpu"

        requested = device.lower().strip()
        if requested in ["", "auto"]:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return requested
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if requested == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if requested == "cpu":
            return "cpu"

        return requested

    def __get_hf_token(self, config):
        env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if env_token is not None and len(env_token.strip()) > 0:
            return env_token.strip()

        api_key_info = config.get("api_key_info", {})
        api_pos = int(api_key_info.get("api_key_use", -1))
        api_keys = api_key_info.get("api_keys", [])
        if api_pos == -1:
            return None
        if not (0 <= api_pos < len(api_keys)):
            raise ValueError("Please enter a valid Hugging Face token index to use.")

        token = str(api_keys[api_pos]).strip()
        if token == "" or token.lower().startswith("your "):
            return None
        return token

    def __load_tokenizer(self, model_name_or_path, tokenizer_kwargs, hf_token):
        try:
            return AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        except TypeError:
            if hf_token and "token" in tokenizer_kwargs:
                fallback_kwargs = dict(tokenizer_kwargs)
                fallback_kwargs.pop("token")
                fallback_kwargs["use_auth_token"] = hf_token
                return AutoTokenizer.from_pretrained(model_name_or_path, **fallback_kwargs)
            raise

    def __load_model(self, model_name_or_path, model_kwargs, hf_token):
        try:
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        except TypeError:
            if hf_token and "token" in model_kwargs:
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs.pop("token")
                fallback_kwargs["use_auth_token"] = hf_token
                return AutoModelForCausalLM.from_pretrained(model_name_or_path, **fallback_kwargs)
            raise

    def __move_model_inputs_to_device(self, model_inputs):
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }

    def __build_model_inputs(self, msg):
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": msg}]
            try:
                chat_inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            except TypeError:
                chat_inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

            if isinstance(chat_inputs, torch.Tensor):
                model_inputs = {"input_ids": chat_inputs}
            elif hasattr(chat_inputs, "items"):
                model_inputs = dict(chat_inputs)
            else:
                model_inputs = {"input_ids": torch.as_tensor(chat_inputs)}
        else:
            model_inputs = self.tokenizer(msg, return_tensors="pt")

        return self.__move_model_inputs_to_device(model_inputs)

    def __eos_token_ids(self):
        eos_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_token_ids.append(self.tokenizer.eos_token_id)

        eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_token_id, int) and eot_token_id not in eos_token_ids:
            eos_token_ids.append(eot_token_id)

        if len(eos_token_ids) == 0:
            return None
        if len(eos_token_ids) == 1:
            return eos_token_ids[0]
        return eos_token_ids

    def query(self, msg):
        try:
            model_inputs = self.__build_model_inputs(msg)
            do_sample = self.do_sample and self.temperature > 0
            generate_kwargs = dict(
                do_sample=do_sample,
                max_new_tokens=self.max_output_tokens,
                eos_token_id=self.__eos_token_ids(),
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if do_sample:
                generate_kwargs["temperature"] = self.temperature
                generate_kwargs["top_p"] = self.top_p
                generate_kwargs["top_k"] = self.top_k

            outputs = self.model.generate(
                **model_inputs,
                **generate_kwargs,
            )
            input_len = model_inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            response = ""

        return response.strip()
