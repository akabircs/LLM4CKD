from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def softmax_two(logp0: float, logp1: float) -> tuple[float, float]:
    m = max(logp0, logp1)
    e0 = math.exp(logp0 - m)
    e1 = math.exp(logp1 - m)
    z = e0 + e1
    return e0 / z, e1 / z


class PromptOnlyClassifier:
    """No-op classifier for prompt auditing."""
    def __init__(self):
        self.last_prompt_tokens = None
    def predict_proba(self, prompt_or_messages):
        return 0.5, 0.5


class HuggingFaceConstrainedClassifier:
    """Compute normalized label probabilities P(label=0/1 | prompt) using a causal LM."""

    def __init__(self, model_id: str, device_map: str = "auto", dtype: str | None = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        kwargs = {"device_map": device_map, "trust_remote_code": True}
        if dtype == "auto":
            kwargs["torch_dtype"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.last_prompt_tokens = None

    def _render(self, prompt_or_messages):
        if isinstance(prompt_or_messages, list):
            if hasattr(self.tokenizer, "apply_chat_template"):
                return self.tokenizer.apply_chat_template(
                    prompt_or_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            from .prompts import messages_to_llama_chat_text

            return messages_to_llama_chat_text(prompt_or_messages)
        return str(prompt_or_messages)

    def _sequence_logprob(self, prompt: str, label: str) -> float:
        # Compute log P(label tokens | prompt). A leading space is intentionally not added
        # because the prompt ends with "Output:" and the paper constrains output to 0/1.
        enc_prompt = self.tokenizer(prompt, return_tensors="pt")
        self.last_prompt_tokens = int(enc_prompt.input_ids.shape[1])
        enc_label = self.tokenizer(label, add_special_tokens=False, return_tensors="pt")
        input_ids = self.torch.cat([enc_prompt.input_ids, enc_label.input_ids], dim=1).to(self.model.device)
        attn = self.torch.ones_like(input_ids).to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
        label_ids = enc_label.input_ids[0].to(self.model.device)
        start = enc_prompt.input_ids.shape[1] - 1
        logp = 0.0
        for i, tok_id in enumerate(label_ids):
            token_logits = logits[0, start + i, :]
            token_logprobs = self.torch.log_softmax(token_logits, dim=-1)
            logp += float(token_logprobs[tok_id].detach().cpu())
        return logp

    def predict_proba(self, prompt_or_messages) -> tuple[float, float]:
        prompt = self._render(prompt_or_messages)
        # Try both bare and whitespace-prefixed labels, then marginalize per class.
        label0_variants = ["0", " 0"]
        label1_variants = ["1", " 1"]
        logp0s = [self._sequence_logprob(prompt, lab) for lab in label0_variants]
        logp1s = [self._sequence_logprob(prompt, lab) for lab in label1_variants]
        logp0 = np.logaddexp.reduce(logp0s)
        logp1 = np.logaddexp.reduce(logp1s)
        return softmax_two(float(logp0), float(logp1))


class OpenAIConstrainedClassifier:
    """OpenAI chat-completions classifier using output-token logprobs when available."""

    def __init__(self, model_id: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.model_id = model_id
        self.client = OpenAI()
        self.last_prompt_tokens = None

    def predict_proba(self, prompt_or_messages) -> tuple[float, float]:
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = prompt_or_messages
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=10,
        )
        try:
            self.last_prompt_tokens = resp.usage.prompt_tokens
        except Exception:
            self.last_prompt_tokens = None
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        logp0 = logp1 = None
        try:
            top = choice.logprobs.content[0].top_logprobs
            for item in top:
                tok = item.token.strip()
                if tok == "0":
                    logp0 = item.logprob if logp0 is None else np.logaddexp(logp0, item.logprob)
                elif tok == "1":
                    logp1 = item.logprob if logp1 is None else np.logaddexp(logp1, item.logprob)
        except Exception:
            pass
        if logp0 is None or logp1 is None:
            # Conservative fallback: deterministic returned token becomes high-confidence probability.
            if content == "1":
                return 0.01, 0.99
            if content == "0":
                return 0.99, 0.01
            return 0.5, 0.5
        return softmax_two(float(logp0), float(logp1))


def make_llm_backend(backend: str, model_id: str | None = None):
    if backend == "prompt_only":
        return PromptOnlyClassifier()
    if backend == "hf":
        if not model_id:
            raise ValueError("--model-id is required for backend=hf")
        return HuggingFaceConstrainedClassifier(model_id=model_id)
    if backend == "openai":
        return OpenAIConstrainedClassifier(model_id=model_id or "gpt-4o-mini")
    raise ValueError("backend must be one of: prompt_only, hf, openai")
