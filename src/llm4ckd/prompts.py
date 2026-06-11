from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .features import serialize_record


def task_block(config: Dict) -> str:
    p = config["prompt"]
    return (
        f"Task: {p['system_instruction']}\n"
        f"Input: {p['input_description']}\n"
        f"Instruction: {p['output_instruction']}"
    )


def format_labeled_example(
    row: pd.Series,
    y: int,
    features: Sequence[str],
    serialization: str,
    index: int | None = None,
) -> str:
    prefix = f"/** Example #{index} **/\n" if index is not None else ""
    record = serialize_record(row, features, serialization)
    return f"{prefix}{record}\nOutput: {int(y)}"


def build_instruction_prompt(
    query_row: pd.Series,
    features: Sequence[str],
    serialization: str,
    config: Dict,
    examples: Sequence[Tuple[pd.Series, int]] | None = None,
) -> str:
    blocks: List[str] = [task_block(config)]
    if examples:
        for i, (ex_row, ex_y) in enumerate(examples, start=1):
            blocks.append(format_labeled_example(ex_row, ex_y, features, serialization, index=i))
    query = serialize_record(query_row, features, serialization)
    blocks.append(f"Now classify the following patient.\n{query}\nOutput:")
    return "\n\n".join(blocks)


def build_chat_messages(
    query_row: pd.Series,
    features: Sequence[str],
    serialization: str,
    config: Dict,
    examples: Sequence[Tuple[pd.Series, int]] | None = None,
) -> List[Dict[str, str]]:
    system = task_block(config)
    user_blocks: List[str] = []
    if examples:
        for i, (ex_row, ex_y) in enumerate(examples, start=1):
            user_blocks.append(format_labeled_example(ex_row, ex_y, features, serialization, index=i))
    user_blocks.append("Now classify the following patient.")
    user_blocks.append(serialize_record(query_row, features, serialization))
    user_blocks.append("Output:")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_blocks)},
    ]


def messages_to_llama_chat_text(messages: Sequence[Dict[str, str]]) -> str:
    """Render messages in a Llama-style chat template for tokenizer-free auditing.

    When using Hugging Face models, prefer the tokenizer's native `apply_chat_template`.
    """
    out = "<|begin_of_text|>\n"
    for msg in messages:
        out += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}\n<|eot_id|>\n"
    out += "<|start_header_id|>assistant<|end_header_id|>\n"
    return out


def build_prompt(
    query_row: pd.Series,
    features: Sequence[str],
    serialization: str,
    prompt_style: str,
    config: Dict,
    examples: Sequence[Tuple[pd.Series, int]] | None = None,
):
    if prompt_style == "instruction":
        return build_instruction_prompt(query_row, features, serialization, config, examples)
    if prompt_style == "chat":
        return build_chat_messages(query_row, features, serialization, config, examples)
    raise ValueError("prompt_style must be 'instruction' or 'chat'")
