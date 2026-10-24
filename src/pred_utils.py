"""Prompt formatting, output cleanup, and end-of-sequence tokens."""
import re

def build_chat(tokenizer, prompt, model_name):
    if "mistral" in model_name.lower():
        prompt = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    elif "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except ValueError:
            prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful assistant<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{messages[1]['content']}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
    else:
        raise ValueError(f"Unsupported chat model: {model_name}")
    return prompt

def post_process(response, model_name):
    if "llama-3" in model_name.lower():
        # Remove repeated prompt scaffolding from compressed-cache continuations.
        for marker in (
            "<|eot_id|>",
            "<|end_of_text|>",
            "</s>",
            ".assistant",
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>system<|end_header_id|>",
        ):
            response = response.split(marker)[0]
        if response.lstrip().lower().startswith("question:"):
            answer_parts = re.split(r"\bAnswer:\s*", response, maxsplit=1, flags=re.I)
            response = answer_parts[1] if len(answer_parts) > 1 else ""
        response = re.split(r"\n+\s*Question\b", response, maxsplit=1, flags=re.I)[0]
        answer_parts = re.split(r"(?:^|\n+)\s*Answer:\s*", response, maxsplit=1, flags=re.I)
        if len(answer_parts) > 1:
            response = answer_parts[1]
        stripped = response.lstrip()
        if stripped.lower().startswith("answer:"):
            response = stripped.split(":", 1)[1]
        response = response.strip()
        response = re.sub(
            r"^(?:[.\s]*)?(?:You are a helpful assistant\.?\s*)+",
            "",
            response,
            flags=re.I,
        ).strip()
        prompt_echo_markers = (
            "you are given",
            "read the following",
            "answer the question based",
            "the following are given",
            "here are",
            "please determine",
            "summarize the dialogue",
            "write a one-page summary",
        )
        lower_response = response.lower()
        first_echo = min(
            (lower_response.find(marker) for marker in prompt_echo_markers
             if lower_response.find(marker) >= 0),
            default=-1,
        )
        if first_echo >= 0:
            prefix = response[:first_echo].strip(" \t\r\n.:;-")
            response = "" if first_echo < 80 and len(prefix.split()) <= 2 else prefix
        if response.lower().startswith(prompt_echo_markers):
            response = ""
    return response

def post_process_code_completion(pred: str) -> str:

    text = pred.replace("\r\n", "\n").replace("\r", "\n")
    if "```" in text:
        text = text.replace("```python", "").replace("```java", "").replace("```", "")
    for marker in (
        "The final answer",
        "Please complete the code",
        "Here is the completed code",
        "This code",
    ):
        if marker in text:
            text = text.split(marker)[0]
    for line in text.split("\n"):
        cleaned = line.rstrip()
        if not cleaned.strip():
            continue
        if cleaned.strip().startswith(("#", "//", "/*", "*", "\"\"\"", "'''", "```")):
            continue
        return cleaned
    return ""

def collect_eos_token_ids(tokenizer, model=None):

    eos_token_ids = set()
    if isinstance(tokenizer.eos_token_id, list):
        eos_token_ids.update(tokenizer.eos_token_id)
    elif tokenizer.eos_token_id is not None:
        eos_token_ids.add(tokenizer.eos_token_id)
    if model is not None and hasattr(model, 'generation_config') and model.generation_config.eos_token_id is not None:
        gen_eos = model.generation_config.eos_token_id
        if isinstance(gen_eos, list):
            eos_token_ids.update(gen_eos)
        else:
            eos_token_ids.add(gen_eos)
    return eos_token_ids

