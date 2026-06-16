LLADA_DEFINED_TOKEN_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<role>' + message['role'] + '</role>\\n\\n' + message['content'] | trim + eos_token %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{{ '<role>assistant</role>\\n\\n' }}"
)

LORA_LITERAL_CHAT_MARKERS = (
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
)


def clean_lora_chat_markers(text):
    for marker in LORA_LITERAL_CHAT_MARKERS:
        text = text.replace(marker, "")
    return text
