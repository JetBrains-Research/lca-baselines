from src.utils.tokenization_utils import TokenizationUtils


def get_context_metrics(messages: list[dict]):
    tokenization_utils = TokenizationUtils('gpt-4o')
    return {
        'system_prompt_tokens': tokenization_utils.count_text_tokens(messages[0]['content']),
        'user_prompt_tokens': tokenization_utils.count_text_tokens(messages[1]['content']),
        'total_tokens': tokenization_utils.count_messages_tokens(messages),
    }