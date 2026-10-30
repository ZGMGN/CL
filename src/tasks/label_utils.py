"""O-LoRA style instructions and normalized exact-match labels."""
import string


def normalize_answer(text):
    return ' '.join(''.join(c for c in text.lower() if c not in string.punctuation).split())


def extract_answer_label(text, labels):
    normalized = normalize_answer(text)
    return next((label for label in labels if normalize_answer(label) == normalized), None)


def build_instruction(example, labels, task_type, fields):
    if task_type == 'SC':
        instruction = 'What is the sentiment of the following paragraph? Choose one from the option.\n'
    elif task_type == 'QQP':
        instruction = 'Whether the "first sentence" and the "second sentence" have the same meaning? Choose one from the option.\n'
    else:
        instruction = 'What is the topic of the following paragraph? Choose one from the option.\n'
    sentence = ''.join(f'{name}: {(example[field] or "").strip()}\n' for field, name in fields)
    return instruction + 'Option: ' + ', '.join(labels) + ' \n' + sentence + '\nAnswer:'
