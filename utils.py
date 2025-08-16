# utils.py

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def clean_text(text):
    # Optional cleanup for Project Gutenberg formatting
    start_marker = "*** START"
    end_marker = "*** END"
    start = text.find(start_marker)
    end = text.find(end_marker)
    
    if start != -1 and end != -1:
        text = text[start:end]

    # Lowercasing for consistency
    text = text.lower()

    # Optional: remove non-printable characters
    import re
    text = re.sub(r'[^a-z0-9\s\.,;:!?\'\"\-\n]', '', text)
    return text
