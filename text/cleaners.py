import re
from unidecode import unidecode
from .numbers import normalize_numbers

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
        ('inc', 'incorporated'),
    ]
]


def expand_abbreviations(text):
    """Check and replace common abbreviations with their full words."""
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def check_stops(text):
    for test in text.split(" "):
        ellipses = re.search(r'(\w+)\.{1,}', test)
        if ellipses:
            word = ellipses.group(0)
            elip = word[0:-1]
            text = text.replace(word, elip + " .")
    return text


def check_ellipse(text):
    for test in text.split(" "):
        ellipses = re.search(r'(\w+)\.{3,}', test)
        if ellipses:
            word = ellipses.group(0)
            elip = word[0:-3]
            text = text.replace(word, elip + " ...")
    return text


def check_arpa_stress(text):
    """Pipeline for arpabet text, including number and abbreviation expansion but checks for arpabet stress."""
    text = text.split()
    new_text = []
    for t in text:
        if t.endswith(')') or t.endswith('),') or t.endswith(')?') or t.endswith(')!') or t.endswith('):'):
            new_text.append(t)
        else:
            new_text.append(expand_numbers(t))

    return ' '.join(new_text)


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = text.strip()
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = check_arpa_stress(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    text = text.replace('--', '-')
    text = text.replace(':', ' : ')
    text = text.replace(';', ' ;')
    text = text.replace('"', '')
    text = text.replace('!', ' !')
    text = text.replace('?', ' ?')
    text = text.replace(',', ' ,')
    text = check_ellipse(text)
    text = check_stops(text)
    return text
