import re
import os
from unidecode import unidecode
from .numbers import normalize_numbers
from dp.phonemizer import Phonemizer


# Load phonemizer model
phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_ipa_forward.pt')

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


def do_ipa(text):
    text = text.split()
    new_text = []
    for t in text:
        if t.startswith('*'):
            t = t.replace("*", "")
            new_text.append(t)
        else:    
            t = phoneme_text(t)
            new_text.append(t)
    
    return ' '.join(new_text)


def phoneme_text(text, lang='en_us'):
    text = phonemizer(text, lang)
    return text.strip()


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = text.strip()
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    text = phoneme_text(text)
    text = collapse_whitespace(text)
    return text
