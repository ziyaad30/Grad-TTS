import re
import string

from text import cleaners
from text import cmudict
from text.cleaners import check_ellipse, check_stops
from text.symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
_curly_re = re.compile(r'(.*?)\{(.+?)}(.*)')

# cmudict = cmudict.CMUDict('text/en_dictionary')


"""def get_arpabet(word, dictionary):
    word_arpabet = dictionary.lookup(word)
    if word_arpabet is not None:
        return "{" + word_arpabet[0] + "}"
    else:
        return word"""


"""def text_to_sequence_old(text, cleaner_names=["english_cleaners"], dictionary=cmudict):
    sequence = []
    space = _symbols_to_sequence(' ')
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            clean_text = _clean_text(text, cleaner_names)
            if dictionary is not None:
                clean_text = [get_arpabet(w, dictionary) for w in clean_text.split(" ")]
                for i in range(len(clean_text)):
                    t = clean_text[i]
                    if t.startswith("{"):
                        sequence += _arpabet_to_sequence(t[1:-1])
                    else:
                        sequence += _symbols_to_sequence(t)
                    sequence += space
            else:
                sequence += _symbols_to_sequence(clean_text)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # remove trailing space
    if dictionary is not None:
        sequence = sequence[:-1] if sequence[-1] == space[0] else sequence

    # print(sequence_to_text(sequence))
    return sequence"""


cmu_dict = {}


def clean_text(text):
    text = text.replace(':', ' : ')
    text = text.replace(';', ' ;')
    text = text.replace('"', '')
    text = text.replace('!', ' !')
    text = text.replace('?', ' ?')
    text = text.replace(',', ' ,')
    text = text.replace('-', ' ')
    text = check_ellipse(text)
    text = check_stops(text)
    return text


with open('text/en_dictionary') as f:
    for entry in f:
        tokens = []
        for t in entry.split():
            tokens.append(t)
        cmu_dict[tokens[0]] = tokens[1:]


def text_to_sequence(text, stop_on_word_error=True):
    phoneme = []
    sequence = []
    text = _clean_text(text, ["english_cleaners"])
    text = text.upper()
    text = text.split(' ')

    for phn in text:
        found = False
        if phn.startswith("{"):
            phn = phn.strip().replace('{', '').replace('}', '') + ' '
            phoneme.append(phn)
            continue
        for word, pronunciation in cmu_dict.items():
            if word == phn:
                found = True
                arpa = ''.join(pronunciation) + ' '
                phoneme.append(arpa)
                break

        if not found:
            if phn not in string.punctuation:
                if stop_on_word_error:
                    raise Exception(f'"{phn}" NOT FOUND IN DICTIONARY!')
                print(f'THE WORD "{phn}" WILL BE USED WITHOUT ARPABET PHONEME.')
                phn = str(phn).replace(' ', '')
                phoneme.append(phn + ' ')
            else:
                phoneme.append(phn)

    text = (''.join(phoneme)
            .replace(' ,', ', ')
            .replace(' .', '. ')
            .replace(' !', '!')
            .replace(' ?', '? ')
            .replace(' ;', '; ')
            .replace(' :', ': ')
            .replace(' -', ' - ')
            .strip())

    sequence += _symbols_to_sequence(text)

    # print(sequence_to_text(sequence))

    return sequence


def sequence_to_text(sequence):
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != '_' and s != '~'
