import timeit
import string
import sys

PRINTABLE = set(r'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#%&\()*+,-./:;<=>?@[]^_{|}~') # There are some exceptional characters
LETTERS = string.ascii_lowercase+string.ascii_uppercase
DIGITS = string.digits
LETTERS_CONFIG = lambda allowed='': f'-c tessedit_char_blacklist=\"{"".join(sorted(set(DIGITS)-set(allowed)))}"'
UPPER_CONFIG = lambda allowed='': f'-c tessedit_char_blacklist=\"{"".join(sorted(set(PRINTABLE)-set(string.ascii_uppercase)))}\"'
LOWER_CONFIG = lambda allowed='': f'-c tessedit_char_blacklist=\"{"".join(sorted(set(PRINTABLE)-set(string.ascii_lowercase)))}\"'
DIGITS_CONFIG = lambda allowed='': f'-c tessedit_char_blacklist=\"{"".join(sorted(PRINTABLE-set(DIGITS)-set(allowed)))}\"'
LETTERS_DIGITS_CONFIG = lambda allowed='': f'-c tessedit_char_blacklist=\"{"".join(sorted(PRINTABLE-(set(list(DIGITS)+list(LETTERS)+list(allowed)))))}\"'
ONLY_CONFIG = lambda only='': f'-c tessedit_char_blacklist=\"{"".join(sorted(PRINTABLE-(set(only))))}\"'


def measure(func, *args, **kwargs):
    '''
    To measure a function runtime
    '''
    start = timeit.default_timer()
    result = func(*args, **kwargs)
    stop = timeit.default_timer()
    eplassed = stop-start
    print('Eplassed: ', eplassed)
    return result


def sort_global_memory():
    sizes_names = {sys.getsizeof(value): key for key, value in globals().items()}
    return {key: sizes_names[key] for key in sorted(sizes_names)[::-1]}