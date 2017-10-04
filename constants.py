import json
from pprint import pprint

import namedtupled

CONFIGURATION_FILENAME = 'conf.json'


def filename_to_named_tuple(filename):
    with open(filename) as data_file:
        c_ = json.load(data_file)
        pprint(c_)
        return namedtupled.map(c_)


def load_constants():
    return filename_to_named_tuple(CONFIGURATION_FILENAME)


c = load_constants()
