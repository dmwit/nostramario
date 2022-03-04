import cv2
import dataclasses
import numpy

from dataclasses import dataclass
from os import path, walk

def load_templates(dirname='templates'):
    _, _, filenames = walk(dirname).__next__()
    templates = {}
    for filename in filenames:
        img = cv2.imread(path.join(dirname, filename), cv2.IMREAD_UNCHANGED)
        if img.shape[0:2] == (8, 8):
            if img.shape[2] == 4:
                img = numpy.maximum(img[:,:,0:3], img[:,:,[3]])
            templates[path.splitext(filename)[0]] = img
    return templates

@dataclass
class GridOccupants:
    special: bool
    names: list[str]
    imgs: list[numpy.ndarray] = dataclasses.field(default_factory=list)

def parse_layout_key(filename):
    key = {}
    with open(filename) as f:
        for line in f:
            if line[2:10] == 'special:':
                key[line[0]] = GridOccupants(True, [line[10:-1].strip()])
            else:
                template_names = line[2:-1].split(',')
                # sigh
                if template_names == ['']: template_names = []
                key[line[0]] = GridOccupants(False, template_names)
    return key

def warn_about_mismatches(templates, key):
    missing_templates = set()
    missing_chars = set(templates)

    for _, occ in key.items():
        if occ.special: continue
        missing_templates.update(name for name in occ.names if name not in templates)
        missing_chars.difference_update(occ.names)

    if missing_templates:
        print(f'WARNING: layout key requested templates for {missing_templates} but they were not loaded')
    if missing_chars:
        print(f'WARNING: loaded templates for {missing_chars} but the layout key doesn\'t mention them')

templates = load_templates()
key = parse_layout_key('layouts/key.txt')
warn_about_mismatches(templates, key)
