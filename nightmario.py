import cv2
import dataclasses
import numpy
import random
import re

from dataclasses import dataclass
from os import path, walk

def load_templates(dirname='templates'):
    _, _, filenames = walk(dirname).__next__()
    templates = {}
    for filename in filenames:
        img = cv2.imread(path.join(dirname, filename), cv2.IMREAD_UNCHANGED)
        if img.shape[0:2] == (8, 8):
            if img.shape[2] == 4:
                mask = numpy.minimum(1, img[:,:,[3]])
                img = img[:,:,0:3]
            else:
                mask = numpy.ones((img.shape[0], img.shape[1], 1), dtype=numpy.uint8)
            templates[path.splitext(filename)[0]] = (img, mask)
    return templates

@dataclass
class GridOccupants:
    special: bool
    names: list[str]
    imgs: list[tuple[numpy.ndarray, numpy.ndarray]] = dataclasses.field(default_factory=list)

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

def populate_key(templates, key):
    for _, occ in key.items():
        if occ.special: continue
        occ.imgs = []
        for name in occ.names:
            try: occ.imgs.append(templates[name])
            except KeyError: pass

@dataclass
class Layout:
    # TODO: make these private and keep them in synch
    screen: list[list[GridOccupants]] = dataclasses.field(default_factory=list)
    lookahead: list[tuple[float, float]] = dataclasses.field(default_factory=list)
    w: int = 0
    h: int = 0

    def sanity_check(self):
        assert(len(self.screen) == self.h)
        for row in self.screen:
            assert(len(row) == self.w)
        for x, y in self.lookahead:
            assert(int(8*x) == 8*x)
            assert(int(8*y) == 8*y)

def parse_layout(filename, key):
    NUMBER = '([0-9]+(.[0-9]+)?)'
    layout = Layout()
    with open(filename) as f:
        for line in f:
            if line.startswith('#'): continue
            elif m := re.fullmatch(f'lookahead at {NUMBER}-{NUMBER}, {NUMBER}\n', line):
                x0 = float(m.group(1))
                x1 = float(m.group(3))
                y  = float(m.group(5))
                assert(x0 + 1 == x1)
                layout.lookahead.append((x0, y))
            elif line.strip():
                layout.screen.append([key[g] for g in line[:-1]])
                layout.w = len(line)-1
            else: pass
    layout.h = len(layout.screen)
    layout.sanity_check()
    return layout

LOW, MED, HI = range(3)
SPEEDS = {
    LOW: (0x00, 0x51, 0x00),
    MED: (0x9e, 0x00, 0x46),
    HI:  (0x75, 0x75, 0x75),
    }

def interval_slice(i): return slice(8*i, 8*(i+1))
def img_slice(x, y): return (interval_slice(y), interval_slice(x), slice(None))

def hallucinate(layout):
    img = numpy.zeros((layout.h*8, layout.w*8, 3), dtype=numpy.uint8)

    # draw grid
    speed = random.choice(list(SPEEDS))
    grid_color = numpy.array([[SPEEDS[speed]]])
    for x in range(0, layout.w, 2):
        for y in range(0, layout.h, 2):
            img[img_slice(x+1, y)] = grid_color
            img[img_slice(x, y+1)] = grid_color

    for x in range(0, layout.w):
        for y in range(0, layout.h):
            ix = img_slice(x, y)
            occ = layout.screen[y][x]
            if occ.imgs and not occ.special:
                sprite, mask = random.choice(occ.imgs)
                img[ix] = sprite*mask + img[ix]*(1-mask)

    return img

templates = load_templates()
key = parse_layout_key('layouts/key.txt')
warn_about_mismatches(templates, key)
populate_key(templates, key)
layout = parse_layout('layouts/1p.txt', key)

while cv2.waitKey(1000) != 113:
    cv2.imshow('hallucination', hallucinate(layout))
