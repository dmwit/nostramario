import cv2
import dataclasses
import numpy
import random
import re

from dataclasses import dataclass
from os import path
from typing import Dict, List, Optional, Set, Tuple

@dataclass
class Position:
    x: int
    y: int

@dataclass
class ImageLayer:
    pos: Position
    learn: bool
    images: List[str]

    def parameter_count(self):
        return len(self.images) if self.learn else 0

    def onehot_ranges(self, offset):
        return [slice_size(offset, len(self.images))] if self.learn else []

@dataclass
class Image:
    image: numpy.ndarray
    mask: numpy.ndarray

    def at(self, target, pos):
        h, w, _ = self.image.shape
        ys = slice_size(pos.y, h)
        xs = slice_size(pos.x, w)
        target[ys, xs, :] = self.mask * self.image + (1-self.mask) * target[ys, xs, :]

TEMPLATE_DIR="templates"
@dataclass
class TemplateCache:
    _images: Dict[str, Image] = dataclasses.field(default_factory=dict)

    def load(self, filename):
        try: return self._images[filename]
        except KeyError:
            image = cv2.imread(path.join(TEMPLATE_DIR, filename), cv2.IMREAD_UNCHANGED)
            if image.shape[2] == 4:
                mask = numpy.minimum(1, image[:, :, [3]])
                image = image[:, :, 0:3]*mask
            else:
                mask = numpy.ones((img.shape[0], img.shape[1], 1), dtype=numpy.uint8)
            self._images[filename] = (result := Image(image, mask))
            return result

@dataclass
class Playfield:
    pos: Position
    w: int
    h: int

    _NUM_SHAPES = 8 # virus, left pill half, right pill half, upper pill half, lower pill half, disconnected pill half, clearing, empty
    _NUM_COLORS = 3 # blue, red, yellow
    _PARAMETERS_PER_POSITION = 8+3 # Playfield._NUM_SHAPES + Playfield._NUM_COLORS

    def parameter_count(self):
        return self.w*self.h*Playfield._PARAMETERS_PER_POSITION

    def onehot_ranges(self, offset):
        return ([]
            + [slice_size(offset + Playfield._PARAMETERS_PER_POSITION*i, Playfield._NUM_SHAPES) for i in range(w*h)]
            + [slice_size(offset + Playfield._PARAMETERS_PER_POSITION*i + Playfield._NUM_SHAPES, Playfield._NUM_COLORS) for i in range(w*h)]
            )

@dataclass
class Scene:
    name: List[str]
    index: int
    background: str
    layers: List[Tuple[int, ImageLayer]]
    playfields: List[Tuple[int, Playfield]]
    parameter_count: int
    alternative_indices: Set[int]

    def render(self, cache):
        image = numpy.array(cache.load(self.background).image)
        parameters = numpy.zeros(self.parameter_count)
        parameters[self.index] = 1
        slices = []

        for offset, layer in self.layers:
            num_images = len(layer.images)
            slices += layer.onehot_ranges(offset)

            index = random.randrange(num_images)
            if layer.learn: parameters[index+offset] = 1
            cache.load(layer.images[index]).at(image, layer.pos)

        return TrainingExample(image, parameters, slices, self.alternative_indices)

@dataclass
class SceneTree:
    background: Optional[str]
    children: List[Tuple[str, 'SceneTree']]
    layers: List[ImageLayer]
    playfields: List[Playfield]
    _scene_parameters: Optional[Set[int]] = None
    _parameter_count: Optional[int] = None

    def flatten(self):
        # [x] is a bit like a mutable version of x
        yield from self._flatten(None, [], [], [], [0], self.parameter_count(), self.scene_parameters())

    def _flatten(self, bg, nm, layers, playfields, index, parameter_count, scene_parameters):
        if bg is None: bg = self.background
        elif self.background is not None:
            raise Exception(f"Ambiguous background in {nm}; it was specified here as {self.background} and in an ancestor as {bg}.")

        for layer in self.layers:
            layers.append((index[0], layer))
            index[0] += layer.parameter_count()
        for playfield in self.playfields:
            playfields.append((index[0], playfield))
            index[0] += playfield.parameter_count()

        for child_nm, child_t in self.children:
            nm.append(child_nm)
            yield from child_t._flatten(bg, nm, layers, playfields, index, parameter_count, scene_parameters)
            nm.pop()

        if not self.children:
            if bg is None: raise Exception(f"No background specified for {nm}.")
            yield Scene(list(nm), index[0], bg, list(layers), list(playfields), parameter_count, scene_parameters)
            index[0] += 1

        # -0 is 0, so we need a conditional
        if self.layers: del layers[-len(self.layers):]
        if self.playfields: del playfields[-len(self.playfields):]

    def parameter_count(self):
        if self._parameter_count is None:
            self._parameter_count = (0
                + (0 if self.children else 1)
                + sum(layer.parameter_count() for layer in self.layers)
                + sum(playfield.parameter_count() for playfield in self.playfields)
                + sum(child.parameter_count() for _, child in self.children)
                )
        return self._parameter_count

    def scene_parameters(self):
        if self._scene_parameters is None:
            self._scene_parameters = set()
            self._initialize_scene_parameters(self._scene_parameters, 0)
        return self._scene_parameters

    def _initialize_scene_parameters(self, indices, index):
        index += sum(layer.parameter_count() for layer in self.layers)
        index += sum(playfield.parameter_count() for playfield in self.playfields)

        for _, child in self.children:
            index = child._initialize_scene_parameters(indices, index)

        if not self.children:
            indices.add(index)
            index += 1

        return index

@dataclass
class TrainingExample:
    image: numpy.ndarray
    classification: numpy.ndarray
    onehot_ranges: List[slice]
    onehot_indices: List[int]

def slice_size(start, length): return slice(start, start+length)

LOCATED_REGEX = re.compile('([1-9][0-9]*)[ \t]*,[ \t]*([1-9][0-9]*)[ \t]*(.+)')
PLAYFIELD_REGEX = re.compile('([1-9][0-9]*)[ \t]*x[ \t]*([1-9][0-9]*)[ \t]*playfield')
def parse_scene_tree(line_source, parent_indent=None):
    t = SceneTree(None, [], [], [])
    self_indent = None

    for line in line_source:
        here_indent, content = split_indentation(line)
        if content == '' or content[0] == '#': continue

        if self_indent is None:
            if parent_indent is None or (here_indent.startswith(parent_indent) and here_indent != parent_indent):
                self_indent = here_indent
            elif parent_indent.startswith(here_indent):
                return (here_indent, content, t)
            else:
                raise Exception(f"Indentation error. Nesting level <{here_indent}> is incomparable with the enclosing indentation of <{parent_indent}>.")

        while content != '':
            if here_indent != self_indent:
                if parent_indent.startswith(here_indent):
                    return (here_indent, content, t)
                else:
                    raise Exception(f"Indentation error. Expected <{self_indent}> or a prefix of <{parent_indent}>, but saw <{here_indent}>.")

            if content.endswith(':'):
                name = content[:-1]
                here_indent, content, child = parse_scene_tree(line_source, self_indent)
                t.children.append((name, child))
            elif match := LOCATED_REGEX.fullmatch(content):
                pos = Position(int(match.group(1)), int(match.group(2)))
                thing = match.group(3)
                if match := PLAYFIELD_REGEX.fullmatch(thing):
                    t.playfields.append(Playfield(pos, int(match.group(1)), int(match.group(2))))
                else:
                    learn = thing[0] != '!'
                    if not learn: thing = thing[1:]
                    t.layers.append(ImageLayer(pos, learn, [filename.strip() for filename in thing.split(';')]))
                break
            elif t.background == None:
                t.background = content
                break
            else:
                raise Exception(f"Attempted to set background to {content}, but there is already a background of {t.background}.")

    return ('', '', t)

SPACE_REGEX = re.compile('([ \t]*)(.*[^ \t\r\n])?[ \t\r\n]*')
def split_indentation(line):
    match = SPACE_REGEX.fullmatch(line)
    return (match.group(1), match.group(2) or '')

with open('layouts/layered.txt') as f:
    _, _, t = parse_scene_tree(f)

print(t.parameter_count(), t.scene_parameters())
for scene in t.flatten(): print(scene)

while cv2.waitKey(100000) != 113:
    cache = TemplateCache()
    for scene in t.flatten():
        example = scene.render(cache)
        name = ' '.join(scene.name)
        print(name)
        print(numpy.nonzero(example.classification))
        print(example.onehot_ranges)
        print(example.onehot_indices)
        cv2.imshow(' '.join(scene.name), example.image)
