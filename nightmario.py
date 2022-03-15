import cv2
import dataclasses
import numpy
import random
import re

from dataclasses import dataclass
from os import path
from typing import Dict, List, Optional, Set, Tuple

@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def __mul__(self, other): return Position(self.x*other, self.y*other)
    __rmul__ = __mul__

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)
    __radd__ = __add__

@dataclass(frozen=True)
class ImageLayer:
    pos: Position
    learn: bool
    images: List[str]

    def parameter_count(self):
        return len(self.images) if self.learn else 0

    def onehot_ranges(self, offset):
        return [slice_size(offset, len(self.images))] if self.learn else []

@dataclass(frozen=True)
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
                mask = numpy.ones((image.shape[0], image.shape[1], 1), dtype=numpy.uint8)
            self._images[filename] = (result := Image(image, mask))
            return result

@dataclass(frozen=True)
class Cell:
    color: int
    shape: int

    BLUE = 0
    RED = 1
    YELLOW = 2
    NUM_COLORS = 3

    EMPTY = 0
    VIRUS = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5
    DISCONNECTED = 6
    CLEARING = 7
    NUM_SHAPES = 8

    _COLOR_CHARS = "bry"
    _SHAPE_CHARS = " xlr^v*o"

    def __post_init__(self):
        if self.color not in range(Cell.NUM_COLORS):
            print(f'WARNING: Creating cell with nonstandard color {self.color}.', file=sys.stderr)
        if self.shape not in range(Cell.NUM_SHAPES):
            print(f'WARNING: Creating cell with nonstandard shape {self.shape}.', file=sys.stderr)
        # For equality testing to work right, we want to make sure empty cells
        # have a consistent "color". But since we set frozen=True in the
        # dataclass declaration, we can't just write self.color = 0.
        # See also: https://stackoverflow.com/questions/53756788/
        if self.shape == Cell.EMPTY: object.__setattr__(self, 'color', 0)

    def onehot_ranges(self, offset):
        ranges = [slice_size(offset + Cell.NUM_COLORS, Cell.NUM_SHAPES)]
        if self.shape != Cell.EMPTY:
            ranges.append(slice_size(offset, Cell.NUM_COLORS))
        return ranges

    def template_name(self, frame_parity):
        if self.shape == Cell.EMPTY: return 'k .png'
        extra = frame_parity+1 if self.shape == Cell.VIRUS else ''
        return f'{Cell._COLOR_CHARS[self.color]}{Cell._SHAPE_CHARS[self.shape]}{extra}.png'

@dataclass(frozen=True)
class Playfield:
    pos: Position
    w: int
    h: int

    _PARAMETERS_PER_POSITION = Cell.NUM_SHAPES + Cell.NUM_COLORS

    def parameter_count(self):
        return self.w*self.h*Playfield._PARAMETERS_PER_POSITION

    def render(self):
        empty = Cell(Cell.BLUE, Cell.EMPTY)
        cells = {}

        max_height = random.randrange(self.h//4) + self.h*5//8
        virus_count = random.randrange(self.w*max_height*7//8)
        positions_remaining = self.w*max_height
        for x in range(self.w):
            for y in range(self.h-max_height, self.h):
                n = random.randrange(positions_remaining)
                if n < virus_count:
                    colors = list(range(Cell.NUM_COLORS))
                    try: colors.remove(cells[Position(x-2,y)].color)
                    except KeyError: pass # position was out of bounds or the cell was empty
                    try: colors.remove(cells[Position(x,y-2)].color)
                    except KeyError: pass
                    except ValueError: pass # tried to remove the same color twice
                    cells[Position(x,y)] = Cell(random.choice(colors), Cell.VIRUS)
                    virus_count -= 1
                positions_remaining -= 1

        return cells

    def parameter_offset(self, pos):
        return Playfield._PARAMETERS_PER_POSITION * (pos.y*self.w + pos.x)

@dataclass(frozen=True)
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

        EMPTY_CELL = Cell(Cell.BLUE, Cell.EMPTY)
        frame_parity = random.randrange(2)
        for offset, playfield in self.playfields:
            cells = playfield.render()
            for y in range(playfield.h):
                for x in range(playfield.w):
                    pos = Position(x,y)

                    cell = cells.get(pos, EMPTY_CELL)
                    pos_offset = offset + playfield.parameter_offset(pos)

                    slices += cell.onehot_ranges(pos_offset)
                    parameters[pos_offset + cell.color] = 1
                    parameters[pos_offset + Cell.NUM_COLORS + cell.shape] = 1
                    cache.load(cell.template_name(frame_parity)).at(image, playfield.pos + 8*pos)

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

@dataclass(frozen=True)
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

def filter_artifacts(image):
    failures = []
    if random.randrange(2): failures += [cv2.IMWRITE_JPEG_QUALITY, 10 + random.randrange(60)]
    if random.randrange(2): failures += [cv2.IMWRITE_JPEG_LUMA_QUALITY, 10 + random.randrange(60)]
    if random.randrange(2): failures += [cv2.IMWRITE_JPEG_CHROMA_QUALITY, 10 + random.randrange(60)]
    success, encoded = cv2.imencode('.jpg', image, failures)
    assert(success)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    assert(any(x > 1 for x in image.shape))
    return image

with open('layouts/layered.txt') as f:
    _, _, t = parse_scene_tree(f)

while cv2.waitKey(1000) != 113:
    cache = TemplateCache()
    for scene in t.flatten():
        example = scene.render(cache)
        name = ' '.join(scene.name)
        cv2.imshow(' '.join(scene.name), filter_artifacts(example.image))
