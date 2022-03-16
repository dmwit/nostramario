import cv2
import dataclasses
import itertools
import numpy
import random
import re
import sys

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

Position.LEFT = Position(-1, 0)
Position.RIGHT = Position(1, 0)
Position.UP = Position(0, -1)
Position.DOWN = Position(0, 1)

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

    BLUE, RED, YELLOW, NUM_COLORS = range(4)
    _COLOR_CHARS = "bry"

    EMPTY, VIRUS, LEFT, RIGHT, UP, DOWN, DISCONNECTED, CLEARING, NUM_SHAPES = range(9)
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

EMPTY_CELL = Cell(Cell.BLUE, Cell.EMPTY)

# Information about the bottle, you know, the 8x16 grid of viruses and pills
# and stuff.
@dataclass(frozen=True)
class Playfield:
    pos: Position
    w: int
    h: int

    _PARAMETERS_PER_POSITION = Cell.NUM_SHAPES + Cell.NUM_COLORS

    def parameter_count(self):
        return self.w*self.h*Playfield._PARAMETERS_PER_POSITION

    def render(self):
        cells = {}

        # add some viruses
        max_height = random.randrange(self.h//4) + self.h*5//8
        virus_count = (original_virus_count := 1 + random.randrange(self.w*max_height*7//8))
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
                    # since there are three colors to start with, and we only
                    # called remove twice, there's guaranteed to be something
                    # left in colors
                    cells[Position(x,y)] = Cell(random.choice(colors), Cell.VIRUS)
                    virus_count -= 1
                positions_remaining -= 1

        # clear some viruses... but not all of them
        clear_columns = set(range(self.w)).difference(pos.x for pos in cells)
        if clear_columns:
            def key(pos): return (-abs(self.h - max_height//3 - pos.y), abs(self.w//2 - pos.x))
        else:
            def key(pos): return (pos.y, abs(self.w//2 - pos.x))
        remaining_viruses = sorted(cells, key=key)
        try:
            for _ in range(random.randrange(max(1, original_virus_count - 1))):
                while True: # in ideal math-land, this loop executes just once on average (!)
                    i = int(random.triangular(0, len(remaining_viruses), 0))
                    if i < len(remaining_viruses): break
                del cells[remaining_viruses[i]]
                del remaining_viruses[i]
        except ValueError as e:
            if e.args != ('empty range for randrange()',): raise e

        # put some junk on the bottom row if we're near the end of a level
        clear_columns = set(range(self.w)).difference(pos.x for pos in cells)
        if clear_columns:
            bottoms = {x: self.h-1 for x in range(self.w) if Position(x, self.h-1) not in cells}
            for _ in range(int(random.gammavariate(3, 2))):
                if not bottoms: break
                x, y = random.choice(list(bottoms.items()))
                if colors := non_clearing_colors(cells, pos := Position(x,y)):
                    cells[pos] = Cell(random.choice(colors), Cell.DISCONNECTED)
                    if y > 0 and (pos := Position(x, y-1)) not in cells:
                        bottoms[x] = y-1
                    else: del bottoms[x]
                else: del bottoms[x]

        # put some pills and stuff
        # made this typo three times in a row. this is my life now
        falling = not random.randrange(5) # can stuff be in midair?
        i = int(random.gammavariate(1,3) if falling else random.gammavariate(2.5, 3.5))
        attemptys = 0
        while i>0 and attemptys<1000:
            attemptys += 1

            # choose the position and shape
            bottom_left = Position(random.randrange(self.w), random.randrange(self.h))
            kind = random.randrange(4)
            support = [bottom_left + Position.DOWN]
            occupancy = [bottom_left]
            if kind < 2: # disconnected
                shapes = [Cell.DISCONNECTED]
            elif kind < 3: # horizontal
                if bottom_left.x == self.w-1: continue
                support.append(bottom_left + Position.DOWN + Position.RIGHT)
                occupancy.append(bottom_left + Position.RIGHT)
                shapes = [Cell.LEFT, Cell.RIGHT]
            elif kind < 4: # vertical
                occupancy.append(bottom_left + Position.UP)
                shapes = [Cell.DOWN, Cell.UP]
                if bottom_left.y == 0:
                    shapes = [Cell.DISCONNECTED]
                    del occupancy[1]
            else: raise Exception(f'strange kind {kind} while generating detritus')

            # double check that the chosen position is empty and, if gravity
            # should be settled, that the chosen position is supported
            if any(pos in cells for pos in occupancy): continue
            if not falling and all(pos not in cells for pos in support): continue

            # choose the colors; bias towards colors that work towards a clear
            # TODO: think about how to be a bit less stupidly inefficient here
            colors = list(itertools.product(*([]
                + 12*self.neighboring_colors(cells, pos)
                + 3*self.neighboring_colors(cells, pos, skip=1)
                + list(range(Cell.NUM_COLORS))
                for pos in occupancy)))

            if not falling:
                # make sure we don't pick colors that should be cleared,
                # because we're not in "unsettled" mode
                new_colors = []
                for choice in colors:
                    for j, pos in enumerate(occupancy):
                        cells[pos] = Cell(choice[j], shapes[j])
                    if all(choice[j] in non_clearing_colors(cells, pos) for j, pos in enumerate(occupancy)):
                        new_colors.append(choice)
                for pos in occupancy: del cells[pos]
                colors = new_colors
            if not colors: continue
            colors = random.choice(colors)

            for j, pos in enumerate(occupancy):
                cells[pos] = Cell(colors[j], shapes[j])

            i -= 1

        return cells

    def neighboring_colors(self, cells, pos, skip=0):
        colors = []
        for direction in [Position.LEFT, Position.RIGHT, Position.DOWN, Position.UP]:
            head = pos
            for _ in range(skip+1):
                while True:
                    head += direction
                    if not self.in_bounds(head): break
                    if cells.get(head, EMPTY_CELL) != EMPTY_CELL: break

                color = cells.get(head, EMPTY_CELL).color
                count = 0
                while self.in_bounds(head) and cells.get(head, EMPTY_CELL).color == color:
                    head += direction
                    count += 1
            colors += count*[color]
        return colors

    def in_bounds(self, pos):
        return pos.x >= 0 and pos.y >= 0 and pos.x < self.w and pos.y < self.h

    def parameter_offset(self, pos):
        return Playfield._PARAMETERS_PER_POSITION * (pos.y*self.w + pos.x)

def non_clearing_colors(cells, pos):
    return [color for color in range(Cell.NUM_COLORS) if max_run(cells, pos, color) < 4]

def max_run(cells, pos, color):
    lengths = [run_length(cells, pos, color, direction) for direction in [Position.LEFT, Position.RIGHT, Position.UP, Position.DOWN]]
    return 1+max(lengths[0]+lengths[1], lengths[2]+lengths[3])

def run_length(cells, pos, color, direction):
    n = 0
    while True:
        pos += direction
        cell = cells.get(pos, EMPTY_CELL)
        if cell.color != color or cell.shape == Cell.EMPTY: break
        n += 1
    return n

# A scene is a collection of instructions for creating a random training
# example. Instructions are nested when in a SceneTree, but in Scene all the
# tree traversal is already done, and instructions from enclosing parts of the
# tree are already included.
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

# the raw data, as close to the directly parsed form as possible
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
    # each slice is an independent classification problem, so should be part of
    # its own cross-correlation calculation
    onehot_ranges: List[slice]
    # the scene identifiers aren't necessarily contiguous, so there's one final
    # classification problem in an arbitrary collection of indices
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

cache = TemplateCache()
while cv2.waitKey(1000) != 113:
    for scene in t.flatten():
        example = scene.render(cache)
        name = ' '.join(scene.name)
        cv2.imshow(' '.join(scene.name), filter_artifacts(example.image))
