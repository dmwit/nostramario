import cv2
import dataclasses
import itertools
import math
import numpy
import numpy.random
import random
import re
import sys
import tensorboardX as tx
import torch

from dataclasses import dataclass
from os import path, walk
from torch import nn
from typing import Dict, List, Optional, Set, Tuple

np_rng = numpy.random.default_rng()

@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def __mul__(self, other): return Position(self.x*other, self.y*other)
    __rmul__ = __mul__

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)
    __radd__ = __add__

    def __neg__(self): return Position(-self.x, -self.y)
    def __sub__(self, other): return Position(self.x - other.x, self.y - other.y)

Position.LEFT = Position(-1, 0)
Position.RIGHT = Position(1, 0)
Position.UP = Position(0, -1)
Position.DOWN = Position(0, 1)
Position.EVERY_WHICH_WAY = (Position.LEFT, Position.RIGHT, Position.UP, Position.DOWN)

@dataclass(frozen=True)
class ImageLayer:
    pos: Position
    learn: bool
    images: List[str]

    def parameter_count(self):
        return len(self.images) if self.learn else 0

    def onehot_ranges(self, offset):
        return [slice_size(offset, len(self.images))] if self.learn else []

    def select(self, offset, params):
        i = random.randrange(len(self.images))
        if self.learn: params[offset+i] = 1
        return ImageLayerSelection(self.pos, self.images[i])

    def reconstruct(self, offset, params):
        return \
            ImageLayerSelection(self.pos, reconstruct_list(params, offset, self.images)) \
            if self.learn else \
            ImageLayerSelection(self.pos, self.images[0])

@dataclass(frozen=True)
class ImageLayerSelection:
    pos: Position
    image: str

    def render(self, cache, image):
        cache.load(self.image).at(image, self.pos)

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
        if self.shape == Cell.EMPTY: return 'transparent_pixel.png'
        extra = frame_parity+1 if self.shape == Cell.VIRUS else ''
        return f'{Cell._COLOR_CHARS[self.color]}{Cell._SHAPE_CHARS[self.shape]}{extra}.png'

    def layer(self, frame_parity, pos):
        return ImageLayerSelection(pos, self.template_name(frame_parity))

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
        return self.w*(self.h+1)*Playfield._PARAMETERS_PER_POSITION

    def select(self, offset, params, frame_parity):
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
        falling = not random.randrange(5) # can stuff be in midair?
        i = int(random.gammavariate(1,3) if falling else random.gammavariate(2.5, 3.5))
        # made this typo three times in a row. this is my life now
        attemptys = 0
        while i>0 and attemptys<1000:
            attemptys += 1

            # choose the position and shape
            bottom_left = self.random_position()
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

        # if stuff isn't falling, maybe put a floating pill (that's being
        # player-controlled) or mark some clears on-screen
        if not falling:
            # 1/2/6 clears/nothing/pill
            extras = random.randrange(9)
            if extras < 1: # mark some clears
                pos = self.random_position()
                max_lengths = {
                        Position.LEFT: pos.x+1,
                        Position.RIGHT: self.w-pos.x,
                        Position.DOWN: self.h-pos.y,
                        Position.UP: pos.y+1,
                    }
                direction = random.choice([direction for direction, length in max_lengths.items() if length >= 4])
                boundary = mark_random_clear(cells, pos, direction, min(6, max_lengths[direction]))

                for i in range(int(random.triangular(0, 4.99, 0))):
                    if random.randrange(3) < 2:
                        pos = random.choice(list(boundary))
                    else: pos = self.random_position()
                    if cells.get(pos, EMPTY_CELL).shape == Cell.CLEARING: continue

                    max_lengths = {}
                    for direction in Position.EVERY_WHICH_WAY:
                        head = pos + direction
                        length = 0
                        while self.in_bounds(head) and cells.get(head, EMPTY_CELL).shape != Cell.CLEARING:
                            head += direction
                            length += 1
                        if length >= 4: max_lengths[direction] = length
                    if not max_lengths: continue

                    direction = random.choice(list(max_lengths))
                    boundary = mark_random_clear(cells, pos, direction, min(6, max_lengths[direction]))

            elif extras < 3: # don't put anything extra
                pass

            elif extras < 9: # put a floating pill
                voids = ([]
                    + [ (pos, Position.UP)
                        for x in range(self.w)
                        for y in range(self.h)
                        if (pos := Position(x,y)) not in cells and Position(x, y-1) not in cells
                      ]
                    + [ (pos, Position.RIGHT)
                        for x in range(self.w-1)
                        for y in range(self.h)
                        if (pos := Position(x,y)) not in cells and Position(x+1, y) not in cells
                      ]
                    )
                if voids:
                    pos, direction = random.choice(voids)
                    bl_shape, other_shape = [Cell.LEFT, Cell.RIGHT] if direction == Position.RIGHT else [Cell.DOWN, Cell.UP]
                    cells[pos] = Cell(random.randrange(Cell.NUM_COLORS), bl_shape)
                    cells[pos + direction] = Cell(random.randrange(Cell.NUM_COLORS), other_shape)

            else: raise Exception(f'strange kind {extras} while generating extra miscellanea')

        for pos in self.positions():
            cell = cells.get(pos, EMPTY_CELL)
            pos_offset = self.parameter_offset(pos)
            params[offset + pos_offset + cell.color] = 1
            params[offset + pos_offset + Cell.NUM_COLORS + cell.shape] = 1

        return PlayfieldSelection(self.pos, self.w, self.h, cells, frame_parity)

    def positions(self):
        for x in range(self.w):
            for y in range(-1, self.h):
                yield Position(x, y)

    def random_position(self):
        return Position(random.randrange(self.w), random.randrange(self.h))

    def neighboring_colors(self, cells, pos, skip=0):
        colors = []
        for direction in Position.EVERY_WHICH_WAY:
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
        return Playfield._PARAMETERS_PER_POSITION * ((pos.y+1)*self.w + pos.x)

    def onehot_ranges(self, offset):
        return [slice_size(offset + self.parameter_offset(pos) + i, size)
            for pos in self.positions()
            for i, size in [(0, Cell.NUM_COLORS), (Cell.NUM_COLORS, Cell.NUM_SHAPES)]
            ]

    def reconstruct(self, offset, params):
        cells = {
            pos: Cell(
                reconstruct_onehot(params, (i := offset + self.parameter_offset(pos)), Cell.NUM_COLORS),
                reconstruct_onehot(params, i + Cell.NUM_COLORS, Cell.NUM_SHAPES),
                )
            for pos in self.positions()
            }
        return PlayfieldSelection(self.pos, self.w, self.h, cells, 0)

def non_clearing_colors(cells, pos):
    return [color for color in range(Cell.NUM_COLORS) if max_run(cells, pos, color) < 4]

def max_run(cells, pos, color):
    lengths = [run_length(cells, pos, color, direction) for direction in Position.EVERY_WHICH_WAY]
    return 1+max(lengths[0]+lengths[1], lengths[2]+lengths[3])

def run_length(cells, pos, color, direction):
    n = 0
    while True:
        pos += direction
        cell = cells.get(pos, EMPTY_CELL)
        if cell.color != color or cell.shape == Cell.EMPTY: break
        n += 1
    return n

def mark_random_clear(cells, pos, direction, max_length):
    color = random.randrange(Cell.NUM_COLORS)
    length = int(random.triangular(4, max_length+0.99, 4))
    cleared = set()

    to_clear = set(pos + i*direction for i in range(length))
    head = pos - direction
    while (cell := cells.get(head, EMPTY_CELL)) != EMPTY_CELL and cell.color == color:
        to_clear.add(head)
        head -= direction
    head = pos + length*direction
    while (cell := cells.get(head, EMPTY_CELL)) != EMPTY_CELL and cell.color == color:
        to_clear.add(head)
        head += direction

    while to_clear:
        pos = to_clear.pop()
        if pos in cleared: continue
        cleared.add(pos)
        for direction in [Position.LEFT, Position.UP]:
            run1 = run_length(cells, pos, color, direction)
            run2 = run_length(cells, pos, color, -direction)
            if run1+run2+1 >= 4:
                to_clear.update(pos + i*direction for i in range(-run2, run1+1))

    boundary = set()
    for pos in cleared:
        cells[pos] = Cell(color, Cell.CLEARING)
        boundary.update(pos + direction for direction in Position.EVERY_WHICH_WAY)
    boundary -= cleared

    return boundary

@dataclass(frozen=True)
class PlayfieldSelection:
    pos: Position
    w: int
    h: int
    cells: Dict[Position, Cell]
    frame_parity: int

    def render(self, cache, image):
        for cell_pos in self.positions():
            self.cells.get(cell_pos, EMPTY_CELL) \
                .layer(self.frame_parity, self.pos + 8*cell_pos) \
                .render(cache, image)

    def positions(self):
        for x in range(self.w):
            for y in range(-1, self.h):
                yield Position(x, y)

@dataclass
class Lookahead:
    # the int is how many clockwise rotations the pill has
    rotations_and_positions: List[Tuple[int, Position]]

    def __init__(self, rot=None, pos=None):
        self.rotations_and_positions = [] if rot is None or pos is None else [(rot, pos)]

    def __add__(self, other):
        result = Lookahead()
        result.rotations_and_positions = self.rotations_and_positions + other.rotations_and_positions
        return result

    def __iadd__(self, other):
        self.rotations_and_positions += other.rotations_and_positions

    def parameter_count(self):
        return 2*Cell.NUM_COLORS

    def select(self, offset, params, frame_parity):
        colors = [random.randrange(Cell.NUM_COLORS), random.randrange(Cell.NUM_COLORS)]
        params[offset + colors[0]] = 1
        params[offset + Cell.NUM_COLORS + colors[1]] = 1
        return LookaheadSelection(colors, *random.choice(self.rotations_and_positions), frame_parity)

    def onehot_ranges(self, offset):
        # abbreviation
        N = Cell.NUM_COLORS
        return [slice_size(offset, N), slice_size(offset+N, N)]

    def reconstruct(self, offset, params):
        N = Cell.NUM_COLORS
        colors = [reconstruct_onehot(params, offset, N), reconstruct_onehot(params, offset+N, N)]
        return LookaheadSelection(colors, *self.rotations_and_positions[0], 0)

@dataclass(frozen=True)
class LookaheadSelection:
    colors: List[int]
    rotation: int # clockwise
    pos: Position
    frame_parity: int

    def cells(self):
        if self.colors:
            tl_color = self.rotation&2 >> 1
            dpos, tl_shape, other_shape = [
                (Position(8, 0), Cell.LEFT, Cell.RIGHT),
                (Position(0, 8), Cell.UP, Cell.DOWN),
                ][self.rotation&1]
            yield self.pos, Cell(self.colors[tl_color], tl_shape)
            yield self.pos + dpos, Cell(self.colors[1-tl_color], other_shape)

    def render(self, cache, image):
        for pos, cell in self.cells():
            cache.load(cell.template_name(self.frame_parity)).at(image, pos)

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
    lookaheads: Dict[str, Tuple[int, Lookahead]]
    parameter_count: int
    alternative_indices: List[int]

    def select(self, params):
        params[self.index] = 1
        frame_parity = random.randrange(2)
        return SceneSelection(self.name, self.background,
            [layer.select(offset, params) for offset, layer in self.layers],
            [playfield.select(offset, params, frame_parity) for offset, playfield in self.playfields],
            {nm: lookahead.select(offset, params, frame_parity) for nm, (offset, lookahead) in self.lookaheads.items()},
            )

    def render(self, cache, device):
        params = torch.zeros(self.parameter_count)
        image = self.select(params).render(cache)

        slices = []
        for offset, layer in self.layers: slices += layer.onehot_ranges(offset)
        for offset, playfield in self.playfields: slices += playfield.onehot_ranges(offset)
        for offset, lookahead in self.lookaheads.values(): slices += lookahead.onehot_ranges(offset)

        return TrainingExample(image, apply_filters(image), params.to(device), slices, self.alternative_indices)

@dataclass(frozen=True)
class SceneSelection:
    name: List[str]
    background: str
    layers: List[ImageLayerSelection]
    playfields: List[PlayfieldSelection]
    lookaheads: Dict[str, LookaheadSelection]

    def render(self, cache):
        image = numpy.array(cache.load(self.background).image)
        for layer in self.layers: layer.render(cache, image)
        for playfield in self.playfields: playfield.render(cache, image)
        for lookahead in self.lookaheads.values(): lookahead.render(cache, image)
        return image

# the raw data, as close to the directly parsed form as possible
@dataclass
class SceneTree:
    background: Optional[str]
    children: List[Tuple[str, 'SceneTree']]
    layers: List[ImageLayer]
    playfields: List[Playfield]
    lookaheads: Dict[str, Lookahead]
    _scene_parameters: Optional[List[int]] = None
    _parameter_count: Optional[int] = None

    def flatten(self):
        # [x] is a bit like a mutable version of x
        yield from self._flatten(None, [], [], [], {}, [0], self.parameter_count(), self.scene_parameters())

    def _flatten(self, bg, nm, layers, playfields, lookaheads, index, parameter_count, scene_parameters):
        if bg is None: bg = self.background
        elif self.background is not None:
            raise Exception(f"Ambiguous background in {nm}; it was specified here as {self.background} and in an ancestor as {bg}.")

        for layer in self.layers:
            layers.append((index[0], layer))
            index[0] += layer.parameter_count()
        for playfield in self.playfields:
            playfields.append((index[0], playfield))
            index[0] += playfield.parameter_count()

        # backing out the lookahead changes is too annoying, just do COW instead
        if self.lookaheads: lookaheads = dict(lookaheads)
        for k, la in self.lookaheads.items():
            if k in lookaheads:
                offset, old_la = lookaheads[k]
                lookaheads[k] = offset, old_la+la
            else:
                lookaheads[k] = index[0], la
                index[0] += la.parameter_count()

        for child_nm, child_t in self.children:
            nm.append(child_nm)
            yield from child_t._flatten(bg, nm, layers, playfields, lookaheads, index, parameter_count, scene_parameters)
            nm.pop()

        if not self.children:
            if bg is None: raise Exception(f"No background specified for {nm}.")
            yield Scene(list(nm), index[0], bg, list(layers), list(playfields), lookaheads, parameter_count, scene_parameters)
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
                + sum(lookahead.parameter_count() for lookahead in self.lookaheads.values())
                + sum(child.parameter_count() for _, child in self.children)
                )
        return self._parameter_count

    def scene_parameters(self):
        if self._scene_parameters is None:
            self._scene_parameters = []
            self._initialize_scene_parameters(self._scene_parameters, 0)
        return self._scene_parameters

    def _initialize_scene_parameters(self, indices, index):
        index += sum(layer.parameter_count() for layer in self.layers)
        index += sum(playfield.parameter_count() for playfield in self.playfields)
        index += sum(lookahead.parameter_count() for lookahead in self.lookaheads.values())

        for _, child in self.children:
            index = child._initialize_scene_parameters(indices, index)

        if not self.children:
            indices.append(index)
            index += 1

        return index

class Scenes:
    def __init__(self, scene_tree):
        self.list = list(scene_tree.flatten())
        self.indices = [scene.index for scene in self.list]

    def select(self):
        return random.choice(self.list)

    def reconstruct(self, params):
        scene = reconstruct_list(params[self.indices], 0, self.list)
        return SceneSelection(scene.name, scene.background,
            [layer.reconstruct(i, params) for i, layer in scene.layers],
            [playfield.reconstruct(i, params) for i, playfield in scene.playfields],
            {nm: lookahead.reconstruct(i, params) for nm, (i, lookahead) in scene.lookaheads.items()},
            )

def reconstruct_onehot(params, offset, size):
    return torch.argmax(params[slice_size(offset, size)])

def reconstruct_list(params, offset, answers):
    return answers[reconstruct_onehot(params, offset, len(answers))]

@dataclass(frozen=True)
class TrainingExample:
    clean_image: numpy.ndarray
    filtered_image: numpy.ndarray
    classification: torch.tensor
    # each slice is an independent classification problem, so should be part of
    # its own cross-correlation calculation
    onehot_ranges: List[slice]
    # the scene identifiers aren't necessarily contiguous, so there's one final
    # classification problem in an arbitrary collection of indices
    onehot_indices: List[int]

# Create hybrid training examples. Linearly mix a collection of images and
# their classifications. Typically you want to supply a collection with just
# two examples. This can aid in convergence and robustness of the network.
def merge_examples(es):
    weights = [random.random() for _ in es]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    merged_clean_image = sum(map(lambda w, e: w*e.clean_image, weights, es)).astype(numpy.uint8)
    merged_filtered_image = sum(map(lambda w, e: w*e.filtered_image, weights, es)).astype(numpy.uint8)
    merged_class = sum(map(lambda w, e: w*e.classification, weights, es))
    merged_ranges = [x[0] for x in itertools.groupby(sorted(itertools.chain.from_iterable(e.onehot_ranges for e in es)))]
    assert(all(e.onehot_indices == es[0].onehot_indices for e in es))

    return TrainingExample(merged_clean_image, merged_filtered_image, merged_class, merged_ranges, es[0].onehot_indices)

def slice_size(start, length): return slice(start, start+length)

LOCATED_REGEX = re.compile('([1-9][0-9]*)[ \t]*,[ \t]*([1-9][0-9]*)[ \t]*(.+)')
PLAYFIELD_REGEX = re.compile('([1-9][0-9]*)[ \t]*x[ \t]*([1-9][0-9]*)[ \t]*playfield')
LOOKAHEAD_REGEX = re.compile('(0|[1-9][0-9]*)[ \t]*([^ \t\r\n]|[^ \t\r\n][^ \r\n]*[^ \t\r\n])[ \t]*lookahead')
def parse_scene_tree(line_source, parent_indent=None):
    t = SceneTree(None, [], [], [], {})
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
                elif match := LOOKAHEAD_REGEX.fullmatch(thing):
                    la = t.lookaheads.setdefault(match.group(2), Lookahead())
                    la += Lookahead(int(match.group(1)), pos)
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

def filter_hsv(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(numpy.int16)
    dh = round(random.normalvariate(0, 5))
    ds = round(random.normalvariate(0, 20))
    dv = round(random.normalvariate(0, 40))
    image = image + [[[dh, ds, dv]]]
    image[:, :, 0] = image[:, :, 0] % 180
    image[:, :, 1:] = numpy.clip(image[:, :, 1:], 0, 255)
    image = cv2.cvtColor(image.astype(numpy.uint8), cv2.COLOR_HSV2BGR)
    return image

def filter_channel_offsets(image):
    src2tgt, tgt2src = random.choice([
        (cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR),
        (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR),
        (cv2.COLOR_BGR2YUV, cv2.COLOR_YUV2BGR),
        (None, None),
        ])
    rotation = random.randrange(2)

    if src2tgt is not None: image = cv2.cvtColor(image, src2tgt)
    image = numpy.rot90(image, k=rotation)

    period = random.randrange(1,20)
    w = image.shape[1]
    offsets = [[round(random.normalvariate(0, 2)) % w for _ in range(3)] for _ in range(period)]

    for i, channel_offsets in enumerate(offsets):
        for j, offset in enumerate(channel_offsets):
            image[i::period, :, j] = numpy.roll(image[i::period, :, j], offset, axis=1)

    image = numpy.rot90(image, k=-rotation)
    if tgt2src is not None: image = cv2.cvtColor(image, tgt2src)
    return image

def filter_speckle(image):
    while True:
        variance = abs(random.normalvariate(0, 10))
        if variance: break

    image = image + numpy.around(np_rng.normal(0, variance, image.shape)).astype(numpy.int16)
    return numpy.clip(image, 0, 255).astype(numpy.uint8)

def filter_blur(image): return cv2.GaussianBlur(image, None, random.uniform(0.1, 4))

def filter_overlay_photo(image):
    photo = load_random_photo()
    rotations = random.randrange(4)
    image = numpy.rot90(image, rotations)
    photo = numpy.rot90(photo, rotations)

    if random.randrange(2):
        # put the photo in a corner
        while True:
            x = int(abs(random.normalvariate(0, image.shape[1]//10)))
            y = int(abs(random.normalvariate(0, image.shape[0]//10)))
            if 0 < x and x < image.shape[1] and x < photo.shape[1] and 0 < y and y < image.shape[0] and y < photo.shape[0]:
                break
        image[:y, :x, :] = photo[-y:, -x:, :]
    else:
        # put the photo on an edge
        while True:
            x = int(abs(random.normalvariate(0, image.shape[1]//15)))
            y = int(random.normalvariate(image.shape[0]//2, image.shape[0]//15))
            if 0 < x and x < image.shape[1] and x < photo.shape[1] and 0 < y and y < image.shape[0]:
                break

        photo_y = photo.shape[0]//2
        dy_lo = min(y, photo_y)
        dy_hi = min(image.shape[0]-y, photo.shape[0]-photo_y)

        image[y-dy_lo:y+dy_hi, :x, :] = photo[photo_y-dy_lo:photo_y+dy_hi, -x:, :]

    return numpy.rot90(image, -rotations)

def filter_scanline(image):
    m = random.randrange(2, 8)
    b = random.randrange(m)
    if random.randrange(2): image[:, b::m, :] = 0
    else: image[b::m, :, :] = 0
    return image

def filter_linear_gradient(image):
    while True:
        x0 = random.randrange(-image.shape[1]//4, image.shape[1]*5//4)
        x1 = random.randrange(-image.shape[1]//4, image.shape[1]*5//4)
        y0 = random.randrange(-image.shape[0]//4, image.shape[0]*5//4)
        y1 = random.randrange(-image.shape[0]//4, image.shape[0]*5//4)
        if x0 != x1 and y0 != y1: break

    # make a copy the first time to protect against a filter that modifies its input
    image0 = random.choice(ALL_FILTERS)(numpy.array(image))
    image1 = random.choice(ALL_FILTERS)(image)

    indices = numpy.indices(image.shape[0:2], dtype=numpy.float64)
    mask = numpy.clip(((indices[0, :, :] - x0)*(y1-y0) + (indices[1, :, :] - y0)*(x1-x0))/(2*(x1-x0)*(y1-y0)), 0, 1)
    mask = numpy.expand_dims(mask, 2)

    return numpy.around(mask*image0 + (1-mask)*image1).astype(numpy.uint8)

PHOTOS_DIR = "/slow/dmwit/imagenet/ILSVRC/Data/CLS-LOC/train"
def load_random_photo():
    directory = PHOTOS_DIR
    while True:
        _, directories, files = walk(directory).__next__()
        if directories and files: raise Exception(f'Error when loading a random image: directory {directory} has both files and directories.')
        if directories:
            directory = path.join(directory, random.choice(directories))
        elif files:
            background = cv2.imread(path.join(directory, random.choice(files)), cv2.IMREAD_UNCHANGED)
            if len(background.shape) == 2: background = numpy.repeat(background[:, :, numpy.newaxis], 3, axis=2)
            return background
        else:
            raise Exception(f'Error when loading a random image: directory {directory} is empty.')

SCALING_METHODS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
def noisy_scale(image):
    fx = random.normalvariate(3, 0.1)
    fy = random.normalvariate(3, 0.1)
    dx = round(random.normalvariate(0, 4))*fx*8 + random.normalvariate(0, 2) + (fx-3)*image.shape[1]/2
    dy = round(random.normalvariate(0, 4))*fy*8 + random.normalvariate(0, 2) + (fy-3)*image.shape[0]/2

    if not random.randrange(100):
        background = numpy.ones((image.shape[0]*3, image.shape[1]*3, 3), dtype=numpy.uint8) \
            * numpy.array([[[random.randrange(256), random.randrange(256), random.randrange(256)]]], dtype=numpy.uint8)
    else:
        background = load_random_photo()
        bg_scale = 3*max(image.shape[0]/background.shape[0], image.shape[1]/background.shape[1])
        background = cv2.resize(background, None, None, bg_scale, bg_scale)
        background = background[
            slice_size(random.randrange(background.shape[0] - 3*image.shape[0] + 1), 3*image.shape[0]),
            slice_size(random.randrange(background.shape[1] - 3*image.shape[1] + 1), 3*image.shape[1]),
            0:3
            ]

    result = cv2.warpAffine(
        image,
        numpy.array([[fx, 0, dx], [0, fy, dy]]),
        (image.shape[1]*3, image.shape[0]*3),
        background,
        random.choice(SCALING_METHODS),
        cv2.BORDER_TRANSPARENT,
        )
    return result

ALL_FILTERS = [filter_artifacts, filter_hsv, filter_channel_offsets, filter_speckle, filter_blur, filter_overlay_photo, filter_scanline, filter_linear_gradient]
def apply_filters(image):
    image = noisy_scale(image)
    while random.randrange(5):
        image = random.choice(ALL_FILTERS)(image)
    return image

LEAKAGE = 0.1

# follows advice from Delving Deep into Rectifiers
class Conv2dReLUInit(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, *args, leakage=LEAKAGE, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, *args, **kwargs)

        try:
            w, h = kernel_size
        except TypeError:
            kernel_param_count = kernel_size*kernel_size
        else:
            kernel_param_count = w*h
        nn.init.normal_(self.weight, std=math.sqrt(2/((1+leakage)*kernel_param_count*in_channels)))

def conv_block(in_channels, out_channels, kernel_size, stride, device):
    return nn.Sequential(
        Conv2dReLUInit(in_channels, out_channels, kernel_size, stride=stride, padding=1, padding_mode='reflect', device=device),
        nn.BatchNorm2d(out_channels, device=device),
        # paper uses ReLU instead of LeakyReLU
        nn.LeakyReLU(LEAKAGE),
        )

class Residual(nn.Module):
    def __init__(self, channels, device):
        super().__init__()
        # paper uses range(2) instead of range(3)
        self.block = nn.Sequential(*(conv_block(channels, channels, 3, 1, device) for _ in range(3)))

    def forward(self, x):
        # paper puts the last nonlinearity after re-adding to x rather than before
        return x+self.block(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, device):
        super().__init__()
        self.projection = Conv2dReLUInit(in_channels, 2*in_channels, 1, leakage=1, stride=2, device=device)
        self.block = nn.Sequential(
            conv_block(in_channels, 2*in_channels, 3, 2, device),
            # paper uses range(1) instead of range(2)
            *(conv_block(2*in_channels, 2*in_channels, 3, 1, device) for _ in range(2)),
            )

    def forward(self, x):
        return self.projection(x)+self.block(x)

class ParseImage(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.block = nn.Sequential(
            # paper uses 64 7x7 kernels and a stride of 2, but also a much smaller input image
            Conv2dReLUInit(3, 16, 11, stride=2, device=device),
            nn.MaxPool2d(3, stride=2),
            nn.BatchNorm2d(16, device=device),
            nn.LeakyReLU(LEAKAGE),
            )

    def forward(self, x):
        return self.block(x-127.5)

class Classifier(nn.Module):
    def __init__(self, scene_tree, device):
        super().__init__()
        blocks = [ParseImage(device)]
        channels = 16
        for length in [3,4,7,4,3,3]:
            blocks.extend(Residual(channels, device) for _ in range(length))
            blocks.append(Downsample(channels, device))
            channels *= 2
        self.block = nn.Sequential(*blocks)

        # for lazy initialization
        self.device = device
        self.fully_connected = None
        self.out_size = scene_tree.parameter_count()

    def forward(self, x):
        # cv2 does HxWxC but torch.nn expects CxWxH, so we need to transpose
        unstructured_features = torch.flatten(self.block(torch.transpose(x, 1, 3)), start_dim=1)
        if self.fully_connected is None:
            self.fully_connected = nn.Linear(unstructured_features.shape[1], self.out_size, device=self.device)
        return self.fully_connected(unstructured_features)

def xe_loss_single(probs, target):
    return nn.functional.cross_entropy(torch.unsqueeze(probs, 0), torch.unsqueeze(target, 0), label_smoothing=0.01)

def losses(tensor, golden, device):
    ls = torch.zeros((len(golden),), dtype=torch.float, device=device)
    for i, example in enumerate(golden):
        ls[i] = xe_loss_single(tensor[i, example.onehot_indices], example.classification[example.onehot_indices])
        for r in example.onehot_ranges:
            ls[i] += xe_loss_single(tensor[i, r], example.classification[r])
        ls[i] /= len(example.onehot_ranges)+1
    return ls

def loss(tensor, golden, device):
    return torch.sum(losses(tensor, golden, device))/max(len(golden), 1)

def global_step(epoch, batch, step):
    return step + STEPS_PER_BATCH*(batch + BATCHES_PER_EPOCH*epoch)

with open('layouts/layered.txt') as f:
    _, _, t = parse_scene_tree(f)
scenes = Scenes(t)

dev = torch.device('cuda')
c = Classifier(t, dev)
cache = TemplateCache()
# initialize the fully-connected layer
with torch.no_grad(): c(torch.tensor(numpy.array([t.flatten().__next__().render(cache, dev).filtered_image]), device=dev))
opt = torch.optim.SGD(c.parameters(), 0.01, momentum=0.9, weight_decay=0.00001)
lr_schedule = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
log = tx.SummaryWriter()

EPOCHS=100
BATCHES_PER_EPOCH=50
STEPS_PER_BATCH=10
MIXES_PER_BATCH=60
EXAMPLES_PER_MIX=2
TEST_EXAMPLES=180
BATCHES_PER_RECONSTRUCTION=40

test = []
for _ in range(TEST_EXAMPLES):
    test.append(scenes.select().render(cache, dev))
test_tensor = torch.tensor(numpy.array([x.filtered_image for x in test]), device=dev)

for epoch in range(EPOCHS):
    for batch in range(BATCHES_PER_EPOCH):
        train = []
        for _ in range(EXAMPLES_PER_MIX*MIXES_PER_BATCH):
            train.append(scenes.select().render(cache, dev))
        train = [merge_examples(train[i::MIXES_PER_BATCH]) for i in range(MIXES_PER_BATCH)]
        train_tensor = torch.tensor(numpy.array([x.filtered_image for x in train]), device=dev)

        c.train(mode=True)
        for step in range(STEPS_PER_BATCH):
            opt.zero_grad()
            log.add_scalar('training loss', l := loss(c(train_tensor), train, dev), n := global_step(epoch, batch, step))
            if step == 0: log.add_scalar('batch start training loss', l, n)
            if step == STEPS_PER_BATCH-1: log.add_scalar('batch end training loss', l, n)
            l.backward()
            opt.step()
        c.train(mode=False)

        with torch.no_grad():
            classifications = c(test_tensor)
            ls = losses(classifications, test, dev).to('cpu')

            log.add_scalar('test loss', torch.sum(ls)/ls.shape[0], n)
            if not (n//STEPS_PER_BATCH) % BATCHES_PER_RECONSTRUCTION:
                classifications = classifications.to('cpu')
                _, indices = torch.sort(ls)
                clean_h, clean_w, _ = test[0].clean_image.shape
                filtered_h, filtered_w, _ = test[0].filtered_image.shape
                h = clean_h + filtered_h
                w = max(2*clean_w, filtered_w)

                reconstructions = numpy.zeros((10, h, w, 3), dtype=numpy.uint8)
                for i in [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]:
                    reconstructions[i, :clean_h, :clean_w, :] = test[indices[i]].clean_image
                    reconstructions[i, :clean_h, clean_w:2*clean_w, :] = scenes.reconstruct(classifications[indices[i]]).render(cache)
                    reconstructions[i, clean_h:, :filtered_w, :] = test[indices[i]].filtered_image
                # OpenCV uses BGR by default, tensorboardX uses RGB by default
                reconstructions = numpy.flip(reconstructions, 3)
                log.add_images('good reconstructions', reconstructions[:5], n, dataformats='NHWC')
                log.add_images('bad reconstructions', reconstructions[5:], n, dataformats='NHWC')

    lr_schedule.step()
