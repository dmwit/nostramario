import cv2
import dataclasses
import itertools
import math
import numpy
import numpy.random
import os.path
import random
import re
import statistics
import sys
import torch

from dataclasses import dataclass
from os import path, walk
from torch import nn
from typing import Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union

from . import params

class Boundaries:
    def __init__(self, m, b):
        self.m = m
        self.b = b
        self.normalize_b()

    def __call__(self, x): return self.m*x + self.b
    def ipoint(self, x): return round(self(x))
    def size(self, pixels): return math.ceil(self.to_coord(pixels))
    def range(self, pixels): return range(self.size(pixels))

    def to_coord(self, pixels): return (pixels-self.b)/self.m
    def floor(self, pixels): return math.floor(self.to_coord(pixels))
    def ceil(self, pixels): return math.ceil(self.to_coord(pixels))
    def round(self, pixels): return round(self.to_coord(pixels))

    def draw(self, img):
        max_coord_this = img.shape[0]-1
        max_coord_other = img.shape[1]-1
        img = img.copy()
        for grid in self.range(max_coord_this):
            v = self.ipoint(grid)
            cv2.line(img, (0, v), (max_coord_other, v), (255, 255, 0))
        return img

    def normalize_b(self):
        self.b -= self.m*math.floor(self.b/self.m)

    def __str__(self):
        return 'y = {:.2f}*x + {:.2f}'.format(self.m, self.b)

    def __repr__(self):
        return 'Boundaries({}, {})'.format(repr(self.m), repr(self.b))

    def __eq__(self, other):
        try: return self is other or (self.m == other.m and self.b == other.b)
        except AttributeError: return NotImplemented

class Grid:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, x, y): return (self.x(x), self.y(y))
    def ipoint(self, x, y): return (self.x.ipoint(x), self.y.ipoint(y))
    def size(self, x, y): return (self.x.size(x), self.y.size(y))
    def range(self, img_or_shape):
        try:
            h, w = img_or_shape.shape[0:2]
        except AttributeError:
            w, h = img_or_shape
        return ((x, y) for x in self.x.range(w) for y in self.y.range(h))

    def to_coord(self, x_pixels, y_pixels): return (self.x.to_coord(x_pixels), self.y.to_coord(y_pixels))
    def floor(self, x_pixels, y_pixels): return (self.x.floor(x_pixels), self.y.floor(y_pixels))
    def ceil(self, x_pixels, y_pixels): return (self.x.ceil(x_pixels), self.y.ceil(y_pixels))
    def round(self, x_pixels, y_pixels): return (self.x.round(x_pixels), self.y.round(y_pixels))

    def draw(self, img):
        img = self.x.draw(numpy.transpose(img, (1,0,2)))
        img = self.y.draw(numpy.transpose(img, (1,0,2)))
        return img

    def __str__(self):
        return 'x = {:.2f}*col + {:.2f}; y = {:.2f}*row + {:.2f}'.format(
                self.x.m,
                self.x.b,
                self.y.m,
                self.y.b)

    def __repr__(self):
        return 'Grid({}, {})'.format(repr(self.x), repr(self.y))

    def __eq__(self, other):
        try: return self is other or (self.x == other.x and self.y == other.y)
        except AttributeError: return NotImplemented

def is_peak(hist, min_peak, i):
    return hist[i] > min_peak and (i == 0 or i == hist.size-1 or (hist[i] >= hist[i-1] and hist[i] >= hist[i+1]))

def find_squares(img):
    h, w, _ = img.shape
    kernel_size = 2*round((h+w)/600)+1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # First, we try to find the color black and whatever the second darkest
    # color is, so we can put the threshold between those two. We'll say that
    # any sufficiently high "peaks" (local maxima, including endpoints) qualify
    # as a color.
    brightness_hist = cv2.calcHist([gray], [0], None, [256], [0, 255])
    min_peak = numpy.sum(brightness_hist)/128 # twice the average count seems to rule out local maxima I don't care about
    for i in range(256):
        if is_peak(brightness_hist, min_peak, i):
            first_peak = i
            break
    else: raise Exception("no sufficiently high peaks")
    second_peak = first_peak+1
    while brightness_hist[second_peak] == brightness_hist[first_peak]:
        if second_peak == 255: break
        second_peak = second_peak+1
    for i in range(second_peak, 256):
        if is_peak(brightness_hist, min_peak, i):
            second_peak = i
            break
    else: raise Exception("only one sufficiently high peak")

    # Now do a simple thresholding and find contours. Simple thresholding
    # should be okay because I don't think lighting plays a part in any of the
    # video streams we care about for now. If that changes in the future we
    # will need to revisit this.
    _, bw = cv2.threshold(gray, math.ceil((first_peak + second_peak)/2), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Pick out the contours that are "square-like". They're "square-like" if
    # they almost fill their axis-aligned bounding box (we don't handle
    # rotations here for now) and if their aspect ratio is pretty close to 1.
    # But we let the aspect ratio be pretty lax, because some people play with
    # settings that make the boxes have a more different aspect ratio.
    squares = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        rx, ry, rw, rh = rect
        area = cv2.contourArea(contour)
        if 0 in [rw, rh, area]: continue
        if rw*rh/area < 1.3 and max(rw,rh)/min(rw,rh) < 1.4 and rx>0 and ry>0 and rx+rw<=w and ry+rh<=h:
            squares.append(rect)

    return squares

def learn_boundaries_from_intervals(xws, threshold=0.2, rates=[0.1, 0.1, 0.1]):
    w = statistics.mode(w for _, w in xws)
    lo = w*(1-threshold)
    hi = w*(1+threshold)
    xs = [x for x, w in xws if w >= lo and w <= hi]
    ws = [w for x, w in xws if w >= lo and w <= hi]
    xs.sort()
    b = Boundaries(statistics.mean(ws), statistics.mode(xs))

    for rate in rates:
        # learn m
        for x in xs:
            dx = x-b.b
            i = round(dx/b.m)
            if i!= 0: b.m += rate*(dx/i-b.m)

        # learn b
        dxb = 0
        for x in xs:
            dxb += x-b(round((x-b.b)/b.m))
        b.b += dxb/len(xs)

    return b

def learn_grid_from_img(img):
    squares = find_squares(img)
    return Grid(
        learn_boundaries_from_intervals([(x, w) for x, _, w, _ in squares]),
        learn_boundaries_from_intervals([(y, h) for _, y, _, h in squares])
        )

np_rng = numpy.random.default_rng()
T = TypeVar('T')

# Sample from a distribution specified by a list of values and a positive
# integer weight for each using a variant of the alias method. Advantages of
# the approach used here:
#
# * O(1) sampling, which is asymptotically better than the O(log n) you'd
#   get from the obvious binary search algorithm.
# * Sampling only requires the generation of a single integer, which
#   potentially consumes many fewer random bits than generating a uniform float
#   in [0, 1) as needed by the standard alias method.
# * No floating point arithmetic (nor fixed-point arithmetic), and in
#   particular no rounding; all calculations are exact. No bias, and no
#   distribution with an outcome so unlikely that it cannot be selected due to
#   rounding errors. (Admittedly this latter part is a bit theoretical in that
#   outcomes so unlikely that they get skipped by rounding errors are
#   incredibly unlikely to get generated anyway...)
# * O(n) precomputation. The popular O(n*log(n)) variant uses a greedy
#   algorithm to decrease how often you reject the default half of a bucket.
#   Since this stores both default and rejection values in a single list, that
#   optimization is not needed; there is no additional indexing stemming from a
#   rejection.
class Categorical(Generic[T]):
    def __init__(self, dist):
        self._buckets: List[Tuple[int, (int, T), (int, T)]] = []
        # could divide all the p's by the gcd of all the p's but I'm lazy
        self._denominator: int = math.lcm(len(dist), p_sum := sum(p for p, _ in dist))
        self._bucket_size: int = self._denominator // len(dist)
        self.values: Tuple[T, ...] = tuple(v for _, v in dist)
        scaling = self._denominator // p_sum
        overfull = []
        underfull = []

        def place(p, val):
            if p < self._bucket_size:
                underfull.append((p, val))
            elif p == self._bucket_size:
                self._buckets.append((p, val, val))
            else: # p > self._bucket_size
                overfull.append((p, val))
        for i, (p, val) in enumerate(dist): place(scaling*p, (i, val))

        while underfull:
            p_small, val_small = underfull.pop()
            p_large, val_large = overfull.pop()
            self._buckets.append((p_small, val_small, val_large))
            place(p_large + p_small - self._bucket_size, val_large)

        assert(not overfull)

    def select(self) -> T: return self.select_with_index()[1]
    def select_with_index(self) -> Tuple[int, T]:
        bucket, residue = divmod(random.randrange(self._denominator), self._bucket_size)
        p, lo, hi = self._buckets[bucket]
        return lo if residue < p else hi

    def map(self, f):
        self.values = tuple(map(f, self.values))
        self._buckets = list((w, (i_lo, f(v_lo)), (i_hi, f(v_hi))) for w, (i_lo, v_lo), (i_hi, v_hi) in self._buckets)

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

    def round(self): return Position(round(self.x), round(self.y))

Position.LEFT = Position(-1, 0)
Position.RIGHT = Position(1, 0)
Position.UP = Position(0, -1)
Position.DOWN = Position(0, 1)
Position.EVERY_WHICH_WAY = (Position.LEFT, Position.RIGHT, Position.UP, Position.DOWN)

@dataclass(frozen=True)
class ImageLayer:
    pos: Position
    learn: bool
    images: Categorical[str]

    def parameter_count(self):
        return len(self.images.values) if self.learn else 0

    def onehot_ranges(self, offset):
        return [slice_size(offset, len(self.images.values))] if self.learn else []

    def select(self, offset, params):
        i, image = self.images.select_with_index()
        if self.learn: params[offset+i] = 1
        return ImageLayerSelection(self.pos, image)

    def reconstruct(self, offset, params):
        return \
            ImageLayerSelection(self.pos, reconstruct_list(params, offset, self.images.values)) \
            if self.learn else \
            ImageLayerSelection(self.pos, self.images.values[0])

@dataclass(frozen=True)
class ImageLayerSelection:
    pos: Position
    image: str

    def render(self, cache, image):
        cache.load(self.image).at(image, self.pos)

@dataclass(frozen=True)
class ImageDirectoryLayer:
    pos: Position
    directories: Categorical[str]

    def __post_init__(self):
        self.directories.map(lambda d: params.PHOTOS_DIR if d == '$' else d)

    def parameter_count(self): return 0
    def onehot_ranges(self, offset): return []

    def select(self, offset, params):
        return ImageDirectoryLayerSelection(self.pos, load_random_photo(self.directories.select()))

    def reconstruct(self, offset, params):
        directories = self.directories.values
        directory = ''
        while directories:
            directory = path.join(directory, directories[0])
            _, directories, files = walk(directory).__next__()
        return ImageDirectoryLayerSelection(self.pos, load_exact_photo(path.join(directory, files[0])))

@dataclass(frozen=True)
class ImageDirectoryLayerSelection:
    pos: Position
    image: numpy.ndarray

    def __init__(self, pos, image):
        if len(image.shape) == 2:
            image = numpy.repeat(image[:, :, numpy.newaxis], 3, axis=2)
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        object.__setattr__(self, 'pos', pos)
        object.__setattr__(self, 'image', image)

    def render(self, cache, image):
        sh, sw, _ = self.image.shape
        th, tw, _ = image.shape
        ah, aw = th - self.pos.y, tw - self.pos.x
        h, w = min(sh, ah), min(sw, aw)
        image[slice_size(self.pos.y, h), slice_size(self.pos.x, w), :] = self.image[:h, :w, :]

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

MAGNIFIER_VIRUS_RADIUS = 18
MAGNIFIER_VIRUS_SIZE = 24
MAGNIFIER_VIRUS_SQUIRM_OFFSETS = [5, 9, 12, 14, 15, 14, 12, 9, 5]
MAGNIFIER_VIRUS_SQUIRM_LENGTH = 170 + len(MAGNIFIER_VIRUS_SQUIRM_OFFSETS)

@dataclass(frozen=True)
class Magnifier:
    center: Position

    def __post_init__(self):
        object.__setattr__(self, 'center', self.center - Position(MAGNIFIER_VIRUS_SIZE/2., MAGNIFIER_VIRUS_SIZE/2.))

    def parameter_count(self): return 0
    def onehot_ranges(self, offset): return []
    def select(self, offset, params):
        theta = random.uniform(0, 2*math.pi)
        viruses = []
        for color in range(Cell.NUM_COLORS):
            # sometimes don't draw a virus
            if not random.randrange(30): continue
            # 540: 3s/virus*60fps*3virus/virus of this color
            frame = random.randrange(540)
            # TODO: figure out and implement the actual NES logic, which surely does not calculate trig functions
            dy = MAGNIFIER_VIRUS_SQUIRM_OFFSETS[frame] if frame < len(MAGNIFIER_VIRUS_SQUIRM_OFFSETS) else 0
            local_theta = theta + 2*math.pi*color/Cell.NUM_COLORS
            direction = Position(math.cos(local_theta), math.sin(local_theta))
            viruses.append(MagnifierVirusSelection(
                (self.center + MAGNIFIER_VIRUS_RADIUS*direction - Position(0, dy)).round(),
                color,
                frame < MAGNIFIER_VIRUS_SQUIRM_LENGTH,
                ))
        return MagnifierSelection(random.randrange(8), viruses)

    def reconstruct(self, offset, params):
        return MagnifierSelection(0, [])

@dataclass(frozen=True)
class MagnifierVirusSelection:
    pos: Position
    color: int
    # TODO: laughing is also an option (when game is lost)
    squirm: bool

    NORMAL_TEMPLATES = ['lean-l', 'stand', 'lean-r', 'stand']
    SQUIRM_TEMPLATES = ['lean', 'stand']

    def render(self, cache, image, frame_parity):
        color = Cell._COLOR_CHARS[self.color]
        if self.squirm:
            stance = 'squirm-'
            animation = MagnifierVirusSelection.SQUIRM_TEMPLATES[(frame_parity >> 2) & 1]
        else:
            stance = ''
            animation = MagnifierVirusSelection.NORMAL_TEMPLATES[frame_parity & 3]
        template = f'magnifier-{color}x-{stance}{animation}.png'
        cache.load(template).at(image, self.pos)

@dataclass(frozen=True)
class MagnifierSelection:
    frame_parity: int
    viruses: List[MagnifierVirusSelection]

    def render(self, cache, image):
        for virus in self.viruses:
            virus.render(cache, image, self.frame_parity)

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

    def onehot_ranges(self, offset):
        return [s
            for pos in self.positions()
            for s in self.cells.get(pos, EMPTY_CELL).onehot_ranges(offset + Playfield.parameter_offset(self, pos))
            ]

    def positions(self):
        for x in range(self.w):
            for y in range(-1, self.h):
                yield Position(x, y)

@dataclass
class Lookahead:
    # the int is how many clockwise rotations the pill has
    rotations_and_positions: List[Tuple[int, Tuple[int, Position]]]
    _dist: Union[None, Categorical[Tuple[int, Position]]]

    def __init__(self, rot=None, pos=None, weight=1):
        self.rotations_and_positions = [] if rot is None or pos is None else [(weight, (rot, pos))]
        self._dist = None

    def __add__(self, other):
        result = Lookahead()
        result.rotations_and_positions = self.rotations_and_positions + other.rotations_and_positions
        return result

    def __iadd__(self, other):
        self.rotations_and_positions += other.rotations_and_positions
        self._dist = None

    def dist(self):
        if self._dist is None: self._dist = Categorical(self.rotations_and_positions)
        return self._dist

    def parameter_count(self):
        return 2*Cell.NUM_COLORS

    def select(self, offset, params, frame_parity):
        colors = [random.randrange(Cell.NUM_COLORS), random.randrange(Cell.NUM_COLORS)]
        params[offset + colors[0]] = 1
        params[offset + Cell.NUM_COLORS + colors[1]] = 1
        return LookaheadSelection(colors, *self.dist().select(), frame_parity)

    def onehot_ranges(self, offset):
        # abbreviation
        N = Cell.NUM_COLORS
        return [slice_size(offset, N), slice_size(offset+N, N)]

    def reconstruct(self, offset, params):
        N = Cell.NUM_COLORS
        colors = [reconstruct_onehot(params, offset, N), reconstruct_onehot(params, offset+N, N)]
        return LookaheadSelection(colors, *self.dist().values[0], 0)

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
    layers: List[Tuple[int, Union[ImageLayer, ImageDirectoryLayer]]]
    playfields: List[Tuple[str, Tuple[int, Playfield]]]
    lookaheads: Dict[str, Tuple[int, Lookahead]]
    magnifiers: List[Tuple[int, Magnifier]]
    parameter_count: int
    alternative_indices: List[int]

    def select(self, params):
        params[self.index] = 1
        frame_parity = random.randrange(2)
        return SceneSelection(self.name, self.background,
            [layer.select(offset, params) for offset, layer in self.layers],
            [(nm, playfield.select(offset, params, frame_parity)) for nm, (offset, playfield) in self.playfields],
            {nm: lookahead.select(offset, params, frame_parity) for nm, (offset, lookahead) in self.lookaheads.items()},
            [magnifier.select(offset, params) for offset, magnifier in self.magnifiers],
            )

    def render(self, cache, device):
        params = torch.zeros(self.parameter_count)
        selection = self.select(params)
        image = selection.render(cache)

        slices = []
        for offset, layer in self.layers: slices += layer.onehot_ranges(offset)
        for (_, playfield), (_, (offset, _)) in zip(selection.playfields, self.playfields):
            slices += playfield.onehot_ranges(offset)
        for offset, lookahead in self.lookaheads.values(): slices += lookahead.onehot_ranges(offset)
        for offset, magnifier in self.magnifiers: slices += magnifier.onehot_ranges(offset)

        return TrainingExample(image, apply_filters(image), params.to(device), slices, self.alternative_indices)

@dataclass(frozen=True)
class SceneSelection:
    name: List[str]
    background: str
    layers: List[Union[ImageLayerSelection, ImageDirectoryLayerSelection]]
    playfields: List[Tuple[str, PlayfieldSelection]]
    lookaheads: Dict[str, LookaheadSelection]
    magnifiers: List[MagnifierSelection]

    def render(self, cache):
        image = numpy.array(cache.load(self.background).image)
        # do layers after the playfield so the win/loss layers can occlude the playfield
        for _, playfield in self.playfields: playfield.render(cache, image)
        for layer in self.layers: layer.render(cache, image)
        # TODO: perhaps we should (think about and) specify the order this
        # happens in a bit more carefully...
        for lookahead in self.lookaheads.values(): lookahead.render(cache, image)
        for magnifier in self.magnifiers: magnifier.render(cache, image)
        return image

# the raw data, as close to the directly parsed form as possible
@dataclass
class SceneTree:
    background: Optional[str]
    children: List[Tuple[str, int, 'SceneTree']]
    layers: List[Union[ImageLayer, ImageDirectoryLayer]]
    playfields: List[Tuple[str, Playfield]]
    lookaheads: Dict[str, Lookahead]
    magnifiers: List[Magnifier]
    _scene_parameters: Optional[List[int]] = None
    _parameter_count: Optional[int] = None

    def flatten(self):
        # [x] is a bit like a mutable version of x
        yield from self._flatten(None, [], [], [], {}, [], [0], self.parameter_count(), self.scene_parameters(), self._denominator())

    def _flatten(self, bg, nm, layers, playfields, lookaheads, magnifiers, index, parameter_count, scene_parameters, weight):
        if bg is None: bg = self.background
        elif self.background is not None:
            raise Exception(f"Ambiguous background in {nm}; it was specified here as {self.background} and in an ancestor as {bg}.")

        for layer in self.layers:
            layers.append((index[0], layer))
            index[0] += layer.parameter_count()
        for k, playfield in self.playfields:
            playfields.append((k, (index[0], playfield)))
            index[0] += playfield.parameter_count()
        for magnifier in self.magnifiers:
            magnifiers.append((index[0], magnifier))
            index[0] += magnifier.parameter_count()

        # backing out the dictionary changes is too annoying, just do COW instead
        if self.lookaheads: lookaheads = dict(lookaheads)
        for k, la in self.lookaheads.items():
            if k in lookaheads:
                offset, old_la = lookaheads[k]
                lookaheads[k] = offset, old_la+la
            else:
                lookaheads[k] = index[0], la
                index[0] += la.parameter_count()

        scaling = sum(child_weight for _, child_weight, _ in self.children) or 1
        for child_nm, child_weight, child_t in self.children:
            nm.append(child_nm)
            yield from child_t._flatten(bg, nm, layers, playfields, lookaheads, magnifiers, index, parameter_count, scene_parameters, (weight * child_weight) // scaling)
            nm.pop()

        if not self.children:
            if bg is None: raise Exception(f"No background specified for {nm}.")
            if len(set(pid for pid, _ in self.playfields)) != len(self.playfields):
                raise Exception(f"Duplicate playfield name.") # TODO: say which name in the error
            yield (weight, Scene(list(nm), index[0], bg, list(layers), list(playfields), lookaheads, list(magnifiers), parameter_count, scene_parameters))
            index[0] += 1

        # -0 is 0, so we need a conditional
        if self.layers: del layers[-len(self.layers):]
        if self.playfields: del playfields[-len(self.playfields):]
        if self.magnifiers: del magnifiers[-len(self.magnifiers):]

    def parameter_count(self):
        if self._parameter_count is None:
            self._parameter_count = (0
                + (0 if self.children else 1)
                + sum(layer.parameter_count() for layer in self.layers)
                + sum(playfield.parameter_count() for _, playfield in self.playfields)
                + sum(lookahead.parameter_count() for lookahead in self.lookaheads.values())
                + sum(child.parameter_count() for _, _, child in self.children)
                )
        return self._parameter_count

    def scene_parameters(self):
        if self._scene_parameters is None:
            self._scene_parameters = []
            self._initialize_scene_parameters(self._scene_parameters, 0)
        return self._scene_parameters

    def _initialize_scene_parameters(self, indices, index):
        index += sum(layer.parameter_count() for layer in self.layers)
        index += sum(playfield.parameter_count() for _, playfield in self.playfields)
        index += sum(lookahead.parameter_count() for lookahead in self.lookaheads.values())

        for _, _, child in self.children:
            index = child._initialize_scene_parameters(indices, index)

        if not self.children:
            indices.append(index)
            index += 1

        return index

    def _denominator(self): return math.lcm(*self._denominators(1))
    def _denominators(self, parent_denom):
        if not self.children:
            if parent_denom:
                yield parent_denom
        else:
            self_denom = sum(child_weight for _, child_weight, _ in self.children) or 1
            for _, _, child in self.children:
                yield from child._denominators(parent_denom*self_denom)

class Scenes:
    def __init__(self, scene_tree):
        scenes = list(scene_tree.flatten())
        self.dist = Categorical(scenes)
        self.indices = [scene.index for scene in self.dist.values]

    def select(self): return self.dist.select()

    def reconstruct(self, params):
        scene = reconstruct_list(params[self.indices], 0, self.dist.values)
        return SceneSelection(scene.name, scene.background,
            [layer.reconstruct(i, params) for i, layer in scene.layers],
            [(nm, playfield.reconstruct(i, params)) for nm, (i, playfield) in scene.playfields],
            {nm: lookahead.reconstruct(i, params) for nm, (i, lookahead) in scene.lookaheads.items()},
            [magnifier.reconstruct(i, params) for i, magnifier in scene.magnifiers],
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

    def to(self, dev): return TrainingExample(self.clean_image, self.filtered_image, self.classification.to(device=dev), self.onehot_ranges, self.onehot_indices)

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

NAME_REGEX = re.compile('([^\r\n]+?)([ \t]*\(([1-9][0-9]*)\))?[ \t]*:')
LOCATED_REGEX = re.compile('(0|[1-9][0-9]*)[ \t]*,[ \t]*(0|[1-9][0-9]*)[ \t]*(.+)')
PLAYFIELD_REGEX = re.compile('([1-9][0-9]*)[ \t]*x[ \t]*([1-9][0-9]*)[ \t]*([^ \t\r\n]|[^ \t\r\n][^ \r\n]*[^ \t\r\n])[ \t]*playfield')
LOOKAHEAD_REGEX = re.compile('(0|[1-9][0-9]*)[ \t]*([^ \t\r\n]|[^ \t\r\n][^ \r\n]*[^ \t\r\n])[ \t]*lookahead([ \t]*\(([1-9][0-9]*)\))?')
def parse_scene_tree(line_source, parent_indent=None):
    t = SceneTree(None, [], [], [], {}, [])
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

            if match := NAME_REGEX.fullmatch(content):
                here_indent, content, child = parse_scene_tree(line_source, self_indent)
                t.children.append((match.group(1), weight_from_group(match.group(3)), child))
            elif match := LOCATED_REGEX.fullmatch(content):
                pos = Position(int(match.group(1)), int(match.group(2)))
                thing = match.group(3)
                if match := PLAYFIELD_REGEX.fullmatch(thing):
                    t.playfields.append((match.group(3), Playfield(pos, int(match.group(1)), int(match.group(2)))))
                elif match := LOOKAHEAD_REGEX.fullmatch(thing):
                    la = t.lookaheads.setdefault(match.group(2), Lookahead())
                    la += Lookahead(int(match.group(1)), pos, weight_from_group(match.group(4)))
                elif thing == 'magnifier':
                    t.magnifiers.append(Magnifier(pos))
                elif thing.startswith('directories'):
                    t.layers.append(ImageDirectoryLayer(pos, parse_categorical(thing[11:])))
                else:
                    learn = thing[0] != '!'
                    if not learn: thing = thing[1:]
                    t.layers.append(ImageLayer(pos, learn, parse_categorical(thing)))
                break
            elif t.background == None:
                t.background = content
                break
            else:
                raise Exception(f"Attempted to set background to {content}, but there is already a background of {t.background}.")

    return ('', '', t)

CATEGORICAL_REGEX = re.compile('[ \t]*(.+?)[ \t]*(\(([1-9][0-9]*)\))?[ \t]*')
def parse_categorical(ss):
    return Categorical([
        ( weight_from_group((match := CATEGORICAL_REGEX.fullmatch(s)).group(3))
        , match.group(1)
        )
        for s in ss.split(';')
        ])

def weight_from_group(s):
    # don't use int(s or 1) in case s is '' for some weird reason
    if s is None: return 1
    return int(s)

SPACE_REGEX = re.compile('([ \t]*)(.*[^ \t\r\n])?[ \t\r\n]*')
def split_indentation(line):
    match = SPACE_REGEX.fullmatch(line)
    return (match.group(1), match.group(2) or '')

# You might wonder why this pattern happens here:
#     def foo(bar = None):
#         if bar is None: bar = params.baz
# Instead of this one:
#     def foo(bar = params.baz):
# In the former, params.baz is re-evaluated on each function call, while in the
# latter, params.baz is evaluated just once, when the function is first
# defined. I'm not 100% on that). Since we're dynamically reloading the params
# module, we want to re-evaluate.
def load_random_photo(directory = None):
    if directory is None: directory = params.PHOTOS_DIR
    while True:
        _, directories, files = walk(directory).__next__()
        if directories and files: raise Exception(f'Error when loading a random image: directory {directory} has both files and directories.')
        if directories:
            directory = path.join(directory, random.choice(directories))
        elif files:
            return load_exact_photo(path.join(directory, random.choice(files)))
        else:
            raise Exception(f'Error when loading a random image: directory {directory} is empty.')

def load_exact_photo(filename):
    background = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # I saw a crash on accessing background.shape once. If it happens again,
    # this print() will help work out why.
    if background is None: print(filename)
    if len(background.shape) == 2: background = numpy.repeat(background[:, :, numpy.newaxis], 3, axis=2)
    if background.shape[2] == 4: background = background[:, :, 0:3]
    return background

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

SCALING_METHODS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
def noisy_scale(image):
    fx = random.normalvariate(3, 0.1)
    fy = random.normalvariate(3, 0.1)
    dx = random.normalvariate(0, 2) + (3-fx)*image.shape[1]/2
    dy = random.normalvariate(0, 2) + (3-fy)*image.shape[0]/2

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

DEFAULT_SCENE_TREE_LOCATION = 'layouts/layered.txt'
