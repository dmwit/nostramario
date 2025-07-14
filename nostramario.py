from enum import Enum
from PIL import Image
import cv2
import math
import numpy as np
import pathlib
import sys
import time

def lerp(lo, hi, progress): return lo*(1-progress)+hi*progress

def division_estimate(num, denom):
    return (num/(denom+1), num/denom, num/(denom-1))

def estimate_grid_size(img):
    rgb = np.uint32(img)
    brightness = np.sqrt((rgb*rgb).sum(2,dtype=np.uint32))
    fft_brightness = np.fft.fft2(brightness)
    mags = np.abs(fft_brightness)
    angles = np.angle(fft_brightness)
    rs, cs = mags.shape
    halfrs = (rs+1) // 2
    halfcs = (cs+1) // 2
    folded = mags[1:halfrs,1:halfcs] + mags[1:halfrs,-1:-halfcs:-1] # just duplicates: + mags[-1:-halfrs:-1,1:halfcs] + mags[-1:-halfrs:-1,-1:-halfcs:-1]
    # assume we aren't zoomed way into some small part of the 256x224 NES output, so we should ignore very low-frequency signals
    folded[:rs//224+1,:] = -float('inf')
    folded[:,:cs//256+1] = -float('inf')
    rmax, cmax = np.unravel_index(np.argmax(folded), folded.shape)
    return (division_estimate(rs/2, rmax+1), division_estimate(cs/2, cmax+1))

def grid_search(f, xlo, xhi, ylo, yhi, resolution=2):
    scores = np.zeros((resolution, resolution))
    while True:
        for i in range(resolution):
            for j in range(resolution):
                x = lerp(xlo, xhi, (i+0.5)/resolution)
                y = lerp(ylo, yhi, (j+0.5)/resolution)
                scores[i, j] = f(x, y)

        best_i, best_j = np.unravel_index(np.argmin(scores), scores.shape)
        if np.all(scores == scores[best_i, best_j]): break

        xlo, xhi = lerp(xlo, xhi, best_i/resolution), lerp(xlo, xhi, (best_i+1)/resolution)
        ylo, yhi = lerp(ylo, yhi, best_j/resolution), lerp(ylo, yhi, (best_j+1)/resolution)

    return (lerp(xlo, xhi, 0.5), lerp(ylo, yhi, 0.5))

def open_template(filename):
    # PIL and cv2 disagree on color order
    template = np.uint8(Image.open(filename))[:, :, [2,1,0,3]]
    template[:, :, 3] = np.fmin(template[:, :, 3], 1)
    return template

SPRITE_SIZE=8
ZOOM=3
ZOOMED_SPRITE_SIZE=ZOOM*SPRITE_SIZE
OCCUPATION_THRESHOLD=50
BOARD_ROW_OFFSET_1P = 9
BOARD_COL_OFFSET_1P = 12
BOARD_WIDTH = 8
BOARD_HEIGHT = 16

def open_sprites(sprites):
    arr = np.zeros((len(sprites), SPRITE_SIZE, SPRITE_SIZE, 3), np.uint8)
    for i, sprite in enumerate(sprites):
        # this is not a np.flip, because we are also optionally discarding the alpha channel
        arr[i] = np.uint8(Image.open(sprite.image_name()).convert())[..., [2, 1, 0]]
    return np.repeat(np.repeat(arr, ZOOM, 1), ZOOM, 2)

def vstack(arrs):
    if not arrs: return np.zeros((0,))
    shape = tuple(map(max, zip(*[arr.shape for arr in arrs])))
    def zero_pad(arr):
        dshape = list(map(lambda big, small: (0, big-small), shape, arr.shape))
        dshape[0] = (0, 0)
        return np.pad(arr, dshape)
    return np.concatenate([zero_pad(arr) for arr in arrs])

# Normal slicing:
#     * Inclusive lower bound + exclusive upper bound
#     * Negative bounds are treated as offsets from the end of the array
#     * Out of bounds accesses are errors
# Slicing in this function:
#     * Inclusive lower bound + size
#     * Negative bounds are treated as out of bounds before the beginning of the array
#     * Out of bounds accesses read zeros
def pad_slice(arr, slices):
    # TODO: for efficiency, slice first, pad after
    assert(arr.ndim == len(slices))
    padding = []
    shifted_slices = []
    for i in range(len(slices)):
        arr_sz = arr.shape[i]
        lo, slice_sz = slices[i]
        pad_lo = max(0, -lo)
        pad_hi = max(0, lo+slice_sz-arr_sz)
        padding.append((pad_lo, pad_hi))
        shifted_slices.append(slice(lo+pad_lo,lo+pad_lo+slice_sz))
    #print("shape:", arr.shape, "; slices:", slices, "; computed padding:", padding, "; shifted slices:", shifted_slices)
    return np.pad(arr, padding)[*shifted_slices]

class Position:
    def __init__(self, r, c):
        self.r = r
        self.c = c

    def __add__(self, vec): return Position(self.r + vec.dr, self.c + vec.dc)
    __radd__ = __add__

    def __sub__(self, pos):
        try:
            return Vector(self.r - pos.r, self.c - pos.c)
        except AttributeError:
            return NotImplemented

    def snap(self): return Position(math.floor(self.r), math.floor(self.c))

    def __repr__(self): return f"Position({self.r.__repr__()}, {self.c.__repr__()})"
    def __str__(self): return f"({self.r.__str__()}, {self.c.__str__()})"

class Vector:
    def __init__(self, dr, dc):
        self.dr = dr
        self.dc = dc

    def __add__(self, vec):
        try:
            return Vector(self.dr + vec.dr, self.dc + vec.dc)
        except AttributeError:
            return NotImplemented

    def __sub__(self, vec): return Vector(self.dr - vec.dr, self.dc - vec.dc)
    def __rsub__(self, pos): return Position(pos.r - self.dr, pos.c - self.dc)

    def __truediv__(self, alpha): return Vector(self.dr / alpha, self.dc / alpha)
    def __mul__(self, alpha): return Vector(self.dr * alpha, self.dc * alpha)
    __rmul__ = __mul__

    def __neg__(self): return Vector(-self.dr, -self.dc)

    def __repr__(self): return f"Vector({self.dr.__repr__()}, {self.dc.__repr__()})"
    def __str__(self): return f"({self.dr.__str__()}, {self.dc.__str__()})"

class Extent:
    def __init__(self, sprite_size):
        self._template_size = None
        self._sprite_size = sprite_size

    def set_template_size(self, size):
        if self._template_size is None:
            self._template_size = size
        assert(self._template_size == size)

    def pixel(self): return self._sprite_size/SPRITE_SIZE
    def sprite(self): return self._sprite_size
    def screen(self): return self._template_size * self.pixel()
    def screen_midpoint(self): return self.screen()/2

    def pixel_approx(self): return math.floor(self.pixel())
    def sprite_approx(self): return math.floor(self.sprite())
    def screen_approx(self): return math.floor(self.screen())
    def screen_midpoint_approx(self): return self.screen_approx()//2

class Grid:
    def __init__(self, h, w):
        self.h = Extent(h)
        self.w = Extent(w)
        self._unscaled_template = None
        self._colors = None
        self._mask = None

    def pixel(self): return Vector(self.h.pixel(), self.w.pixel())
    def sprite(self): return Vector(self.h.sprite(), self.w.sprite())
    def screen(self): return Vector(self.h.screen(), self.w.screen())
    def screen_midpoint(self): return Vector(self.h.screen_midpoint(), self.w.screen_midpoint())

    def pixel_approx(self): return Vector(self.h.pixel_approx(), self.w.pixel_approx())
    def sprite_approx(self): return Vector(self.h.sprite_approx(), self.w.sprite_approx())
    def screen_approx(self): return Vector(self.h.screen_approx(), self.w.screen_approx())
    def screen_midpoint_approx(self): return Vector(self.h.screen_midpoint_approx(), self.w.screen_midpoint_approx())

    def unscaled_template(self): return Vector(self._colors.shape[0], self._colors.shape[1])
    def sprite_width(self): return Vector(0, self.w.sprite())
    def sprite_height(self): return Vector(self.h.sprite(), 0)

    def set_unscaled_template(self, template):
        self._unscaled_template = template
        h, w, _ = template.shape
        self.h.set_template_size(h)
        self.w.set_template_size(w)
        template = cv2.resize(template, (self.w.screen_approx(), self.h.screen_approx()), interpolation=cv2.INTER_NEAREST)
        self._colors = template[:, :, :3]
        self._mask = template[:, :, 3]
        return self

    def best_screen_raw(self, frame, origin=Position(0, 0)):
        scores = cv2.matchTemplate(frame, self._colors, cv2.TM_SQDIFF, mask=self._mask)
        dr, dc = np.unravel_index(np.argmin(scores), scores.shape)
        return Screen(origin + Vector(dr, dc), self, scores[dr, dc])

    def best_screen_pad(self, frame):
        padh = self.h.screen_midpoint_approx()
        padw = self.w.screen_midpoint_approx()
        padded = np.pad(frame, ((padh, padh), (padw, padw), (0, 0)))
        return self.best_screen_raw(padded, Position(-padh, -padw))

    def best_screen_like(self, screen, frame):
        if self._unscaled_template is None:
            self.set_unscaled_template(screen.grid._unscaled_template)
        origin = (screen.midpoint_approx() - self.screen_midpoint() - self.sprite() / 2).snap()
        extent = self.screen() + self.sprite()
        padded = pad_slice(frame, ((origin.r, math.ceil(extent.dr)), (origin.c, math.ceil(extent.dc)), (0, 3)))
        return self.best_screen_raw(padded, origin)

class Screen:
    def __init__(self, top_left, grid, score):
        self.top_left = top_left
        self.grid = grid
        self.score = score

    def midpoint_approx(self): return self.top_left + self.grid.screen_midpoint_approx()

    def extract_raw(self, frame, top_left, unscaled_size, scaled_size):
        tl = top_left.snap()
        br = (top_left + unscaled_size).snap() - tl
        unscaled = pad_slice(frame, ((tl.r, br.dr), (tl.c, br.dc), (0, 3)))
        return cv2.resize(unscaled, (scaled_size.dc, scaled_size.dr), interpolation=cv2.INTER_CUBIC)

    def extract_screen(self, frame):
        return self.extract_raw(frame, self.top_left, self.grid.screen(), ZOOM*self.grid.unscaled_template())

    def extract_1p_board(self, frame):
        h = self.grid.sprite_height()
        w = self.grid.sprite_width()
        return self.extract_raw(frame,
            self.top_left + BOARD_COL_OFFSET_1P*w + BOARD_ROW_OFFSET_1P*h,
            BOARD_WIDTH*w + BOARD_HEIGHT*h,
            ZOOMED_SPRITE_SIZE*Vector(BOARD_HEIGHT, BOARD_WIDTH)
            )

    def render(self, frame, origin=Position(0, 0)):
        frame = np.copy(frame)
        color = (0, 255, 0)

        top_left = self.top_left - origin
        bottom_right = top_left + self.grid.screen_approx() - Vector(1, 1)
        cv2.rectangle(frame, (top_left.dc, top_left.dr), (bottom_right.dc, bottom_right.dr), color, 1)

        midpoint = self.midpoint_approx() - origin
        cv2.line(frame, (midpoint.dc-3, midpoint.dr), (midpoint.dc+3, midpoint.dr), color, 1)
        cv2.line(frame, (midpoint.dc, midpoint.dr-3), (midpoint.dc, midpoint.dr+3), color, 1)

        dr_sprite = self.grid.sprite().dr
        dc_sprite = self.grid.sprite().dc
        bottom_right = (Position(0, 0) + top_left + Vector((BOARD_ROW_OFFSET_1P + BOARD_HEIGHT) * dr_sprite, (BOARD_COL_OFFSET_1P + BOARD_WIDTH) * dc_sprite)).snap()
        top_left = (Position(0, 0) + top_left + Vector(BOARD_ROW_OFFSET_1P * dr_sprite, BOARD_COL_OFFSET_1P * dc_sprite)).snap()
        cv2.rectangle(frame, (top_left.c, top_left.r), (bottom_right.c, bottom_right.r), color, 1)

        return frame

# This works in BGR space. I also experimented with working in LAB space, but
# this is a bit faster and the results are only marginally different.
class Posterizer:
    def __init__(self, bgr):
        if bgr.shape[-1] != 3:
            raise ValueError("Expected a 3-channel array (perhaps you accidentally sent an image with transparency?)")
        bgr = np.reshape(bgr, (-1, 3))
        bgrs = set()
        minl = float('inf')
        for i in range(bgr.shape[0]):
            cur_bgr = tuple(bgr[i])
            l = sum(cur_bgr)
            if l > 0 and cur_bgr not in bgrs:
                bgrs.add(cur_bgr)
                minl = min(minl, l)
        self._lthreshold = minl / 2
        # Since we're deciding "black or not" based on _lthreshold, we only
        # compare against the non-black colors in bgr2indexed. But we still
        # want a version that has the zeros for use in bgr2bgr. So we keep
        # around two arrays, one with black and one without.
        #
        # Since we're going to be squaring differences in bgr2indexed, we use
        # int32 to allow negative differences and large squares.
        self._bgrs = np.array(list(bgrs), dtype=np.int32)
        self._zbgrs = np.array([(0, 0, 0)] + list(bgrs), dtype=np.uint8)

    def bgr2indexed(self, bgr):
        return np.where(np.sum(bgr, -1) <= self._lthreshold, 0, 1 + np.argmin(np.sum(np.square(np.expand_dims(bgr, -2) - self._bgrs), -1), -1))

    def indexed2bgr(self, indexes): return self._zbgrs[indexes]
    def bgr2bgr(self, bgr): return self.indexed2bgr(self.bgr2indexed(bgr))

# arr.shape = (num_sprites, sprite_height, sprite_width)
# arr.dtype = uint8
# returns an array with shape (palette_size, num_sprites, sprite_height, sprite_width)
# where palette_size = max(arr)+1
def distance_field(arr):
    n, h, w = arr.shape
    p = 1+np.max(arr)

    a, b = 1 + np.indices((h, w))
    palette_missing = np.fmin(a, b)
    palette_missing = np.fmin(palette_missing, np.flip(palette_missing, 0))
    palette_missing = np.fmin(palette_missing, np.flip(palette_missing, 1))

    stamp = np.sqrt(np.sum(np.square(np.indices((2*h-1, 2*w-1)) - [[[h-1]], [[w-1]]]), 0))

    overscan = np.zeros((p, n, 3*h-2, 3*w-2))
    overscan[:, :, h-1:2*h-1, w-1:2*w-1] = palette_missing

    for sprite in range(n):
        for r in range(h):
            for c in range(w):
                color = arr[sprite, r, c]
                overscan[color, sprite, r:r+2*h-1, c:c+2*w-1] = np.fmin(overscan[color, sprite, r:r+2*h-1, c:c+2*w-1], stamp)

    return np.sum(palette_missing), overscan[:, :, h-1:2*h-1, w-1:2*w-1]

class StrategyColor(Enum):
    BLUE = 0
    RED = 1
    YELLOW = 2

    def __init__(self, i):
        self.image_name = 'bry'[i]

class BoardShape(Enum):
    VIRUS = 0
    WEST_HALF_OF_PILL = 1
    EAST_HALF_OF_PILL = 2
    NORTH_HALF_OF_PILL = 3
    SOUTH_HALF_OF_PILL = 4
    DISCONNECTED = 5
    CLEARING = 6

    def __init__(self, i):
        self.image_name = 'xlr^v*o'[i]
        self.parameters = [None] if i > 4 else [False, True]

class AbstractSprite:
    CONTROLLABLE = set([BoardShape.WEST_HALF_OF_PILL, BoardShape.EAST_HALF_OF_PILL, BoardShape.NORTH_HALF_OF_PILL, BoardShape.SOUTH_HALF_OF_PILL])

    def __init__(self, color=None, shape=None, misc=None):
        self.color = color
        self.shape = shape
        self.misc = misc
        summary = \
            ('k' if color is None else color.image_name) + \
            (' ' if shape is None else shape.image_name)
        if shape is BoardShape.VIRUS:
            summary += '2' if misc else '1'
        elif shape in self.CONTROLLABLE:
            summary += 'c' if misc else ''
        self.summary = summary

    def __eq__(self, other): return self.color == other.color and self.shape == other.shape and self.misc == other.misc
    def __hash__(self): return hash((self.color, self.shape, self.misc))

    def image_name(self): return f'sprites/{self.summary}.png'
    def __str__(self): return self.summary
    def __repr__(self): return f'AbstractSprite(color={self.color}, shape={self.shape}, misc={self.misc})'

all_abstract_sprites = [AbstractSprite()] + [AbstractSprite(color, shape, param) for color in StrategyColor for shape in BoardShape for param in shape.parameters]
abstract_sprite_indices = {sprite: i for i, sprite in enumerate(all_abstract_sprites)}

def enum_ix(color):
    return lambda shape, misc: abstract_sprite_indices[AbstractSprite(color=color, shape=shape, misc=misc)]

def str_ix(color):
    eix = enum_ix(color)
    # Wow. I feel dirty and brilliant simultaneously.
    def f(**kwargs):
        shape_name, misc = next(iter(kwargs.items()))
        return eix(BoardShape[shape_name], misc)
    return f

def transition_weights(temperature):
    FRAMES_PER_RUN = 40*60*60
    PILLS_PER_RUN = 2300
    VIRUSES_PER_RUN = 924
    ROTATIONS_PER_PILL = 3
    DROPS_PER_PILL = 10
    SLIDES_PER_PILL = 4
    MOVES_PER_PILL = ROTATIONS_PER_PILL + DROPS_PER_PILL + SLIDES_PER_PILL
    CLEARS_PER_RUN = 600
    CELLS_PER_CLEAR = 5
    FRAMES_PER_VIRUS_ANIMATION = 8
    MEAN_FRAMES_PER_CLEAR_ANIMATION = 18
    MEAN_VIRUSES_VISIBLE = 3835/132 # assumptions: every virus takes the same time to clear, 0-20, only gameplay (not cutscenes)
    NUM_COLORS = len(StrategyColor)
    NUM_ORIENTATIONS = 2
    SLIDE_DIRECTIONS = 2
    MEAN_PILL_HALVES_VISIBLE = 15
    BOARD_SPACES = 8*16
    FALLING_TRANSITIONS_PER_PAIR = CLEARS_PER_RUN / NUM_COLORS / 2 # dunno lol
    CONNECTED_PILL_HALVES = (BoardShape.WEST_HALF_OF_PILL, BoardShape.NORTH_HALF_OF_PILL, BoardShape.SOUTH_HALF_OF_PILL, BoardShape.EAST_HALF_OF_PILL)
    ALL_PILL_HALVES = (BoardShape.DISCONNECTED,) + CONNECTED_PILL_HALVES

    # We will try to fill ws[tgt, src] with an estimate of many times a src
    # sprite turns into a tgt sprite in a typical 0-20 run. But we'll start
    # with a nonzero everywhere so that we don't rule any transitions out
    # completely -- who knows if somebody's trying to use this on a DM variant
    # or something.
    ws = np.ones((len(all_abstract_sprites), len(all_abstract_sprites)), dtype=np.float64)
    empty_ix = abstract_sprite_indices[AbstractSprite()]

    for color in StrategyColor:
        eix = enum_ix(color)
        six = str_ix(color)

        ws[empty_ix, empty_ix] = (BOARD_SPACES - MEAN_VIRUSES_VISIBLE - MEAN_PILL_HALVES_VISIBLE) * FRAMES_PER_RUN
        ws[six(VIRUS=False), empty_ix] = VIRUSES_PER_RUN / NUM_COLORS / 2
        ws[six(VIRUS=True), empty_ix] = VIRUSES_PER_RUN / NUM_COLORS / 2
        for shape in ALL_PILL_HALVES:
            ws[eix(shape, shape.parameters[0]), empty_ix] = FALLING_TRANSITIONS_PER_PAIR
        for shape in CONNECTED_PILL_HALVES:
            ws[eix(shape, True), empty_ix] = PILLS_PER_RUN * MOVES_PER_PILL / NUM_COLORS / NUM_ORIENTATIONS

        for misc in [False, True]:
            src = six(VIRUS=misc)
            # /2 because half go to the `not misc` variant
            animation_toggles = MEAN_VIRUSES_VISIBLE / NUM_COLORS * FRAMES_PER_RUN / FRAMES_PER_VIRUS_ANIMATION / 2
            ws[src, src] = (FRAMES_PER_VIRUS_ANIMATION - 1) * animation_toggles
            ws[six(VIRUS=not misc), src] = animation_toggles
            ws[six(CLEARING=None), src] = CLEARS_PER_RUN / NUM_COLORS / 2

        for i, shape in enumerate(ALL_PILL_HALVES):
            src = eix(shape, shape.parameters[0])
            ws[six(DISCONNECTED=None), src] = 2 * CLEARS_PER_RUN / NUM_COLORS
            ws[six(CLEARING=None), src] = 2 * CLEARS_PER_RUN / NUM_COLORS
            # intentionally overwrites ws[six(DISCONNECTED=None), src]
            # assignment from above when shape = DISCONNECTED
            ws[src, src] = FRAMES_PER_RUN * MEAN_PILL_HALVES_VISIBLE / NUM_COLORS / len(ALL_PILL_HALVES)

            # when stuff is falling
            ws[empty_ix, src] = FALLING_TRANSITIONS_PER_PAIR
            for shape_above in ALL_PILL_HALVES:
                for color_above in StrategyColor:
                    tgt = abstract_sprite_indices[AbstractSprite(color=color_above, shape=shape_above, misc=shape_above.parameters[0])]
                    ws[tgt, src] += FALLING_TRANSITIONS_PER_PAIR

        for shape in CONNECTED_PILL_HALVES:
            src = eix(shape, True)
            ws[src, src] = (FRAMES_PER_RUN - PILLS_PER_RUN * MOVES_PER_PILL) / NUM_COLORS / len(CONNECTED_PILL_HALVES) * 2
            ws[eix(shape, False), src] = PILLS_PER_RUN / NUM_COLORS / NUM_ORIENTATIONS

        sliding = PILLS_PER_RUN * SLIDES_PER_PILL / NUM_COLORS / NUM_ORIENTATIONS / SLIDE_DIRECTIONS
        dropping = PILLS_PER_RUN * DROPS_PER_PILL / NUM_COLORS / NUM_ORIENTATIONS
        slide_rotating = PILLS_PER_RUN * (SLIDES_PER_PILL + ROTATIONS_PER_PILL) / 10 / NUM_COLORS / NUM_ORIENTATIONS / SLIDE_DIRECTIONS
        rare = PILLS_PER_RUN / NUM_COLORS / NUM_ORIENTATIONS / 100 # rare, but not so rare as using a DM variant
        rotating = PILLS_PER_RUN * ROTATIONS_PER_PILL / NUM_COLORS / NUM_ORIENTATIONS

        src = six(WEST_HALF_OF_PILL=True)
        ws[empty_ix, src] = sliding + dropping
        ws[six(EAST_HALF_OF_PILL=True), src] = sliding
        ws[six(NORTH_HALF_OF_PILL=True), src] = rare
        ws[six(SOUTH_HALF_OF_PILL=True), src] = rotating

        src = six(EAST_HALF_OF_PILL=True)
        ws[empty_ix, src] = sliding + dropping
        ws[six(WEST_HALF_OF_PILL=True), src] = sliding
        ws[six(NORTH_HALF_OF_PILL=True), src] = rare
        ws[six(SOUTH_HALF_OF_PILL=True), src] = slide_rotating

        src = six(NORTH_HALF_OF_PILL=True)
        ws[empty_ix, src] = sliding + sliding + dropping

        src = six(SOUTH_HALF_OF_PILL=True)
        ws[empty_ix, src] = sliding + sliding
        ws[six(WEST_HALF_OF_PILL=True), src] = rotating
        ws[six(EAST_HALF_OF_PILL=True), src] = slide_rotating
        ws[six(NORTH_HALF_OF_PILL=True), src] = dropping

        src = six(CLEARING=None)
        ws[src, src] = CLEARS_PER_RUN * CELLS_PER_CLEAR * MEAN_FRAMES_PER_CLEAR_ANIMATION / NUM_COLORS
        ws[empty_ix, src] = CLEARS_PER_RUN * CELLS_PER_CLEAR / NUM_COLORS

    # normalize so that we have probabilities rather than weights
    ws /= np.sum(ws, 0, keepdims=True)

    # who are you going to believe, your lying eyes or your lying memory?
    np.power(ws, 1/temperature, out=ws)
    ws /= np.sum(ws, 0, keepdims=True)

    return ws

def initial_weights():
    ws = np.ones((len(all_abstract_sprites),))
    ws[abstract_sprite_indices[AbstractSprite()]] = 100
    for color in StrategyColor:
        six = str_ix(color)
        ws[six(VIRUS=False)] = 5
        ws[six(VIRUS=True)] = 5
        ws[six(CLEARING=None)] = 0.1
    return ws/np.sum(ws)

# Arguments:
# transitions: NUM_STATES x NUM_STATES, a transition matrix; first dimension is new state, second dimension is old state
# estimates: ... x NUM_STATES, a distribution on states that's our best guess about the previous time step
# obs_probs: ... x NUM_STATES, some fixed (but potentially unknown) constant times the probability of the current time step's observation for each state
#
# Returns:
# ... x NUM_STATES, a distribution on states that's our best guess about the current time step
def hmm_forward(transitions, estimates, obs_probs):
    estimates = np.squeeze(transitions @ np.expand_dims(estimates, -1), -1) * obs_probs
    return estimates / np.sum(estimates, -1, keepdims=True)

class Step:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        print(self.message, end="", flush=True)
        self.start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
        return self.start_time

    def __exit__(self, *args):
        print("", time.clock_gettime(time.CLOCK_MONOTONIC) - self.start_time, "s")

with Step("Preprocessing reference images"):
    template = open_template("1p-hi-masked.png")
    sprite_images = open_sprites(all_abstract_sprites)
    posterizer = Posterizer(sprite_images)
    max_sprite_distance, sprite_distances = distance_field(posterizer.bgr2indexed(sprite_images))

with Step("Generating transition matrix"):
    transitions = transition_weights(20)

for filename in sys.argv[1:]:
    progress_summary = f' ({filename})' if len(sys.argv) > 2 else ''

    with Step(f"Estimating game screen location{progress_summary}"):
        filename = sys.argv[1] if len(sys.argv) > 1 else "dmhero-short.mp4"
        video_in = cv2.VideoCapture(filename)
        video_fps = video_in.get(cv2.CAP_PROP_FPS)
        video_out = None
        success, frame = video_in.read()
        (h_lo, h_best, h_hi), (w_lo, w_best, w_hi) = estimate_grid_size(frame)
        screen_estimate = Grid(h_best, w_best).set_unscaled_template(template).best_screen_pad(frame)

    with Step(f"Refining game screen location estimate{progress_summary}"):
        score_count = 0
        def score_hw_candidate(h_candidate, w_candidate):
            global score_count
            score_count += 1
            if score_count & 3 == 0: print(".", end="", flush=True)
            return Grid(h_candidate, w_candidate).best_screen_like(screen_estimate, frame).score
        bh, bw = grid_search(score_hw_candidate, h_lo, h_hi, w_lo, w_hi)
        best_screen = Grid(bh, bw).best_screen_like(screen_estimate, frame)

    with Step(f"Processing video{progress_summary}\n") as start_time:
        frame_number = 0
        estimates = initial_weights() # will get broadcast to a larger size shortly
        while success:
            board = best_screen.extract_1p_board(frame)
            poster_index = posterizer.bgr2indexed(board)
            tiles = np.lib.stride_tricks.sliding_window_view(poster_index, (ZOOMED_SPRITE_SIZE, ZOOMED_SPRITE_SIZE))[::ZOOMED_SPRITE_SIZE, ::ZOOMED_SPRITE_SIZE, np.newaxis, ...]
            distances = np.sum(np.choose(tiles, sprite_distances), (3,4))
            raw_sprites = np.argmin(distances, 2)
            old_estimates = estimates
            estimates = hmm_forward(transitions, estimates, max_sprite_distance - distances)
            forward_sprites = np.argmax(estimates, 2)
            frame_components = [board]
            for row in raw_sprites:
                frame_components.append(np.concatenate(sprite_images[row], 1))
            for row in forward_sprites:
                frame_components.append(np.concatenate(sprite_images[row], 1))
            # for r, row in enumerate(distances):
            #     for c, col in enumerate(row):
            #         for i, abstract_sprite in enumerate(all_abstract_sprites):
            #             o = old_estimates[r,c] if len(old_estimates.shape) > 1 else old_estimates
            #             print(f"{frame_number:5} {r:2} {c} {abstract_sprite!s:<4}{o[i]:>8.3f} {max_sprite_distance-distances[r,c,i]:>8.3f} {estimates[r,c,i]:>8.3f} {list(transitions[i]*o)}")
            # print("")

            frame_height = sum(arr.shape[0] for arr in frame_components)
            frame_width = max(arr.shape[1] for arr in frame_components)

            if video_out is None:
                video_height = frame_height
                video_width = frame_width
                video_out = cv2.VideoWriter(filename + "-with-grid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (video_width, video_height))

            if frame_height < video_height:
                print("WARNING: small frame", frame_number, "(expected height", video_height, ", but saw", frame_height, ")")
                frame_components.append(np.zeros((video_height - frame_height, video_width, 3)))
            frame = np.uint8(vstack(frame_components))
            if frame.shape[0] > video_height or frame.shape[1] > video_width:
                print("WARNING: large frame", frame_number, "(expected ", video_width, "x", video_height, ", but saw", frame.shape[1], "x", frame.shape[0], ")")
                frame = frame[:video_height, :video_width]

            video_out.write(frame)
            end_time = time.clock_gettime(time.CLOCK_MONOTONIC)
            print("    frame:", frame_number, "fps:", (frame_number+1)/(end_time-start_time), " "*20, end="\r")

            success, frame = video_in.read()
            frame_number += 1
        print("\n   ", end="")
        if video_out is not None: video_out.release()
