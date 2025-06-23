from PIL import Image
from PIL import ImageDraw
from skimage import color
import cv2
import math
import numpy as np
import pathlib
import skimage
import sys
import time

def log_clip(arr): return np.log(np.maximum(arr,1e-100))
def rescale_255(arr):
    lo = np.min(arr)
    hi = np.max(arr)
    return np.uint8((arr-lo)/(hi-lo)*255.99)
def array_img(arr): return Image.fromarray(rescale_255(arr))
def log_array_img(arr): return Image.fromarray(rescale_255(log_clip(arr)))
def sq(x): return x*x
def lerp(lo, hi, progress): return lo*(1-progress)+hi*progress
def circle_lerp(lo, hi, progress):
    if abs(hi-lo) < math.pi: return lerp(lo, hi, progress)
    return lerp(lo, hi-2*math.pi, progress)
def arr_circle_lerp(arr, ix):
    return circle_lerp(arr[math.floor(ix)], arr[math.ceil(ix)], ix-math.floor(ix))
def quad_max(xs, ys):
    def alpha(y0, x0, x1, x2): return y0/(x0-x1)/(x0-x2)
    x0, x1, x2 = xs
    y0, y1, y2 = ys
    a0, a1, a2 = alpha(y0, x0, x1, x2), alpha(y1, x1, x0, x2), alpha(y2, x2, x0, x1)
    return (x0*a1+x0*a2+x1*a0+x1*a2+x2*a0+x2*a1)/2/(a0+a1+a2)

class Estimate:
    def __init__(self, lo, best, hi):
        self.lo = lo
        self.best = best
        self.hi = hi

    def lrange(self, n):
        for i in range(n):
            frac = i/(n-1)
            yield self.lo*frac + self.hi*(1 - frac)

def division_estimate(num, denom, ddenom):
    return Estimate(num/(denom+ddenom), num/denom, num/(denom-ddenom))

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
    return (division_estimate(rs/2, rmax+1, 1), division_estimate(cs/2, cmax+1, 1))

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

def open_sprites(directory):
    sprite_dict = {file: np.uint8(Image.open(file).convert())[:, :, [2,1,0]] for file in pathlib.Path(directory).iterdir()}
    sprite_arr = np.zeros((1,1,len(sprite_dict),ZOOMED_SPRITE_SIZE,ZOOMED_SPRITE_SIZE,3), np.int32)
    sprite_names = []
    i = 0
    for name, arr in sprite_dict.items():
        sprite_names.append(name.with_suffix("").name)
        sprite_arr[0, 0, i, :, :, :] = np.repeat(np.repeat(arr, ZOOM, 0), ZOOM, 1)
        i += 1
    return sprite_names, sprite_arr

def score_positions(img, template):
    pad0 = template.shape[0]//2
    pad1 = template.shape[1]//2
    padded = np.pad(img, ((pad0, pad0), (pad1, pad1), (0, 0)))
    mask = template[:, :, 3]
    template = template[:, :, :3]
    return cv2.matchTemplate(padded, template, cv2.TM_SQDIFF, mask=mask)

def resize_template(template, grid_size):
    return cv2.resize(template, (math.floor(grid_size[1]/8*template.shape[1]), math.floor(grid_size[0]/8*template.shape[0])), interpolation=cv2.INTER_NEAREST)

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

def id_transform(arr): return arr
def lab_transform(arr): return skimage.color.rgb2lab(np.flip(np.uint8(arr), -1))
def ab_transform(arr): return lab_transform(arr)[..., 1:3]
def phase_magnitude_transform(arr):
    fft = np.fft.fft2(arr, axes=(-3,-2))
    return np.concatenate([np.angle(fft), np.abs(fft)], -1)
def real_imaginary_transform(arr):
    fft = np.fft.fft2(arr, axes=(-3,-2))
    return np.concatenate([np.real(fft), np.imag(fft)], -1)
def phase_transform(arr): return np.angle(np.fft.fft2(arr, axes=(-3,-2)))
def magnitude_transform(arr): return np.abs(np.fft.fft2(arr, axes=(-3,-2)))
def safe_log_transform(arr): return np.log(np.fmax(arr, 1e-10))
def compose_transform(t1, t2): return lambda arr: t1(t2(arr))
all_transforms = [
    (id_transform, "id"),
    (lab_transform, "lab"),
    #(ab_transform, "ab"),
    #(phase_magnitude_transform, "phase+magnitude"),
    #(real_imaginary_transform, "real+imaginary"),
    #(phase_transform, "phase"),
    #(magnitude_transform, "magnitude"),
    #(compose_transform(safe_log_transform, magnitude_transform), "log.magnitude"),
    #(compose_transform(phase_magnitude_transform, lab_transform), "phase+magnitude.lab"),
    #(compose_transform(real_imaginary_transform, lab_transform), "real+imaginary.lab"),
    #(compose_transform(phase_transform, lab_transform), "phase.lab"),
    #(compose_transform(magnitude_transform, lab_transform), "magnitude.lab"),
    #(compose_transform(compose_transform(safe_log_transform, magnitude_transform), lab_transform), "log.magnitude.lab"),
    ]

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

    def pixel(self): return self._sprite_size/8
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
        return self.extract_raw(frame, self.top_left + 12*w + 9*h, 8*w + 16*h, ZOOM*8*Vector(16, 8))

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
        bottom_right = (Position(0, 0) + top_left + Vector(25 * dr_sprite, 20 * dc_sprite)).snap()
        top_left = (Position(0, 0) + top_left + Vector(9 * dr_sprite, 12 * dc_sprite)).snap()
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
            cur_bgr = tuple(bgr[i, :])
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

    def indexed2bgr(self, indexes): return self._zbgrs[indexes, :]
    def bgr2bgr(self, bgr): return self.indexed2bgr(self.bgr2indexed(bgr))

# probably too smart for its own good
def from_grayscale(arr):
    if np.all(0 <= arr) and np.all(arr <= 1):
        arr = 255.99*arr
    if arr.dtype in [np.float16, np.float32, np.float64, np.float128]:
        arr = np.uint8(arr)
    return np.repeat(arr[..., np.newaxis], 3, -1)

def add_to_key(d, k, v):
    try:
        d[k].add(v)
    except KeyError:
        d[k] = set([v])

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

    return overscan[:, :, h-1:2*h-1, w-1:2*w-1]

template = open_template("1p-hi-masked.png")
sprite_names, sprite_arr = open_sprites("sprites")
posterizer = Posterizer(sprite_arr)
sprite_distances = distance_field(posterizer.bgr2indexed(sprite_arr[0, 0]))

filename = sys.argv[1] if len(sys.argv) > 1 else "dmhero-short.mp4"
video_in = cv2.VideoCapture(filename)
video_fps = video_in.get(cv2.CAP_PROP_FPS)
video_out = None
success, frame = video_in.read()
h_estimate, w_estimate = estimate_grid_size(frame)
screen_estimate = Grid(h_estimate.best, w_estimate.best).set_unscaled_template(template).best_screen_pad(frame)

def score_hw_candidate(h_candidate, w_candidate):
    return Grid(h_candidate, w_candidate).best_screen_like(screen_estimate, frame).score
bh, bw = grid_search(score_hw_candidate, h_estimate.lo, h_estimate.hi, w_estimate.lo, w_estimate.hi)
best_screen = Grid(bh, bw).best_screen_like(screen_estimate, frame)

frame_number = 0
start_time = time.clock_gettime(time.CLOCK_MONOTONIC)

while success:
    board = best_screen.extract_1p_board(frame)
    poster_index = posterizer.bgr2indexed(board)
    tiles = np.lib.stride_tricks.sliding_window_view(poster_index, (ZOOMED_SPRITE_SIZE, ZOOMED_SPRITE_SIZE))[::ZOOMED_SPRITE_SIZE, ::ZOOMED_SPRITE_SIZE, np.newaxis, ...]
    sprites = np.argmin(np.sum(np.choose(tiles, sprite_distances), (3,4)), 2)
    frame_components = [board]
    for row in sprites:
        frame_components.append(np.concatenate(sprite_arr[0, 0, row], 1))

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
        frame = frame[:video_height, :video_width, :]

    video_out.write(frame)
    end_time = time.clock_gettime(time.CLOCK_MONOTONIC)
    print("frame:", frame_number, "fps:", frame_number/(end_time-start_time), " "*20, end="\r")

    success, frame = video_in.read()
    frame_number += 1
print("")
if video_out is not None: video_out.release()
