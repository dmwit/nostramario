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

def detect_grid(img):
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
    rmax, cmax = np.divmod(np.argmax(folded), folded.shape[-1])
    rmax += 1
    cmax += 1
    rsize = quad_max([rs/(2*r) for r in range(rmax-1,rmax+2)], log_clip(mags[rmax-1:rmax+2,cmax]))
    csize = quad_max([cs/(2*c) for c in range(cmax-1,cmax+2)], log_clip(mags[rmax,cmax-1:cmax+2]))
    rmax_interp = rs/(2*rsize)
    cmax_interp = cs/(2*csize)
    rphase = arr_circle_lerp(angles[:,0], rmax_interp)
    cphase = arr_circle_lerp(angles[0,:], cmax_interp)
    size = np.array([rsize,csize])
    dphase = 0.5
    offset = np.array([rphase/math.pi-dphase, cphase/math.pi-dphase]) * size
    return (size, offset)

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
grid = detect_grid(frame)
tmpl = resize_template(template, grid[0])
scores = score_positions(frame, tmpl)

y, x = np.unravel_index(np.argmin(scores), scores.shape)
h, w, _ = tmpl.shape
x -= w//2
y -= h//2

frame_number = 0
start_time = time.clock_gettime(time.CLOCK_MONOTONIC)

while success:
    screen = cv2.resize(pad_slice(frame, ((y,h), (x,w), (0,3))), (ZOOM*template.shape[1], ZOOM*template.shape[0]), interpolation=cv2.INTER_CUBIC)
    board = screen[9*ZOOMED_SPRITE_SIZE:25*ZOOMED_SPRITE_SIZE,12*ZOOMED_SPRITE_SIZE:20*ZOOMED_SPRITE_SIZE,:]
    poster_index = posterizer.bgr2indexed(board)
    tiles = np.lib.stride_tricks.sliding_window_view(poster_index, (ZOOMED_SPRITE_SIZE, ZOOMED_SPRITE_SIZE))[::ZOOMED_SPRITE_SIZE, ::ZOOMED_SPRITE_SIZE, np.newaxis, ...]
    distances = np.choose(tiles, sprite_distances)
    frame_components = [screen]
    empty_components = []
    white_component = 255*np.ones((1,screen.shape[1],3))
    for r in range(distances.shape[0]):
        for c in range(distances.shape[1]):
            sprite_scores = np.argsort(np.sum(distances[r, c], (1, 2)))
            sprite_component = np.concatenate(sprite_arr[0, 0, sprite_scores], 1)
            orig_tile_component = board[r*ZOOMED_SPRITE_SIZE:(r+1)*ZOOMED_SPRITE_SIZE, c*ZOOMED_SPRITE_SIZE:(c+1)*ZOOMED_SPRITE_SIZE]
            posterized_tile_component = posterizer.indexed2bgr(tiles[r, c, 0])
            new_components = [np.concatenate([orig_tile_component, posterized_tile_component, sprite_component], 1), white_component]
            if sprite_names[sprite_scores[0]] == 'k ':
                empty_components += new_components
            else:
                frame_components += new_components
    frame_components += empty_components

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
