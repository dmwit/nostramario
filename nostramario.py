from PIL import Image
from PIL import ImageDraw
from skimage import color
import cv2
import math
import numpy as np
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

def draw_grid(img, grid):
    size, offset = grid
    h, w = size
    y, x = offset
    ih, iw, _ = img.shape
    color = (0, 255, 0)

    r = 0
    while (ry := math.floor(y + r*h*2)) < ih:
        if ry >= 0: cv2.line(img, (0, ry), (iw-1, ry), color)
        r += 1

    c = 0
    while (cx := math.floor(x + c*w*2)) < iw:
        if cx >= 0: cv2.line(img, (cx, 0), (cx, ih-1), color)
        c += 1

def draw_outline(img, tmpl, pos):
    y, x = pos
    h, w, _ = tmpl.shape
    x -= w//2
    y -= h//2
    color = (0, 255, 0)
    cv2.line(img, (x, y), (x+w, y), color)
    cv2.line(img, (x+w, y), (x+w, y+h), color)
    cv2.line(img, (x+w, y+h), (x, y+h), color)
    cv2.line(img, (x, y+h), (x, y), color)
    cv2.line(img, (x+w//2, y), (x+w//2, y+h), color)
    cv2.line(img, (x, y+h//2), (x+w, y+h//2), color)

def open_template(filename):
    # PIL and cv2 disagree on color order
    template = np.uint8(Image.open(filename))[:, :, [2,1,0,3]]
    template[:, :, 3] = np.fmin(template[:, :, 3], 1)
    return template

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
    shape = tuple(map(max, *[arr.shape for arr in arrs]))
    def zero_pad(arr):
        dshape = list(map(lambda big, small: (0, big-small), shape, arr.shape))
        dshape[0] = (0, 0)
        return np.pad(arr, dshape)
    return np.concatenate([zero_pad(arr) for arr in arrs])

template = open_template("1p-hi-masked.png")

filename = sys.argv[1] if len(sys.argv) > 1 else "dmhero-short.mp4"
video_in = cv2.VideoCapture(filename)
video_fps = video_in.get(cv2.CAP_PROP_FPS)
video_width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
video_out = None
frame_number = 0
start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
while True:
    success, frame = video_in.read()
    if not success: break
    grid = detect_grid(frame)
    tmpl = resize_template(template, grid[0])
    scores = score_positions(frame, tmpl)
    pos = np.unravel_index(np.argmin(scores), scores.shape)
    #draw_grid(frame, grid)
    cv2.putText(frame, str(frame_number), (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    draw_outline(frame, tmpl, pos)

    # add a white line at the bottom to help notice templates that are bigger than the slack space allows for
    frame_components = [frame, tmpl[:, :, :3], 255*np.ones((1, video_width, 3))]
    frame_height = sum(arr.shape[0] for arr in frame_components)

    if video_out is None:
        video_height = frame_height + tmpl.shape[0]//10
        video_out = cv2.VideoWriter(filename + "-with-grid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (video_width, video_height))

    if frame_height < video_height:
        frame_components.append(np.zeros((video_height - frame_height, video_width, 3)))
    frame = np.uint8(vstack(frame_components))
    if frame.shape[0] > video_height or frame.shape[1] > video_width:
        frame = frame[:video_height, :video_width, :]

    video_out.write(frame)
    end_time = time.clock_gettime(time.CLOCK_MONOTONIC)
    print("frame:", frame_number, "fps:", frame_number/(end_time-start_time), " "*20, end="\r")
    frame_number += 1
print("")
if video_out is not None: video_out.release()
