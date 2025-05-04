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
    # assume we aren't zoomed way into some small part of the 256x240 NES output, so we should ignore very low-frequency signals
    folded[:rs//240+1,:] = -float('inf')
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

filename = sys.argv[1] if len(sys.argv) > 1 else "dmhero-short.mp4"
video_in = cv2.VideoCapture(filename)
video_fps = video_in.get(cv2.CAP_PROP_FPS)
video_width = video_in.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video_in.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_out = cv2.VideoWriter(filename + "-with-grid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (int(video_width), int(video_height)))
frame_number = 0
start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
while True:
    success, frame = video_in.read()
    if not success: break
    draw_grid(frame, detect_grid(frame))
    video_out.write(frame)
    end_time = time.clock_gettime(time.CLOCK_MONOTONIC)
    print("frame:", frame_number, "fps:", frame_number/(end_time-start_time), " "*20, end="\r")
    frame_number += 1
print("")
video_out.release()
