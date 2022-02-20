import cv2
import math
import nostramario
import numpy
import sys
import Xlib.display
import Xlib.ext.composite
import Xlib.X

def find_fceux(dpy=None):
    if dpy is None: dpy = Xlib.display.Display()

    for i in range(dpy.screen_count()):
        windows = [dpy.screen(i).root]
        j = 0
        while j < len(windows):
            windows = windows + windows[j].query_tree().children
            j = j+1
    fceuxs = [w for w in windows if w.get_wm_class() == ('fceux', 'fceux')]

    if len(fceuxs) == 1:
        Xlib.ext.composite.redirect_window(fceuxs[0], Xlib.ext.composite.RedirectAutomatic)
        return fceuxs[0]
    if len(fceuxs) == 0:
        print('Could not find an fceux window, exiting')
        sys.exit(1)
    print('Found multiple fceux windows, with these IDs:')
    for fceux in fceuxs: print((fceux.id, fceux.query_tree().children))
    sys.exit(2)

def screenshot_window(win):
    pixmap = Xlib.ext.composite.name_window_pixmap(win)
    geometry = pixmap.get_geometry()
    x = geometry.border_width
    y = x
    w = geometry.width - 2*geometry.border_width
    h = geometry.height - 2*geometry.border_width
    img = pixmap.get_image(x, y, w, h, Xlib.X.ZPixmap, 0xffffffff).data
    return numpy.frombuffer(img, dtype=numpy.uint8).reshape(h, w, 4)[:, :, :3]

def is_peak(hist, min_peak, i):
    return hist[i] > min_peak and (i == 0 or i == hist.size-1 or (hist[i] >= hist[i-1] and hist[i] >= hist[i+1]))

if __name__ == "__main__":
    fceux = find_fceux()

    while True:
        img = screenshot_window(fceux)
        #img = cv2.imread('input.png')
        h, w, _ = img.shape
        kernel_size = 2*round((h+w)/600)+1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

        _, bw = cv2.threshold(gray, math.ceil((first_peak + second_peak)/2), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        with_contours = numpy.array(img)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if 0 in [w, h, area]: continue
            if w*h/area < 1.3 and max(w,h)/min(w,h) < 1.4:
                color = (255, 255, 0)
                if len(contour) > 100:
                    color = (0, 255, 255)
                    with_contours = cv2.rectangle(with_contours, (x, y), (x+w-1, y+h-1), (255, 0, 255), 1)
                with_contours = cv2.drawContours(with_contours, [contour], 0, color)

        cv2.imshow('original', img)
        cv2.imshow('processed', with_contours)
        if cv2.waitKey(-1) != 32: break
