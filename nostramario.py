import cv2
import math
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

if __name__ == "__main__":
    fceux = find_fceux()

    w, h, _ = screenshot_window(fceux).shape
    scale = math.sqrt(float(w*h)/(256.0*244.0))

    template = cv2.imread("templates/progress-box.png", cv2.IMREAD_UNCHANGED)
    w, h, _ = [round(dim*scale) for dim in template.shape]
    template = cv2.resize(template, (h, w))
    template_colors = template[:,:,:3]
    template_mask = numpy.array(template[:,:,3])

    while cv2.waitKey(1) == -1:
        img = numpy.array(screenshot_window(fceux))
        matches = cv2.matchTemplate(img, template_colors, cv2.TM_CCORR_NORMED, template_mask)
        r, c = cv2.minMaxLoc(matches)[3]
        cv2.imshow('progress box', cv2.rectangle(img, (r, c), (r+h, c+w), (255, 255, 0), 2))
