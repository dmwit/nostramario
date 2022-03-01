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

def indicate_match(img, match, tl, br):
    tlx, tly = tl
    brx, bry = br
    color = {'b': (255, 0, 0), 'r': (0, 0, 255), 'y': (0, 255, 255), 'k': (0, 0, 0)}[match[0]]
    lines = {
        'x1': [((0, 0), (1, 1)), ((0, 1), (1, 0))],
        'x2': [((0.2, 0.2), (0.8, 0.8)), ((0.2, 0.8), (0.8, 0.2))],
        'l': [((1, 1), (0, 0.5)), ((0, 0.5), (1, 0))],
        'r': [((0, 1), (1, 0.5)), ((1, 0.5), (0, 0))],
        '*': [((0, 0.5), (0.5, 0)), ((0, 0.5), (0.5, 1)), ((1, 0.5), (0.5, 0)), ((1, 0.5), (0.5, 1))],
        '^': [((0, 1), (0.5, 0)), ((0.5, 0), (1, 1))],
        'v': [((0, 0), (0.5, 1)), ((0.5, 1), (1, 0))],
        }.get(match[1:], [])
    text = match[1] if '0' <= match[1] and match[1] <= '9' else ''

    def lerp(x, y): return (tlx + round((0.6*x+0.2)*(brx-tlx-1)), tly + round((0.6*y+0.2)*(bry-tly-1)))

    for pt1, pt2 in lines:
        img = cv2.line(img, lerp(*pt1), lerp(*pt2), (255, 255, 255), 3)
    for pt1, pt2 in lines:
        img = cv2.line(img, lerp(*pt1), lerp(*pt2), color, 2)
    img = cv2.putText(img, text, lerp(0, 1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 3)
    img = cv2.putText(img, text, lerp(0, 1), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    return img

if __name__ == "__main__":
    #fceux = find_fceux()
    vidcap = cv2.VideoCapture('input.mp4')

    colors = "bry"
    shapes = ["x1", "x2"] + [c for c in "lr*v^"]
    digits = ['k' + str(d) for d in [' '] + list(range(10))]
    templates = nostramario.load_templates(
        (nm, nm + ".png")
        for nm in
            digits +
            [color + shape for color in colors for shape in shapes]
        )

    millis = 1
    #millis = math.floor(1000/vidcap.get(cv2.CAP_PROP_FPS))
    k = cv2.waitKey(millis)
    # q to quit
    while k != 113:
        #img = screenshot_window(fceux)
        #img = cv2.imread('input.png')
        success, img = vidcap.read()
        if not success: break

        processed = numpy.array(img)

        try:
            g = nostramario.learn_grid_from_img(img)
        except: k = cv2.waitKey(millis)
        else:
            for x, y in g.range(img):
                if templates.match(img, g, x, y).shape != (32,):
                    print(x, y, templates.match(img, g, x, y).shape)
                #processed = indicate_match(processed, match, g.ipoint(x, y), g.ipoint(x+1, y+1))
                k = cv2.waitKey(0)
                if k == 113: break
            else:
                #cv2.imwrite('output.png', processed)
                cv2.imshow('processed', processed)
            if k != 113: k = cv2.waitKey(millis)
