import cv2
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

if __name__ == "__main__":
    fceux = find_fceux()
    img = screenshot_window(fceux)
    grid = nostramario.Grid \
        ( nostramario.Boundaries(23.95, 22.52)
        , nostramario.Boundaries(23.97,  4.01)
        )
    # img = cv2.imread('input.png')
    # grid = nostramario.Grid \
    #     ( nostramario.Boundaries(41.74, 34.15)
    #     , nostramario.Boundaries(39.23,  2.05)
    #     )

    # grid = nostramario.learn_grid_from_img(img)
    template = nostramario.load_template("progress-box.png")
    matches = numpy.zeros((14, 23))
    for x in range(23):
        for y in range(14):
            matches[y,x] = template.match_single(img, grid, x, y)
    cv2.imshow('grid.png', grid.draw(img))
    cv2.imshow('matches', matches)
    cv2.waitKey(0)
