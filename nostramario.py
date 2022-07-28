import cv2
import math
import nostramario
import numpy
import sys
import Xlib.display
import Xlib.ext.composite
import Xlib.X

from nostramario import *

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
    #fceux = find_fceux()
    #vidcap = cv2.VideoCapture('input.mp4')

    with open(DEFAULT_SCENE_TREE_LOCATION) as f:
        _, _, t = parse_scene_tree(f)
    scenes = Scenes(t)
    classifier = torch.load(params.MODEL_PATH)['model']
    classifier.cpu()
    classifier.train(mode=False) # I think it starts this way but let's be defensive
    cache = TemplateCache()

    millis = -1
    #millis = math.floor(1000/vidcap.get(cv2.CAP_PROP_FPS))
    k = cv2.waitKey(millis)
    # q to quit
    i = 0
    while k != 113:
        img = load_random_photo('captures')
        #img = screenshot_window(fceux)
        #img = cv2.imread('input.png')
        #success, img = vidcap.read()
        #if not success: break

        processed = scenes.reconstruct(classifier(torch.tensor(numpy.array([img])))[0]).render(cache)

        i += 1
        cv2.imshow('original', img)
        cv2.imshow('processed', processed)
        print(i)
        k = cv2.waitKey(millis)
