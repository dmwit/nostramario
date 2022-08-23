import argparse
import cv2
import math
import nostramario
import numpy
import os.path as path
import sys
import Xlib.display
import Xlib.ext.composite
import Xlib.X

# TODO: delete once VideoWriter is working
import os
import subprocess

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

def parse_cli():
    parser = argparse.ArgumentParser(description="Use a neural net to try to extract information from a screen of Dr. Mario.")
    parser.add_argument('--mode', dest='mode', action='store', default='random', choices=['random', 'video'], help='Select between random mode, which repeatedly selects a random image from the given directory and displays its reconstruction in a GUI, or video, which reconstructs every frame of a video into another video')
    parser.add_argument('--source', dest='source', action='store', metavar='FILE', help='Where to look for input; defaults to captures for random mode and input.mp4 for video mode')
    parser.add_argument('--target', dest='target', action='store', metavar='FILE', help='Where to store output; ignored in random mode and defaults to <whatever>-reconstruction.mp4 for video mode')
    parser.add_argument('--millis', dest='millis', action='store', type=int, default=-1, metavar='N', help='How often (in milliseconds) to process an image; -1 (the default) to wait for the user to request it, 0 for as fast as possible; ignored in video mode')
    parser.add_argument('--net', dest='model_path', action='store', default=params.MODEL_PATH, metavar='FILE', help=f'The neural net to load (default {params.MODEL_PATH})')
    # TODO: argument for CPU vs. GPU maybe
    # TODO: argument for scene tree file maybe
    args = parser.parse_args()
    if args.source is None: args.source = 'captures' if args.mode == 'random' else 'input.mp4'
    if args.target is None: args.target = args.source + '-reconstruction.mp4'
    return args

class NotVideoWriter:
    dir: str
    base: str
    target: str
    fps: float
    n: int

    def __init__(self, target, fps):
        self.dir, self.base = path.split(target)
        self.dir = self.dir or '.'
        self.target = target
        self.fps = fps
        self.n = 0

    def isOpened(self): return True
    def __filename(self, x): return f'{self.dir}/.{self.base}-{x}.jpg'

    def write(self, img):
        cv2.imwrite(self.__filename(self.n), img)
        self.n += 1

    def release(self):
        # the -vf pad thing is from
        # https://stackoverflow.com/q/20847674/791604
        subprocess.run(['ffmpeg', '-framerate', str(self.fps),  '-i', self.__filename('%d'), '-c:v', 'libx264', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', self.target])
        for i in range(self.n):
            os.remove(self.__filename(i))
        self.n = 0

    def __del__(self):
        for i in range(self.n):
            os.remove(self.__filename(i))

class BoundsEvents:
    def __init__(self):
        self.mode = False

    def __call__(self, event, x, y, flags, param):
        if event != cv2.EVENT_FLAG_LBUTTON: return

        if self.mode:
            self.brx = x
            self.bry = y
        else:
            self.tlx = x
            self.tly = y

        self.mode = not self.mode

    def initialized(self):
        return hasattr(self, 'bry')

    def mapping(self, tlx, tly, brx, bry):
        m11 = (self.brx - self.tlx) / (brx - tlx)
        m22 = (self.bry - self.tly) / (bry - tly)
        return numpy.array(
            [[m11, 0, self.tlx - tlx*m11],
            [0, m22, self.tly - tly*m22]]
            )

    def select(self, vidcap):
        success, img = vidcap.read()
        assert(success)
        cv2.imshow('select playfield corners', img)
        cv2.setMouseCallback('select playfield corners', self)

    def disable_selection(self):
        cv2.destroyWindow('select playfield corners')

if __name__ == "__main__":
    args = parse_cli()
    #fceux = find_fceux()

    with open(DEFAULT_SCENE_TREE_LOCATION) as f:
        _, _, t = parse_scene_tree(f)
    scenes = Scenes(t)
    classifier = torch.load(args.model_path)['model']
    classifier.cpu()
    classifier.train(mode=False) # I think it starts this way but let's be defensive
    cache = TemplateCache()

    if args.mode == 'random':
        k = cv2.waitKey(args.millis)
        # q to quit
        i = 0
        while k != 113:
            img = load_random_photo(args.source)
            #img = screenshot_window(fceux)
            #success, img = vidcap.read()
            #if not success: break

            processed = scenes.reconstruct(classifier(torch.tensor(numpy.array([img])))[0]).render(cache)

            i += 1
            cv2.imshow('original', img)
            cv2.imshow('processed', processed)
            print(i)
            k = cv2.waitKey(millis)
    else:
        vidcap = cv2.VideoCapture(args.source)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        bounds = BoundsEvents()
        bounds.select(vidcap)

        while not bounds.initialized():
            k = cv2.waitKey(30)
            if k == 113: sys.exit(0)
            if k == 83:
                bounds.disable_selection()
                for _ in range(round(5*fps)): vidcap.read()
                bounds.select(vidcap)
            if k == 82:
                bounds.disable_selection()
                for _ in range(round(60*fps)): vidcap.read()
                bounds.select(vidcap)
        bounds.disable_selection()

        # the coordinates of the playfield for 1p games
        m = bounds.mapping(288, 216, 479, 599)
        m_inv = cv2.invertAffineTransform(m)
        tl = tuple(numpy.matmul(m, [0, 0, 1]).astype(int))
        br = tuple(numpy.matmul(m, [768, 672, 1]).astype(int))

        # restart from the beginning
        vidcap = cv2.VideoCapture(args.source)
        # TODO: can't get VideoWriter working, why not?
        # vidout = cv2.VideoWriter(args.target, cv2.VideoWriter.fourcc(*'MPG4'), fps, (768, 672))
        vidout = NotVideoWriter(args.target, fps)
        assert(vidout.isOpened())
        n = 1

        try:
            while True:
                success, img = vidcap.read()
                if not success: break
                net_input = cv2.warpAffine(img, m, (768, 672), None, cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
                net_output = scenes.reconstruct(classifier(torch.tensor(numpy.array([net_input])))[0]).render(cache)

                h, w, _ = img.shape
                net_output = cv2.resize(net_output, (round(h*768/672), h), None, 0, 0, cv2.INTER_NEAREST)
                img = cv2.rectangle(img, tl, br, (255, 255, 0), 2)

                frame = numpy.concatenate((img, net_output), axis=1)
                vidout.write(frame)

                if round(math.log(n, 1.1)) < round(math.log(n+1, 1.1)):
                    print(f'frame {n}')
                n += 1
                cv2.imshow('current frame', frame)
                if cv2.waitKey(1) == 113: raise KeyboardInterrupt

            vidout.release()
        except KeyboardInterrupt:
            del vidout
