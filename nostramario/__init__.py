import cv2
import math
import numpy
import os.path
import statistics

class Boundaries:
    def __init__(self, m, b):
        self.m = m
        self.b = b
        self.normalize_b()

    def __call__(self, x): return self.m*x + self.b
    def ipoint(self, x): return round(self(x))
    def size(self, pixels): return math.ceil((pixels-self.b)/self.m)
    def range(self, pixels): return range(self.size(pixels))

    def draw(self, img):
        max_coord_this = img.shape[0]-1
        max_coord_other = img.shape[1]-1
        img = img.copy()
        for grid in self.range(max_coord_this):
            v = self.ipoint(grid)
            cv2.line(img, (0, v), (max_coord_other, v), (255, 255, 0))
        return img

    def normalize_b(self):
        self.b -= self.m*math.floor(self.b/self.m)

    def __str__(self):
        return 'y = {:.2f}*x + {:.2f}'.format(self.m, self.b)

    def __repr__(self):
        return 'Boundaries({}, {})'.format(repr(self.m), repr(self.b))

class Grid:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, x, y): return (self.x(x), self.y(y))
    def ipoint(self, x, y): return (self.x.ipoint(x), self.y.ipoint(y))
    def size(self, x, y): return (self.x.size(x), self.y.size(y))
    def range(self, img_or_shape):
        try:
            h, w = img_or_shape.shape[0:2]
        except AttributeError:
            w, h = img_or_shape
        return ((x, y) for x in self.x.range(w) for y in self.y.range(h))

    def draw(self, img):
        img = self.x.draw(numpy.transpose(img, (1,0,2)))
        img = self.y.draw(numpy.transpose(img, (1,0,2)))
        return img

    def __str__(self):
        return 'x = {:.2f}*col + {:.2f}; y = {:.2f}*row + {:.2f}'.format(
                self.x.m,
                self.x.b,
                self.y.m,
                self.y.b)

    def __repr__(self):
        return 'Grid({}, {})'.format(repr(self.x), repr(self.y))

def is_peak(hist, min_peak, i):
    return hist[i] > min_peak and (i == 0 or i == hist.size-1 or (hist[i] >= hist[i-1] and hist[i] >= hist[i+1]))

def find_squares(img):
    h, w, _ = img.shape
    kernel_size = 2*round((h+w)/600)+1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # First, we try to find the color black and whatever the second darkest
    # color is, so we can put the threshold between those two. We'll say that
    # any sufficiently high "peaks" (local maxima, including endpoints) qualify
    # as a color.
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

    # Now do a simple thresholding and find contours. Simple thresholding
    # should be okay because I don't think lighting plays a part in any of the
    # video streams we care about for now. If that changes in the future we
    # will need to revisit this.
    _, bw = cv2.threshold(gray, math.ceil((first_peak + second_peak)/2), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Pick out the contours that are "square-like". They're "square-like" if
    # they almost fill their axis-aligned bounding box (we don't handle
    # rotations here for now) and if their aspect ratio is pretty close to 1.
    # But we let the aspect ratio be pretty lax, because some people play with
    # settings that make the boxes have a more different aspect ratio.
    squares = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        rx, ry, rw, rh = rect
        area = cv2.contourArea(contour)
        if 0 in [rw, rh, area]: continue
        if rw*rh/area < 1.3 and max(rw,rh)/min(rw,rh) < 1.4 and rx>0 and ry>0 and rx+rw<=w and ry+rh<=h:
            squares.append(rect)

    return squares

def learn_boundaries_from_intervals(xws, threshold=0.2, rates=[0.1, 0.1, 0.1]):
    w = statistics.mode(w for _, w in xws)
    lo = w*(1-threshold)
    hi = w*(1+threshold)
    xs = [x for x, w in xws if w >= lo and w <= hi]
    ws = [w for x, w in xws if w >= lo and w <= hi]
    xs.sort()
    b = Boundaries(statistics.mean(ws), statistics.mode(xs))

    for rate in rates:
        # learn m
        for x in xs:
            dx = x-b.b
            i = round(dx/b.m)
            if i!= 0: b.m += rate*(dx/i-b.m)

        # learn b
        dxb = 0
        for x in xs:
            dxb += x-b(round((x-b.b)/b.m))
        b.b += dxb/len(xs)

    return b

def learn_grid_from_img(img):
    squares = find_squares(img)
    return Grid(
        learn_boundaries_from_intervals([(x, w) for x, _, w, _ in squares]),
        learn_boundaries_from_intervals([(y, h) for _, y, _, h in squares])
        )

class TemplateLibrary:
    def __init__(self, labeled_tmpls, w, h):
        self._labels, self._tmpls = list(zip(*labeled_tmpls))
        self._w = w
        self._h = h
        self._cache = {}

    def labels(): return list(self._labels)

    def match(self, img, grid, c, r):
        if not self._tmpls: return []

        (tlx, tly) = grid.ipoint(c, r)
        (brx, bry) = grid.ipoint(c+self._w, r+self._h)
        # we want to resize the template to the grid size, not the size of the
        # slice of the image we get, so it's important we use this computation
        # rather than asking for the shape of the subimage
        w = brx - tlx
        h = bry - tly
        cv2size = (w, h)
        img = img[max(0,tly):bry, max(0,tlx):brx, :]
        if 0 in img.shape: return [0 for _ in self._tmpls]

        if cv2size not in self._cache:
            stmpls = []
            masks = []
            match_everywhere = 255*numpy.ones((h,w))

            for tmpl in self._tmpls:
                stmpl = cv2.resize(tmpl, cv2size)
                if stmpl.shape[2] > 3:
                    masks.append(numpy.array(stmpl[:,:,3]))
                    stmpls.append(numpy.array(stmpl[:,:,:3]))
                else:
                    masks.append(match_everywhere)
                    stmpls.append(stmpl)

            mask = numpy.concatenate(masks)
            stmpl = numpy.concatenate(stmpls)
            self._cache[cv2size] = (mask, stmpl)
        else: mask, stmpl = self._cache[cv2size]

        return cv2.matchTemplate(img, stmpl, cv2.TM_SQDIFF_NORMED, mask)[0::h,0]

def load_templates(labeled_paths, sprite_size=8):
    labeled_paths = list(labeled_paths) # if it's a generator, realize it, because we want to iterate over it twice
    imgs = [cv2.imread(os.path.join("templates", path), cv2.IMREAD_UNCHANGED) for _, path in labeled_paths]
    h, w = imgs[0].shape[0:2]

    if w % sprite_size != 0 or h % sprite_size != 0:
        raise Exception('templates are not a nice round multiple of sprite size')

    if [img.shape[0:2] for img in imgs] != [(h, w) for _ in imgs]:
        raise Exception('mismatched template shapes')

    return TemplateLibrary([(label, img) for (label, _), img in zip(labeled_paths, imgs)], w//sprite_size, h//sprite_size)
