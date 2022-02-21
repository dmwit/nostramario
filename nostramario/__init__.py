import cv2
import math
import numpy
import os.path
import statistics

class Boundaries:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    # Adding lines is a bit odd, conceptually. We will typically use this when
    # one of the arguments represents some parameters we're learning, and the
    # other represents partial derivatives of the goodness function for those
    # parameters.
    def __add__(self, other):
        return Boundaries(self.m + other.m, self.b + other.b)

    def __iadd__(self, other):
        self.m += other.m
        self.b += other.b
        return self

    def __mul__(self, other):
        return Boundaries(self.m * other, self.b * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        self.m *= other
        self.b *= other
        return self

    def __call__(self, x): return self.m*x + self.b
    def ipoint(self, x): return round(self(x))
    def size(self, pixels): return math.ceil((pixels-self.b)/self.m)

    def draw(self, img):
        max_coord_this = img.shape[0]-1
        max_coord_other = img.shape[1]-1
        img = img.copy()
        for grid in range(self.size(max_coord_this)):
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
    b.normalize_b()

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

class Template:
    def __init__(self, img, w, h):
        self._img = img
        self._w = w
        self._h = h
        self._cache = {}

    def match_single(self, img, grid, c, r):
        (tlx, tly) = grid.ipoint(c, r)
        (brx, bry) = grid.ipoint(c+self._w, r+self._h)
        w = brx - tlx
        h = bry - tly
        cv2size = (w, h)
        shape = (h, w, self._img.shape[2])
        if cv2size not in self._cache:
            simg = cv2.resize(self._img, cv2size)
            if self._img.shape[2] > 3:
                mask = numpy.array(simg[:,:,3])
                simg = numpy.array(simg[:,:,:3])
            else:
                mask = 255*numpy.ones((h,w,1))
            self._cache[cv2size] = (mask, simg)
        mask, template = self._cache[cv2size]
        return cv2.matchTemplate \
            ( img[tly:bry, tlx:brx, :]
            , template
            , cv2.TM_CCORR_NORMED
            , mask
            )[0,0]

def load_template(path, sprite_size=8):
    img = cv2.imread(os.path.join("templates", path), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[0:2]
    return Template(img, w//sprite_size, h//sprite_size)
