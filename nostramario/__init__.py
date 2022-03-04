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
    def size(self, pixels): return math.ceil(self.to_coord(pixels))
    def range(self, pixels): return range(self.size(pixels))

    def to_coord(self, pixels): return (pixels-self.b)/self.m
    def floor(self, pixels): return math.floor(self.to_coord(pixels))
    def ceil(self, pixels): return math.ceil(self.to_coord(pixels))
    def round(self, pixels): return round(self.to_coord(pixels))

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

    def __eq__(self, other):
        try: return self is other or (self.m == other.m and self.b == other.b)
        except AttributeError: return NotImplemented

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

    def to_coord(self, x_pixels, y_pixels): return (self.x.to_coord(x_pixels), self.y.to_coord(y_pixels))
    def floor(self, x_pixels, y_pixels): return (self.x.floor(x_pixels), self.y.floor(y_pixels))
    def ceil(self, x_pixels, y_pixels): return (self.x.ceil(x_pixels), self.y.ceil(y_pixels))
    def round(self, x_pixels, y_pixels): return (self.x.round(x_pixels), self.y.round(y_pixels))

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

    def __eq__(self, other):
        try: return self is other or (self.x == other.x and self.y == other.y)
        except AttributeError: return NotImplemented

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
    def __init__(self, labeled_tmpls):
        self._labels, tmpls = list(zip(*labeled_tmpls))
        self._g = None
        self._g_trunc = None
        self._clear_cache()

        self._tmpls = []
        self._masks = []
        for tmpl in tmpls:
            if tmpl.shape[2] > 3:
                self._tmpls.append(numpy.array(tmpl[:,:,:3]))
                self._masks.append(numpy.array(tmpl[:,:,[3]]))
            else:
                self._tmpls.append(tmpl)
                self._masks.append(255*numpy.ones((tmpl.shape[0], tmpl.shape[1], 1), dtype=numpy.uint8))

    def labels(self): return list(self._labels)

    def set_grid(self, g):
        if self._g == g: return
        self._g = g
        self._g_trunc = Grid(Boundaries(g.x.m, 0), Boundaries(g.y.m, 0))
        self._clear_cache()

    def _clear_cache(self):
        self._cache_size = (-1, -1)
        self._cache_stmpls = None
        self._cache_smasks = None
        self._cache_strengths = None
        self._cache_w = None
        self._cache_h = None

    def match(self, img):
        img = self._setup(img)
        return self._strength(self._cache_stmpls-img)/numpy.sqrt(self._cache_strengths*self._strength(img))

    def _setup(self, img):
        tlx_coord, tly_coord = self._g.ceil(0, 0)
        brx_coord, bry_coord = self._g.floor(img.shape[1], img.shape[0])
        tlx_pixel, tly_pixel = self._g.ipoint(tlx_coord, tly_coord)

        # This is a bit subtle. It could be that the pixel size of the grid in
        # self._g and self._g_trunc are off by one compared to one another.
        # Since we're going to use _g_trunc for everything from here on, we
        # need the size to match its specs, but since only _g tells us where to
        # look in the original image we need the absolute top left to match its
        # specs.
        brx_pixel, bry_pixel = (tlx_pixel + self._g_trunc.x.ipoint(brx_coord-tlx_coord)
                               ,tly_pixel + self._g_trunc.y.ipoint(bry_coord-tly_coord)
                               )
        img = img[tly_pixel:bry_pixel, tlx_pixel:brx_pixel, :]
        self._cache_w = brx_coord - tlx_coord
        self._cache_h = bry_coord - tly_coord

        if img.shape != self._cache_size:
            scalings = {}
            h, w, c = img.shape
            n = len(self._tmpls)
            self._cache_size = img.shape
            self._cache_stmpls = numpy.zeros((n, h, w, c), dtype=numpy.int32)
            self._cache_smasks = numpy.zeros((n, h, w, 1), dtype=numpy.int32)
            self._cache_regions = numpy.zeros((self._cache_w, self._cache_h, h, w), dtype=numpy.int32)

            for x_coord in range(self._cache_w):
                for y_coord in range(self._cache_h):
                    single_tlx_pixel, single_tly_pixel = self._g_trunc.ipoint(x_coord, y_coord)
                    single_brx_pixel, single_bry_pixel = self._g_trunc.ipoint(x_coord+1, y_coord+1)

                    x_pixel_range = slice(single_tlx_pixel, single_brx_pixel)
                    y_pixel_range = slice(single_tly_pixel, single_bry_pixel)
                    w = single_brx_pixel - single_tlx_pixel
                    h = single_bry_pixel - single_tly_pixel
                    scale = (w, h)

                    if scale not in scalings:
                        stmpl = numpy.array([cv2.resize(tmpl, scale) for tmpl in self._tmpls], dtype=numpy.int32)
                        smask = numpy.array([cv2.resize(mask, scale) for mask in self._masks], dtype=numpy.uint8)
                        scalings[scale] = (stmpl, smask)

                    stmpl, smask = scalings[scale]
                    self._cache_stmpls[:, y_pixel_range, x_pixel_range, :] = stmpl
                    self._cache_smasks[:, y_pixel_range, x_pixel_range, 0] = smask
                    self._cache_regions[x_coord, y_coord, y_pixel_range, x_pixel_range] = numpy.ones((h, w))

            self._cache_strengths = self._strength(self._cache_stmpls).astype(numpy.float64)

        return img

    def _strength(self, imgs):
        # doing imgs*smasks first implicitly casts imgs from uint8 to int32 if
        # necessary, because smasks is int32
        pixel_strengths = numpy.sum(imgs*(imgs*self._cache_smasks), axis=3)
        out = numpy.zeros((self._cache_w, self._cache_h, pixel_strengths.shape[0]), dtype=numpy.int32)
        for x in range(self._cache_w):
            x_slice = slice(self._g_trunc.x.ipoint(x), self._g_trunc.x.ipoint(x+1))
            for y in range(self._cache_h):
                y_slice = slice(self._g_trunc.y.ipoint(y), self._g_trunc.y.ipoint(y+1))
                out[x,y,:] = numpy.sum(pixel_strengths[:, y_slice, x_slice], axis=(1,2))
        return out

def load_templates(labeled_paths):
    labeled_paths = list(labeled_paths) # if it's a generator, realize it, because we want to iterate over it twice
    imgs = [cv2.imread(os.path.join("templates", path), cv2.IMREAD_UNCHANGED) for _, path in labeled_paths]
    h, w = imgs[0].shape[0:2]

    if [img.shape[0:2] for img in imgs] != [(h, w) for _ in imgs]:
        raise Exception('mismatched template shapes')

    return TemplateLibrary([(label, img) for (label, _), img in zip(labeled_paths, imgs)])
