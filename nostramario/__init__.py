import collections
import cv2
import math
import numpy
import os.path
import sklearn.cluster

def interval_goodness(expected, actual):
    return math.pow(2/3, 10*abs(1-actual/expected))

def transitions(row):
    for i in range(row.shape[0]-1):
        label0 = row[i]
        label1 = row[i+1]
        if label0 == label1: continue
        yield (i, label0, label1)

def vote_for_grid_colors(votes, arr):
    for row in arr:
        start = {row[0]: 0}
        run = {}
        for i, label0, label1 in transitions(row):
            if label0 in start:
                run[label0] = i - start[label0]
            if label1 in start:
                votes[label1] = votes[label1] + interval_goodness(8, run[label1])*interval_goodness(16, i+1-start[label1])
            start[label1] = i+1

def vote_for_grid_position(arr, grid0, grid1):
    cols = collections.defaultdict(lambda:0)
    grids = [grid0, grid1]
    grid_sum = grid0+grid1
    for row in arr:
        end = {}
        for i, label0, label1 in transitions(row):
            if label0 in grids: end[label0] = i
            other_label1 = grid_sum - label1
            if label1 in grids and other_label1 in end:
                col = (end[other_label1] + i + 1) / 2
                cols[col] += 1
    return cols

def half_integers_around(x, L):
    return map(lambda x:x/2, range(2*math.floor(x - L), 2*math.ceil(x + L)+1))

def parzen(x, L=4):
    absx = abs(x)
    if absx < L/2:
        v = x/L
        return 1+6*v*v*(abs(v)-1)
    elif absx < L:
        return 2*(1-absx/L)**3
    else: return 0

def parzen_derivative(x, L=4):
    absx = abs(x)
    if absx < L/2:
        return 12*x*(1.5*absx/L - 1)/(L*L)
    elif absx < L:
        v = 1-absx/L
        return 6*v*v*(v-1)/x
    else: return 0

class Params:
    def __init__(self, lo, hi, n):
        self.bounds = Boundaries((hi - lo) / max(1, n-1), lo)
        self.n = n

    def lerp(self):
        return (self.bounds(i) for i in range(self.n))

    def __str__(self):
        return '{} points along {}'.format(self.n, self.bounds)

    def __repr__(self):
        return 'Params({}, {}, {})'.format(repr(self.bounds(0)), repr(self.bounds(self.n-1)), repr(self.n))

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

    def goodness(self, votes, chunks, L=4):
        return sum(parzen(window_center - x)*votes[x]
                   for c in range(chunks)
                   for window_center in [self(c)]
                   for x in half_integers_around(window_center, L)
                  )

    def goodness_derivative(self, votes, chunks, L=4):
        return sum( (Boundaries(c*(d := parzen_derivative(window_center - x)*votes[x]), d)
                     for c in range(chunks)
                     for window_center in [self(c)]
                     for x in half_integers_around(window_center, L)
                    )
                  , start = Boundaries(0, 0)
                  )

    def ascend_gradient(self, votes, chunks, n=100, L=4, learning_rate=1e-6):
        for i in range(n):
            self += learning_rate * self.goodness_derivative(votes, chunks, L)
        return self

    def learn(self, votes, chunks, n=100, L=4, learning_rate=1e-6):
        return max \
            ( (Boundaries(m, b).ascend_gradient(votes, chunks, n, L, learning_rate)
               for m in self.m.lerp()
               for b in self.b.lerp()
              )
            , key=lambda x:x.goodness(votes, chunks, L)
            )

    def learn_from_img(self, img, grid0, grid1, chunks, n=100, L=4, learning_rate=1e-6):
        votes = vote_for_grid_position(img, grid0, grid1)
        bounds = self.learn(votes, chunks, n, L, learning_rate).ascend_gradient(votes, chunks, 10000, L, learning_rate)
        bounds.b %= bounds.m
        return bounds

    def size(self, pixels): return math.ceil((pixels-self.b)/self.m)

    def draw(self, img):
        max_coord_this = img.shape[0]-1
        max_coord_other = img.shape[1]-1
        img = img.copy()
        for grid in range(self.size(max_coord_this)):
            # TODO: why is this +1 needed?
            v = round(self(grid))+1
            cv2.line(img, (0, v), (max_coord_other, v), (255, 255, 0))
        return img

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

def learn_grid_from_img(img):
    simg = cv2.resize(img, (256, 224))

    # palettize the image
    clusterer = sklearn.cluster.KMeans(n_clusters=16)
    colors = numpy.reshape(simg, (256*224, 3))
    clusterer.fit(colors)
    pimg = numpy.reshape(clusterer.labels_, (224, 256))

    votes = collections.defaultdict(lambda:0)
    vote_for_grid_colors(votes, pimg)
    vote_for_grid_colors(votes, numpy.transpose(pimg))
    svotes = sorted(votes.items(), key=lambda x:x[1])
    grid0 = svotes[-1][0]
    grid1 = svotes[-2][0]

    params = Boundaries(Params(4, 12, 100), Params(0, 0, 1))
    x_scale = img.shape[1] / pimg.shape[1]
    y_scale = img.shape[0] / pimg.shape[0]
    return Grid \
        ( x_scale * params.learn_from_img(pimg, grid0, grid1, 32)
        , y_scale * params.learn_from_img(numpy.transpose(pimg), grid0, grid1, 28)
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