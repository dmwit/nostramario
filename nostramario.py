import collections
import cv2
import math
import numpy
import sklearn.cluster
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

def interval_goodness(expected, actual): return math.pow(2/3, 10*abs(1-actual/expected))

def vote_for_grid_colors(votes, arr):
    for row in arr:
        start = {row[0]: 0}
        run = {}
        for i in range(row.shape[0]-1):
            label0 = row[i]
            label1 = row[i+1]
            if label0 == label1: continue
            if label0 in start:
                run[label0] = i - start[label0]
            if label1 in start:
                votes[label1] = votes[label1] + interval_goodness(8, run[label1])*interval_goodness(16, i+1-start[label1])
            start[label1] = i+1

def vote_for_grid_position(grid0, grid1, arr):
    cols = collections.defaultdict(lambda:0)
    grids = [grid0, grid1]
    grid_sum = grid0+grid1
    for row in arr:
        end = {}
        for i in range(row.shape[0]-1):
            label0 = row[i]
            label1 = row[i+1]
            if label0 == label1: continue
            if label0 in grids: end[label0] = i
            other_label1 = grid_sum - label1
            if label1 in grids and other_label1 in end:
                col = (end[other_label1] + i + 1) // 2
                cols[col] = cols[col] + 1
    return cols

def parzen_derivative(x, L=4):
    absx = abs(x)
    if absx < L/2:
        return 12*x*(3*absx/(2*L) - 1)/(L*L)
    elif absx < L:
        v = 1-absx/L
        return 6*v*v*(v-1)/x
    else: return 0

def grid_goodness_derivative(votes, chunks, m, b, L=4):
    dm = 0
    db = 0

    for c in range(chunks):
        window_center = m*c+b
        for x in range(math.floor(window_center - L), math.ceil(window_center + L)+1):
            d = parzen_derivative(window_center - x)*votes[x]
            dm = dm + c*d
            db = db + d

    return (dm, db)

def parzen(x, L=4):
    absx = abs(x)
    if absx < L/2:
        v = x/L
        return 1+6*v*v*(abs(v)-1)
    elif absx < L:
        return 2*(1-absx/L)**3
    else: return 0

def grid_goodness(votes, chunks, m, b, L=4):
    g = 0

    for c in range(chunks):
        window_center = m*c+b
        lo = math.floor(window_center - L)
        hi = math.ceil(window_center + L)
        g = g + sum(parzen(window_center - x)*votes[x] for x in range(lo, hi+1))

    return g

def learn_grid_once(votes, chunks, m, b, n=100, learning_rate=1e-6):
    for i in range(n):
        (dm, db) = grid_goodness_derivative(votes, chunks, m, b)
        m = m + learning_rate * dm
        b = b + learning_rate * db
    return (m, b)

def learn_grid_many(votes, chunks, m_lo, m_hi, b_lo, b_hi, n_m=100, n_b=1, n_once=100, learning_rate=1e-6):
    results = []
    for i in range(n_m):
        m0 = m_lo + i * (m_hi - m_lo) / max(1, n_m-1)
        for j in range(n_b):
            b0 = b_lo + j * (b_hi - b_lo) / max(1, n_b-1)
            (m1, b1) = learn_grid_once(votes, chunks, m0, b0, n_once, learning_rate)
            g = grid_goodness(votes, chunks, m1, b1)
            results.append((g, m1, b1))
    results.sort()
    return results

def draw_grid(img, x_m, x_b, y_m, y_b):
    img = numpy.copy(img)
    x_b = math.fmod(x_b, x_m)
    y_b = math.fmod(y_b, y_m)
    max_x = img.shape[1]-1
    max_y = img.shape[0]-1
    cols = math.ceil(max_x/x_m)+1
    rows = math.ceil(max_y/y_m)+1
    for col in range(cols):
        x = round(x_m*col + x_b)
        img = cv2.line(img, (x, 0), (x, max_y), (255, 255, 0))
    for row in range(rows):
        y = round(y_m*row + y_b)
        img = cv2.line(img, (0, y), (max_x, y), (255, 255, 0))
    return img

if __name__ == "__main__":
    fceux = find_fceux()
    img = screenshot_window(fceux)
    # img = cv2.imread('input.png')

    img = cv2.resize(img, (256, 224))

    # palettize the image
    clusterer = sklearn.cluster.KMeans(n_clusters=16)
    colors = numpy.reshape(img, (256*224, 3))
    clusterer.fit(colors)
    pimg = numpy.reshape(clusterer.labels_, (224, 256))

    votes = collections.defaultdict(lambda:0)
    vote_for_grid_colors(votes, pimg)
    vote_for_grid_colors(votes, numpy.transpose(pimg))
    svotes = sorted(votes.items(), key=lambda x:x[1])
    grid0 = svotes[-1][0]
    grid1 = svotes[-2][0]
    cols = vote_for_grid_position(grid0, grid1, pimg)
    rows = vote_for_grid_position(grid0, grid1, numpy.transpose(pimg))

    (x_g, x_m, x_b) = learn_grid_many(cols, 32, 4, 12, 0, 0)[-1]
    (y_g, y_m, y_b) = learn_grid_many(rows, 28, 4, 12, 0, 0)[-1]
    (x_m, x_b) = learn_grid_once(cols, 32, x_m, x_b, 10000)
    (y_m, y_b) = learn_grid_once(rows, 28, y_m, y_b, 10000)
    cv2.imwrite('grid.png', draw_grid(img, x_m, x_b, y_m, y_b))
