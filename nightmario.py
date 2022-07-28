import cv2
import importlib
import multiprocessing
import numpy
import random
import tensorboardX as tx
import torch

from os import urandom
from torch import nn

from nostramario import *

def generate_examples_forever(q, scenes):
    random.seed(urandom(2048))
    cache = TemplateCache()
    # The pytorch documentation on multiprocessing has all kinds of caveats and
    # tricks that you need to pay attention to if you want to share GPU tensors
    # between processes. Let's just keep it simple and toss everything on the
    # CPU, accept the communication cost, and copy to the GPU in the consumer
    # process. This way it's hard to go wrong; and it seems unlikely that the
    # communication costs are a serious consideration for what we're doing.
    dev = torch.device('cpu')
    while True: q.put(scenes.select().render(cache, dev))

def xe_loss_single(probs, target):
    return nn.functional.cross_entropy(torch.unsqueeze(probs, 0), torch.unsqueeze(target, 0), label_smoothing=0.01)

def losses(tensor, golden, device):
    ls = torch.zeros((len(golden),), dtype=torch.float, device=device)
    for i, example in enumerate(golden):
        ls[i] = xe_loss_single(tensor[i, example.onehot_indices], example.classification[example.onehot_indices])
        for r in example.onehot_ranges:
            ls[i] += xe_loss_single(tensor[i, r], example.classification[r])
    return ls

def loss(tensor, golden, device):
    return torch.sum(losses(tensor, golden, device))/max(len(golden), 1)

def log_params(n):
    log.add_scalar('hyperparameters/learning rate', params.LEARNING_RATE, n)
    log.add_scalar('hyperparameters/momentum', params.MOMENTUM, n)
    log.add_scalar('hyperparameters/weight decay', params.WEIGHT_DECAY, n)
    log.add_scalar('hyperparameters/steps per batch', params.STEPS_PER_BATCH, n)
    log.add_scalar('hyperparameters/examples per mix', params.EXAMPLES_PER_MIX, n)

with open(DEFAULT_SCENE_TREE_LOCATION) as f:
    _, _, t = parse_scene_tree(f)
scenes = Scenes(t)
# We can't increase (or decrease) the max queue size later if the number of
# training examples per batch changes. So just pick some bound that's a bit
# bigger than the current number of examples and hope it's big enough.
q = multiprocessing.Queue(16*max(1, params.EXAMPLES_PER_MIX*params.MIXES_PER_BATCH))
# cpu_count-1 to leave a CPU for non-nightmario uses lmao
for _ in range(max(1, multiprocessing.cpu_count() - 1)):
    multiprocessing.Process(target=generate_examples_forever, args=(q, scenes), daemon=True).start()

dev = torch.device('cuda')
cache = TemplateCache()
log = tx.SummaryWriter()
try:
    data = torch.load(params.MODEL_PATH)
    c = data['model']
    n = data['training step']
    log_params(n)
except FileNotFoundError:
    c = Classifier(t, dev)
    # initialize the fully-connected layer
    with torch.no_grad(): c(torch.tensor(numpy.array([scenes.select().render(cache, dev).filtered_image]), device=dev))
    n = 0
opt = torch.optim.SGD(c.parameters(), params.LEARNING_RATE, momentum=params.MOMENTUM, weight_decay=params.WEIGHT_DECAY)

# while True:
#     scene = scenes.select()
#     selection = scene.render(cache, dev)
#     cv2.imshow('clean', selection.clean_image)
#     cv2.imshow('filtered', selection.filtered_image)
#     cv2.imshow('reconstructed', scenes.reconstruct(selection.classification).render(cache))
#     if cv2.waitKey(100) == 113:
#         break

# This bit is done single-threaded. In the long run, the multi-threaded
# approach produces the right distribution, but in the short term it may
# briefly be biased towards scenes that render more quickly.
test = []
for _ in range(params.TEST_EXAMPLES):
    test.append(scenes.select().render(cache, dev))
test_tensor = torch.tensor(numpy.array([x.filtered_image for x in test]), device=dev)

min_loss = None
while(True):
    if n % params.STEPS_PER_PARAMETER_RELOAD < params.STEPS_PER_BATCH:
        importlib.reload(params)
        for g in opt.param_groups:
            g['lr'] = params.LEARNING_RATE
            g['momentum'] = params.MOMENTUM
            g['weight_decay'] = params.WEIGHT_DECAY
        log_params(n)

    train = []
    for _ in range(params.EXAMPLES_PER_MIX*params.MIXES_PER_BATCH):
        train.append(q.get().to(dev))
    train = [merge_examples(train[i::params.MIXES_PER_BATCH]) for i in range(params.MIXES_PER_BATCH)]
    train_tensor = torch.tensor(numpy.array([x.filtered_image for x in train]), device=dev)

    c.train(mode=True)
    for step in range(params.STEPS_PER_BATCH):
        opt.zero_grad()
        log.add_scalar('loss/training', l := loss(c(train_tensor), train, dev), n)
        if step == 0: log.add_scalar('loss/training/batch start', l, n)
        if step == params.STEPS_PER_BATCH-1: log.add_scalar('loss/training/batch end', l, n)
        l.backward()
        opt.step()
        n += 1
    c.train(mode=False)

    with torch.no_grad():
        classifications = c(test_tensor)
        ls = losses(classifications, test, dev).to('cpu')
        test_loss = torch.sum(ls)/ls.shape[0]
        log.add_scalar('loss/test', test_loss, n)

        if min_loss is None: min_loss = test_loss
        elif test_loss < min_loss:
            min_loss = test_loss
            torch.save({'model': c, 'training step': n}, params.MODEL_PATH)
            log.add_scalar('loss/test/min', test_loss, n)

        if params.STEPS_PER_SAVE > 0 and n % params.STEPS_PER_SAVE < params.STEPS_PER_BATCH:
            torch.save({'model': c, 'training step': n}, params.MODEL_PATH)

        if n % params.STEPS_PER_RECONSTRUCTION < params.STEPS_PER_BATCH:
            classifications = classifications.to('cpu')
            _, indices = torch.sort(ls)
            clean_h, clean_w, _ = test[0].clean_image.shape
            filtered_h, filtered_w, _ = test[0].filtered_image.shape
            h = clean_h + filtered_h
            w = max(2*clean_w, filtered_w)

            reconstructions = numpy.zeros((15, h, w, 3), dtype=numpy.uint8)
            for i in [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]:
                reconstructions[i, :clean_h, :clean_w, :] = test[indices[i]].clean_image
                reconstructions[i, :clean_h, clean_w:2*clean_w, :] = scenes.reconstruct(classifications[indices[i]]).render(cache)
                reconstructions[i, clean_h:, :filtered_w, :] = test[indices[i]].filtered_image
            for i, j in enumerate(random.sample(list(range(5, indices.shape[0]-5)), 5)):
                reconstructions[i+5, :clean_h, :clean_w, :] = test[indices[j]].clean_image
                reconstructions[i+5, :clean_h, clean_w:2*clean_w, :] = scenes.reconstruct(classifications[indices[j]]).render(cache)
                reconstructions[i+5, clean_h:, :filtered_w, :] = test[indices[j]].filtered_image
            # OpenCV uses BGR by default, tensorboardX uses RGB by default
            reconstructions = numpy.flip(reconstructions, 3)
            log.add_images('reconstructions/good', reconstructions[:5], n, dataformats='NHWC')
            log.add_images('reconstructions/random', reconstructions[5:10], n, dataformats='NHWC')
            log.add_images('reconstructions/bad', reconstructions[10:], n, dataformats='NHWC')
