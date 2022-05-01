This is the beginnings of a package for applying computer vision techniques to
NES Dr. Mario. This version is based on ResNet-style machine learning. There
are two programs:

* **nostramario** is currently defunct, but eventually will be a program for
  taking in images of Dr. Mario and spitting out information about the image.
  (Nostradamus was a visionary, see?)
* **nightmario** trains the neural net. (It dreams up twisted, corrupted Dr.
  Mario images to feed to the net; otherwise it's bog-standard training.)

To run **nightmario**, you will need:

* [Python 3](https://www.python.org/)
* [pytorch](https://pytorch.org/)
* [OpenCV](https://opencv.org/) (if the "easy way" below doesn't work, see also the [official installation instructions](https://docs.opencv.org/4.x/da/df6/tutorial_py_table_of_contents_setup.html))
* [tensorboardX](https://github.com/lanpa/tensorboardX)
* [ImageNet](https://www.image-net.org/) or some similar large collection of non-Dr. Mario images
* [tensorboard](https://www.tensorflow.org/tensorboard) (optional; can be used to monitor progress)

I'm gonna assume you can work out how to install Python and pip yourself. I think the Windows installer hides pip behind one of the advanced options, so make sure to poke around and enable it. Once you have, it *should* be that installing the libraries above is as simple as running

    pip3 install torch torchvision torchaudio opencv-python tensorboardX

from a command prompt. In some installations, `pip3` is called `pip`.

At this point, you may want to double-check that pytorch has access to your GPU. In a Python interpreter:

    >>> import torch
    >>> torch.cuda.is_available()

If it prints `True` you're good to go. You should use `cuda.is_available()` even if you are using a different GPU access mode than CUDA (e.g. RocM).

The easiest way to get some ImageNet data is to get the public subset available from the [ILSVRC](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=imagenet_object_localization_patched2019.tar.gz). You will need to create an account on kaggle (it will prompt you to do so at the right moment). It's largish -- 155GiB packed, 161GiB unpacked -- so use your slow but spacious drive if you've got one. On Windows, I think 7zip can handle .tar.gz files; on Linux, you can expand it with:

    tar xf imagenet_object_localization_patched2019.tar.gz

You will want to muck about with some of the settings in `params.py`; notably `PHOTOS_DIR` and `MIXES_PER_BATCH`. For example, if you unpacked the ILSVRC data set into `C:\ILSVRC`, you might write:

    PHOTOS_DIR = "C:\\ILSVRC\\Data\\CLS-LOC\\train"

The `MIXES_PER_BATCH` setting controls how much GPU memory **nightmario** uses, so experiment until you find the highest value your GPU can handle. For a sense of scale: my card has 16GiB and can handle `MIXES_PER_BATCH=60`. Generating the test data happens at the start of the program and takes a while, so you might want to turn `TEST_EXAMPLES` down to `1` while doing this fiddling.

At this point, you should be able to run `python3 nightmario.py` from this directory to start training. In some installations, `python3` is named `python`. If you install and run tensorboard, then you can visit `localhost:6006` in a browser to see some pretty graphs. On my computer (3.6GHz i7, Radeon RX 6800) it takes about 15 minutes for the first data points to show up, then additional points appear every 2 minutes.
