import os
import tensorflow as tf
import tensorflow.keras.backend as K
import time
import horovod.tensorflow as hvd
import random
import math


class suppress_stderr(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR)]
        self.save_fds = [os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 2)

        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def init_horovod(opts):
    """ Run initialisation options"""

    NTRAIN = sum(1 for _ in open(opts.train if opts.train is not None else opts.validate))

    print("Now hvd.init")
    if opts.horovod:
        hvd.init()
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        if opts.cuda:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print("hvd.size() = ", hvd.size())
            print("GPU's", gpus, "with Local Rank", hvd.local_rank())
            print("GPU's", gpus, "with Rank", hvd.rank())

            if gpus:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank() % 4], 'GPU')

        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        random.seed(opts.seed + hvd.rank())

        # opts.lr_factor *= hvd.size()
        opts.warmup = int(opts.warmup / math.sqrt(hvd.size()))

    else:
        random.seed(opts.seed)

    print("Past hvd.init()")
