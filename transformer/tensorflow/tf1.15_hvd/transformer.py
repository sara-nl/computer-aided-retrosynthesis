import sys
import os
import numpy as np
from rdkit import Chem
import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm import tqdm
from pprint import pprint
import json

from utils import suppress_stderr, init_horovod
from model import buildNetwork
from options import get_options
from inference import validate
from train import train
from data.preprocessing import VOCAB_SIZE

# suppress INFO, WARNING, and ERROR messages of Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed = 0
tf.random.set_random_seed(seed)
np.random.seed(seed)


def main(opts):
    if opts.horovod:
        init_horovod(opts)

    pprint(vars(opts))

    mdl, mdl_encoder, mdl_decoder = buildNetwork(opts.layers, opts.heads, opts.embedding_size, VOCAB_SIZE,
                                                 opts.key_size, opts.n_hidden, opts)

    if opts.validate is not None:
        mdl.load_weights(opts.model)
        with K.get_session().as_default():
            acc = validate(opts.validate, mdl_encoder, mdl_decoder, opts.temperature, opts.beam, opts)
        sys.exit(0)

    os.makedirs("storage", exist_ok=True)

    print("Training ...")
    if opts.train is not None:
        with open(os.path.join(opts.output_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

        train(mdl, opts)


if __name__ == '__main__':
    opts = get_options()
    main(opts)
