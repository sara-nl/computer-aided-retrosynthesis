from pprint import pprint
from tensorflow.keras import backend as K
import sys
from rdkit import Chem

from options import get_options
from model import buildNetwork
from inference import gen_beam
from data.preprocessing import VOCAB_SIZE


def from_and_to_smiles(smiles: str):
    smiles = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(smiles)
    return smiles


SQUALENE = 'CC(=CCC/C(=C/CC/C(=C/CC\C=C(/C)\CC\C=C(/C)\CCC=C(C)C)/C)/C)C'
OTHER = 'C[C@@H]1C[C@@H]2CC[C@H]3C(=C)C[C@@H](O3)CC[C@]45C[C@@H]6[C@H](O4)[C@H]7[C@@H](O6)[C@@H](O5)[C@@H]8[C@@H](O7)CC[C@@H](O8)CC(=O)O[C@@H]9[C@H]([C@H]3[C@H](C[C@@H]4[C@H](O3)C[C@@]3(O4)C[C@H]4[C@@H](O3)[C@H](C[C@]3(O4)C[C@@H]([C@H]4[C@@H](O3)C[C@H]([C@H](O4)CN)O)C)C)O[C@H]9C[C@H](C1=C)O2)C'
SQUALENE = from_and_to_smiles(SQUALENE)
OTHER = from_and_to_smiles(OTHER)


def main(opts):
    pprint(vars(opts))

    mdl, mdl_encoder, mdl_decoder = buildNetwork(opts.layers, opts.heads, opts.embedding_size, VOCAB_SIZE,
                                                 opts.key_size, opts.n_hidden, opts)

    mdl.load_weights('output/2node_2000epochs_speedy_warmup_20200909T152412/final.h5')
    output_str = ''

    with K.get_session().as_default():
        beams = gen_beam(mdl_encoder, mdl_decoder, opts.temperature, OTHER, 5, opts.max_predict)

        for beam in beams:
            try:
                output_str += OTHER + ' >> ' + beam[0][0] + ' . ' + beam[0][1] + '\n'
            except IndexError:
                pass

    print(output_str)


if __name__ == '__main__':
    opts = get_options()
    main(opts)
