import numpy as np
import random

from layers import GetPosEncodingMatrix


CHARS = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"  # UPTSO
VOCAB_SIZE = len(CHARS)

CHAR_TO_IX = {ch: i for i, ch in enumerate(CHARS)}
IX_TO_CHAR = {i: ch for i, ch in enumerate(CHARS)}


def gen_right(data, opts):
    batch_size = len(data)
    nr = len(data[0]) + 1

    y = np.zeros((batch_size, nr), np.int8)
    my = np.zeros((batch_size, nr), np.int8)
    py = np.zeros((batch_size, nr, opts.embedding_size), np.float32)

    GEO = GetPosEncodingMatrix(opts.max_predict, opts.embedding_size)

    for cnt in range(batch_size):
        reactants = "^" + data[cnt]
        for i, p in enumerate(reactants):
            y[cnt, i] = CHAR_TO_IX[p]
            py[cnt, i] = GEO[i + 1, :opts.embedding_size]
        my[cnt, :i + 1] = 1
    return y, my, py


def gen_left(data, opts):
    batch_size = len(data)
    nl = len(data[0]) + 2

    x = np.zeros((batch_size, nl), np.int8)
    mx = np.zeros((batch_size, nl), np.int8)
    px = np.zeros((batch_size, nl, opts.embedding_size), np.float32)

    GEO = GetPosEncodingMatrix(opts.max_predict, opts.embedding_size)

    for cnt in range(batch_size):
        product = "^" + data[cnt] + "$"
        for i, p in enumerate(product):
            x[cnt, i] = CHAR_TO_IX[p]
            px[cnt, i] = GEO[i + 1, :opts.embedding_size]
        mx[cnt, :i + 1] = 1
    return x, mx, px


def gen_data(data, progn=False):
    batch_size = len(data)

    # search for max lengths
    left = []
    right = []
    for line in data:
        line = line.split(">")
        # left.append( list(filter(None, re.split(token_regex, line[0].strip() ))) ) 
        left.append(line[0].strip())
        if len(line) > 1:
            # right.append( list(filter(None, re.split(token_regex, line[2].strip() ))) ) 
            right.append(line[2].strip())
        else:
            right.append("")

    nl = len(left[0])
    nr = len(right[0])
    for i in range(1, batch_size, 1):
        nl_a = len(left[i])
        nr_a = len(right[i])
        if nl_a > nl:
            nl = nl_a
        if nr_a > nr:
            nr = nr_a

            # add start and end symbols
    nl += 2
    nr += 1

    # products
    x = np.zeros((batch_size, nl), np.int8)
    mx = np.zeros((batch_size, nl), np.int8)

    # reactants
    y = np.zeros((batch_size, nr), np.int8)
    my = np.zeros((batch_size, nr), np.int8)

    # for output
    z = np.zeros((batch_size, nr, VOCAB_SIZE), np.int8)

    for cnt in range(batch_size):
        product = "^" + left[cnt] + "$"
        reactants = "^" + right[cnt]

        if progn == False: reactants += "$"
        for i, p in enumerate(product):
            x[cnt, i] = CHAR_TO_IX[p]

        mx[cnt, :i + 1] = 1
        for i in range((len(reactants) - 1) if progn == False else len(reactants)):
            y[cnt, i] = CHAR_TO_IX[reactants[i]]
            if progn == False:
                z[cnt, i, CHAR_TO_IX[reactants[i + 1]]] = 1

        my[cnt, :i + 1] = 1

    return [x, mx, y, my], z


def data_generator_deprecated(fname, batch_size):
    f = open(fname, "r")
    lines = []

    while True:
        for i in range(batch_size):
            line = f.readline()
            if len(line) == 0:
                f.seek(0, 0)
                if len(lines) > 0:
                    yield gen_data(lines)
                lines = []
                break
            lines.append(line)
        if len(lines) > 0:
            yield gen_data(lines)
            lines = []


def data_generator(fname, batch_size):
    with open(fname, 'r') as in_file:
        all_lines = in_file.read().splitlines()

    lines = []

    while True:
        for i in range(batch_size):
            lines.append(random.choice(all_lines))

        yield gen_data(lines)

        lines = []


def val_generator(fname='data/retrosynthesis-test.smi'):
    with open(fname, 'r') as in_file:
        all_lines = in_file.read().splitlines()

    return gen_data(all_lines)
