import os
import math
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from mpi4py import MPI
import pdb

from data.preprocessing import CHAR_TO_IX, IX_TO_CHAR, VOCAB_SIZE, gen_data


def generate(mdl, product):
    res = ""
    for i in range(1, 70):
        lines = []
        lines.append(product + " >> " + res)

        v = gen_data(lines, True)

        n = mdl.predict(v[0])
        p = n[0, -1, :]

        w = np.argmax(p)
        if w == CHAR_TO_IX["$"]:
            break
        res += IX_TO_CHAR[w]

    return res


def gen_greedy(mdl_encoder, mdl_decoder, T, product, max_predict):
    product_encoded, product_mask = mdl_encoder(product)
    res = ""
    score = 0.0
    for i in range(1, max_predict):
        p = mdl_decoder(res, product_encoded, product_mask, T)
        w = np.argmax(p)
        score -= math.log10(np.max(p))
        if w == CHAR_TO_IX["$"]:
            break
        res += IX_TO_CHAR[w]

    reags = res.split(".")
    sms = set()
    with suppress_stderr():
        for r in reags:
            r = r.replace("$", "")
            m = Chem.MolFromSmiles(r)
            if m is not None:
                sms.add(Chem.MolToSmiles(m))
        if len(sms):
            return [sorted(list(sms)), score]

    return ["", 0.0]


def gen_beam(mdl_encoder, mdl_decoder, T, product, beam_size=1, max_predict=float('inf')):
    product_encoded, product_mask = mdl_encoder(product)
    if beam_size == 1:
        return [gen_greedy(mdl_encoder, mdl_decoder, T, product, max_predict)]

    lines = []
    scores = []
    final_beams = []

    for i in range(beam_size):
        lines.append("")
        scores.append(0.0)

    for step in range(max_predict):
        if step == 0:
            p = mdl_decoder("", product_encoded, product_mask, T)
            nr = np.zeros((VOCAB_SIZE, 2))
            for i in range(VOCAB_SIZE):
                nr[i, 0] = -math.log10(p[i])
                nr[i, 1] = i
        else:
            cb = len(lines)
            nr = np.zeros((cb * VOCAB_SIZE, 2))
            for i in range(cb):
                p = mdl_decoder(lines[i], product_encoded, product_mask, T)
                for j in range(VOCAB_SIZE):
                    nr[i * VOCAB_SIZE + j, 0] = -math.log10(p[j]) + scores[i]
                    nr[i * VOCAB_SIZE + j, 1] = i * 100 + j

        y = nr[nr[:, 0].argsort()]

        new_beams = []
        new_scores = []

        for i in range(beam_size):

            c = IX_TO_CHAR[y[i, 1] % 100]
            beamno = int(y[i, 1]) // 100

            if c == '$':
                added = lines[beamno] + c
                if added != "$":
                    final_beams.append([lines[beamno] + c, y[i, 0]])
                beam_size -= 1
            else:
                new_beams.append(lines[beamno] + c)
                new_scores.append(y[i, 0])

        lines = new_beams
        scores = new_scores

        if len(lines) == 0: break

    for i in range(len(final_beams)):
        final_beams[i][1] = final_beams[i][1] / len(final_beams[i][0])

    final_beams = list(sorted(final_beams, key=lambda x: x[1]))[:5]
    answer = []

    for k in range(5):
        reags = set(final_beams[k][0].split("."))
        sms = set()

        with suppress_stderr():
            for r in reags:
                r = r.replace("$", "")
                m = Chem.MolFromSmiles(r)
                if m is not None:
                    sms.add(Chem.MolToSmiles(m))
                    # print(sms)
            if len(sms):
                answer.append([sorted(list(sms)), final_beams[k][1]])

    return answer


def validate_one_example(input):

    line, mdl_encoder, mdl_decoder, T, beam_size, opts = input

    if len(line) == 0:
        return

    reaction = line.split(">")
    product = reaction[0].strip()
    reagents = reaction[2].strip()

    answer = []

    reags = set(reagents.split("."))
    sms = set()
    with suppress_stderr():
        for r in reags:
            m = Chem.MolFromSmiles(r)
            if m is not None:
                sms.add(Chem.MolToSmiles(m))
        if len(sms):
            answer = sorted(list(sms))
    if len(answer) == 0:
        return

    beams = []
    try:
        beams = gen_beam(mdl_encoder, mdl_decoder, T, product, beam_size, opts.max_predict)
    except KeyboardInterrupt:
        return
    except:
        pass

    if len(beams) == 0:
        return

    answer_s = set(answer)

    ans = []
    for k in range(len(beams)):
        ans.append([beams[k][0], beams[k][1]])

    ex_1, ex_3, ex_5 = 0, 0, 0

    for step, beam in enumerate(ans):
        right = answer_s.intersection(set(beam[0]))

        if len(right) == 0:
            continue
        if len(right) == len(answer):
            if step == 0:
                ex_1 += 1
                ex_3 += 1
                ex_5 += 1
                # print("CNT: ", cnt, ex_1 / cnt * 100.0, answer, beam[1], beam[1] / len(".".join(answer)), 1.0)
                break
            if step < 3:
                ex_3 += 1
                ex_5 += 1
                break
            if step < 5:
                ex_5 += 1
                break
        break

    return np.array([ex_1, ex_3, ex_5])


def validate(ftest, mdl_encoder, mdl_decoder, T, beam_size, opts):
    NTEST = sum(1 for _ in open(ftest, "r"))

    with open(ftest, 'r') as fv:
        lines = fv.readlines()

    # Set up the MPI prerequisites
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    # Define the size of each chunk
    m = int(math.ceil(float(len(lines)) / world_size))

    # Get the specific chunk
    if rank == world_size - 1:
        x_chunk = lines[rank * m:]
    else:
        x_chunk = lines[rank * m:(rank + 1) * m]

    x_chunk = [(x, mdl_encoder, mdl_decoder, T, beam_size, opts) for x in x_chunk]

    # Perform the parallel computations
    r_chunk = list(map(validate_one_example, tqdm(x_chunk)))

    # Get all the results
    r = comm.allreduce(r_chunk)

    if rank == 0:
        r = [x for x in r if x is not None]
        r = sum(r) / len(lines)
        print(r)


        # print("Exact: ", idx, ex_1 / cnt * 100.0, ex_3 / cnt * 100.0, ex_5 * 100.0 / cnt, cnt)

    # print("Exact: ", T, ex_1 / cnt * 100.0, ex_3 / cnt * 100.0, ex_5 * 100.0 / cnt, cnt)


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
