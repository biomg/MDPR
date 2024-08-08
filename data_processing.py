import os
import pathlib
import re

import numpy
import numpy as np
import torch

from sklearn import metrics
from transformers import AutoTokenizer, EsmForProteinFolding
import esm
#
# # Load ESM-2 model
model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#
batch_converter = alphabet.get_batch_converter()
#
# tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
#
model_fold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)


def get_coordinate_hot(tcrsequence):
    tcrs_cnos = {"W": ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'N', 'C', 'C', 'C'],
                 'F': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'C'],
                 'G': ['N', 'C', 'C', 'O'],
                 'A': ['N', 'C', 'C', 'C', 'O'],
                 'V': ['N', 'C', 'C', 'C', 'O', 'C', 'C'],
                 'I': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C'],
                 'L': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C'],
                 'M': ['N', 'C', 'C', 'C', 'O', 'C', 'S', 'C'],
                 'P': ['N', 'C', 'C', 'C', 'O', 'C', 'C'],
                 'Y': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'O', 'C'],
                 'S': ['N', 'C', 'C', 'C', 'O', 'O'],
                 'T': ['N', 'C', 'C', 'C', 'O', 'C', 'O'],
                 'N': ['N', 'C', 'C', 'C', 'O', 'C', 'N', 'O'],
                 'Q': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'N', 'O'],
                 'C': ['N', 'C', 'C', 'C', 'O', 'S'],
                 'K': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'N'],
                 'R': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'N', 'N', 'N', 'C'],
                 'H': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'N', 'C', 'N'],
                 'D': ['N', 'C', 'C', 'C', 'O', 'C', 'O', 'O'],
                 'E': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'O', 'O'],
                 }
    cnos = {'C': 1, 'N': 2, 'O': 3, 'S': 4}

    output = model_fold.infer_pdb(str(tcrsequence))
    with open("result.pdb", "w") as f:
        f.write(output)
        f.close()
    with open('result.pdb', 'r') as f:
        line = f.readline()
        line = f.readline()
        all_xyz = []
        all_hot = []
        while line.split()[0] == "ATOM":
            # print(line)
            if (line.split()[0] == "ATOM"):
                xyz = line.split()[6:9]
                all_hot.append(line.split()[10])
                all_xyz.append(xyz)
                line = f.readline()

        f.close()
        all_xyz = np.array(all_xyz)
        tcrs_cnos = {"W": ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'N', 'C', 'C', 'C'],
                     'F': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'C'],
                     'G': ['N', 'C', 'C', 'O'],
                     'A': ['N', 'C', 'C', 'C', 'O'],
                     'V': ['N', 'C', 'C', 'C', 'O', 'C', 'C'],
                     'I': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C'],
                     'L': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C'],
                     'M': ['N', 'C', 'C', 'C', 'O', 'C', 'S', 'C'],
                     'P': ['N', 'C', 'C', 'C', 'O', 'C', 'C'],
                     'Y': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'O', 'C'],
                     'S': ['N', 'C', 'C', 'C', 'O', 'O'],
                     'T': ['N', 'C', 'C', 'C', 'O', 'C', 'O'],
                     'N': ['N', 'C', 'C', 'C', 'O', 'C', 'N', 'O'],
                     'Q': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'N', 'O'],
                     'C': ['N', 'C', 'C', 'C', 'O', 'S'],
                     'K': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'N'],
                     'R': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'N', 'N', 'N', 'C'],
                     'H': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'N', 'C', 'N'],
                     'D': ['N', 'C', 'C', 'C', 'O', 'C', 'O', 'O'],
                     'E': ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'O', 'O'],
                     }
        cnos = {'C': [0, 0, 0, 1], 'N': [0, 0, 1, 0], 'O': [0, 1, 0, 0], 'S': [1, 0, 0, 0]}
        # $HOME/ljj/python37/bin/python3 tcrs_cno.py

        tcrs = []
        for tcr in tcrsequence:
            for cno_one in tcrs_cnos[tcr]:
                tcrs.append(cno_one)
        cno_xyz = []
        for (i, cno_one) in enumerate(tcrs):
            for (a, one) in enumerate(cnos[cno_one]):
                if one == 1:
                    x = [0.0, 0.0, 0.0, 0.0]
                    y = [0.0, 0.0, 0.0, 0.0]
                    z = [0.0, 0.0, 0.0, 0.0]
                    x[a] = float(all_xyz[i][0])
                    y[a] = float(all_xyz[i][1])
                    z[a] = float(all_xyz[i][2])
                    cno_xyz.append([x, y, z])
        cno_xyz = numpy.array(cno_xyz)
        p = numpy.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
        cno_xyz = numpy.concatenate((p, p, cno_xyz), axis=0)

        # for i in range(152 - len(cno_xyz)):
        #     cno_xyz = numpy.concatenate((cno_xyz, p), axis=0)

        return cno_xyz.reshape(1, -1, 3, 4).astype(float)


def get_word_vector_index(Seq):
    Seq = Seq[1:-1]
    data2 = []

    data2.append(("protein1", Seq))
    # print(data)

    batch_labels, batch_strs, batch_tokens = batch_converter(data2)

    batch_tokens = batch_tokens.numpy()
    batch_tokens = batch_tokens[0]
    for p in range(38 - batch_tokens.shape[0]):
        batch_tokens = np.append(batch_tokens, 1)
    batch_tokens = batch_tokens.reshape(1, batch_tokens.shape[0])
    batch_tokens = torch.from_numpy(batch_tokens)
    return batch_tokens

