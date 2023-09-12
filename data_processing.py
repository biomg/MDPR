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
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
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

        for i in range(152 - len(cno_xyz)):
            cno_xyz = numpy.concatenate((cno_xyz, p), axis=0)

        return cno_xyz.reshape(1, 152, 3, 4).astype(float)


def get_word_vector_index(Seq):
    Seq = Seq[1:-1]
    data2 = []

    data2.append(("protein1", Seq))
    # print(data)

    batch_labels, batch_strs, batch_tokens = batch_converter(data2)

    batch_tokens = batch_tokens.numpy()
    batch_tokens = batch_tokens[0]
    for p in range(17 - batch_tokens.shape[0]):
        batch_tokens = np.append(batch_tokens, 1)
    batch_tokens = batch_tokens.reshape(1, batch_tokens.shape[0])
    batch_tokens = torch.from_numpy(batch_tokens)
    return batch_tokens


curPath = os.getcwd()
def GetFeatureLabels(TumorCDR3s, NonTumorCDR3s):
    nt = len(TumorCDR3s)
    nc = len(NonTumorCDR3s)
    LLt = [len(ss) for ss in TumorCDR3s]
    LLt = np.array(LLt)

    LLc = [len(ss) for ss in NonTumorCDR3s]
    LLc = np.array(LLc)
    NL = range(12, 18)
    FeatureDict = {}
    LabelDict = {}
    for LL in NL:
        vvt = np.where(LLt == LL)[0]

        vvc = np.where(LLc == LL)[0]
        Labels = [1] * len(vvt) + [0] * len(vvc)
        Labels = np.array(Labels)
        Labels = Labels.astype(np.int32)
        data = []
        for ss in TumorCDR3s[vvt]:
            if len(pat.findall(ss)) > 0:
                continue
            data.append(AAindexEncoding(ss))
        #            data.append(OneHotEncoding(ss))
        for ss in NonTumorCDR3s[vvc]:
            if len(pat.findall(ss)) > 0:
                continue
            data.append(AAindexEncoding(ss))
        #            data.append(OneHotEncoding(ss))
        data = np.array(data)
        features = {'x': data, 'LL': LL}
        FeatureDict[LL] = features
        LabelDict[LL] = Labels
    return FeatureDict, LabelDict




def AAindexEncoding(Seq):


    return Seq






pat = re.compile('[\\*_XB]')  ## non-productive CDR3 patterns






def get_data(ftumor, fnormal,rate=0.33,dir_prefix=curPath + '/tmp'):

    ftumor=ftumor
    fnormal=fnormal
    pathlib.Path(dir_prefix).mkdir(parents=True, exist_ok=True)
    tumorCDR3s = []
    g = open(ftumor)
    for ll in g.readlines():
        rr = ll.strip()
        if not rr.startswith('C') or not rr.endswith('F'):
            print("Non-standard CDR3s. Skipping.")
            continue
        tumorCDR3s.append(rr)
    normalCDR3s = []
    g = open(fnormal)
    for ll in g.readlines():
        rr = ll.strip()
        if not rr.startswith('C') or not rr.endswith('F'):
            print("Non-standard CDR3s. Skipping.")
            continue
        normalCDR3s.append(rr)
    count = 0
    nt = len(tumorCDR3s)

    nn = len(normalCDR3s)

    vt_idx = range(0, nt)
    vn_idx = range(0, nn)
    nt_s = int(np.ceil(nt * (1 - rate)))

    nn_s = int(np.ceil(nn * (1 - rate)))
    PredictClassList = []
    PredictLabelList = []
    AUCDictList = []
    n=1
    while count < n:
        print("==============Training cycle %d.=============" % (count))
        ID = str(count)
        vt_train = np.random.choice(vt_idx, nt_s, replace=False)
        vt_test = [x for x in vt_idx if x not in vt_train]
        vn_train = np.random.choice(vn_idx, nn_s, replace=False)
        vn_test = [x for x in vn_idx if x not in vn_train]
        sTumorTrain = np.array(tumorCDR3s)[vt_train]

        sNormalTrain = np.array(normalCDR3s)[vn_train]
        sTumorTest = np.array(tumorCDR3s)[vt_test]
        sNormalTest = np.array(normalCDR3s)[vn_test]
        ftrain_tumor = dir_prefix + '/sTumorTrain-' + str(ID) + '.txt'

        ftrain_normal = dir_prefix + '/sNormalTrain-' + str(ID) + '.txt'
        feval_tumor = dir_prefix + '/sTumorTest-' + str(ID) + '.txt'
        feval_normal = dir_prefix + '/sNormalTest-' + str(ID) + '.txt'
        h = open(ftrain_tumor, 'w')
        _ = [h.write(x + '\n') for x in sTumorTrain]
        h.close()
        h = open(ftrain_normal, 'w')
        _ = [h.write(x + '\n') for x in sNormalTrain]
        h.close()
        h = open(feval_tumor, 'w')
        _ = [h.write(x + '\n') for x in sTumorTest]
        h.close()
        h = open(feval_normal, 'w')
        _ = [h.write(x + '\n') for x in sNormalTest]
        h.close()
        g = open(ftrain_tumor)

        Train_Tumor = []
        for line in g.readlines():
            Train_Tumor.append(line.strip())
        Train_Tumor = np.array(Train_Tumor)

        g = open(ftrain_normal)
        Train_Normal = []
        for line in g.readlines():
            Train_Normal.append(line.strip())
        Train_Normal = np.array(Train_Normal)
        TrainFeature, TrainLabels = GetFeatureLabels(Train_Tumor, Train_Normal)

        g = open(feval_tumor)
        Eval_Tumor = []
        for line in g.readlines():
            Eval_Tumor.append(line.strip())
        Eval_Tumor = np.array(Eval_Tumor)
        g = open(feval_normal)
        Eval_Normal = []
        for line in g.readlines():
            Eval_Normal.append(line.strip())
        Eval_Normal = np.array(Eval_Normal)
        EvalFeature, EvalLabels = GetFeatureLabels(Eval_Tumor, Eval_Normal)


        count = count + 1
        Train_data = []
        for x in TrainFeature[12]["x"]:
            Train_data.append(x)
        for x in TrainFeature[13]["x"]:
            Train_data.append(x)
        for x in TrainFeature[14]["x"]:
            Train_data.append(x)
        for x in TrainFeature[15]["x"]:
            Train_data.append(x)
        for x in TrainFeature[16]["x"]:
            Train_data.append(x)
        for x in TrainFeature[17]["x"]:
            Train_data.append(x)


        Train_label = []
        for y in TrainLabels[12]:
            Train_label.append(y)
        for y in TrainLabels[13]:
            Train_label.append(y)
        for y in TrainLabels[14]:
            Train_label.append(y)
        for y in TrainLabels[15]:
            Train_label.append(y)
        for y in TrainLabels[16]:
            Train_label.append(y)
        for y in TrainLabels[17]:
            Train_label.append(y)




        Eval_data = []
        for x in EvalFeature[12]["x"]:
            Train_data.append(x)
        for x in EvalFeature[13]["x"]:
            Train_data.append(x)
        for x in EvalFeature[14]["x"]:
            Train_data.append(x)
        for x in EvalFeature[15]["x"]:
            Train_data.append(x)
        for x in EvalFeature[16]["x"]:
            Train_data.append(x)
        for x in EvalFeature[17]["x"]:
            Train_data.append(x)


        Eval_label = []
        for y in EvalLabels[12]:
            Train_label.append(y)
        for y in EvalLabels[13]:
            Train_label.append(y)
        for y in EvalLabels[14]:
            Train_label.append(y)
        for y in EvalLabels[15]:
            Train_label.append(y)
        for y in EvalLabels[16]:
            Train_label.append(y)
        for y in EvalLabels[17]:
            Train_label.append(y)


        yes=[]
        no=[]
        for (i,input) in enumerate(Train_data):
                     c_xyz = get_coordinate_hot(input)

                     c_tcrs = get_word_vector_index(input)
                     c_tcrs=c_tcrs.numpy()
                     print(i)
                     if Train_label[i]==0:
                         no.append((c_tcrs[0],c_xyz[0],Train_label[i]))
                     else:

                         yes.append((c_tcrs[0],c_xyz[0],Train_label[i]))

        numpy.save("data_yes",arr=yes)
        numpy.save("data_no",arr=no)


    return 0


get_data(ftumor="TrainingData/CancerTrain.txt",fnormal="TrainingData/ControlTrain.txt")