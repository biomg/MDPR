import os
import pathlib
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy
from sklearn import metrics
import torch

from torch import nn
class spatial_feature_extraction_module(nn.Module):

    def __init__(self):
        super(spatial_feature_extraction_module, self).__init__()

        self.conv1 = nn.Conv2d(3, 30, kernel_size=(11, 4), stride=1)
        nn.init.xavier_normal_(self.conv1.weight, gain=1)

        self.conv2 = nn.Conv1d(30, 30, kernel_size=(11,), stride=1, padding='same')
        nn.init.xavier_normal_(self.conv2.weight, gain=1)

        self.conv3 = nn.Conv1d(60, 60, kernel_size=(11,), stride=1, padding='same')
        nn.init.xavier_normal_(self.conv3.weight, gain=1)  # best   5  11  13  15

        self.Flatten = nn.Flatten()

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(8520, 1024)
        nn.init.xavier_normal_(self.linear1.weight, gain=1)

        self.linear2 = nn.Linear(1024, 2)
        nn.init.xavier_normal_(self.linear1.weight, gain=1)
        # self.linear3=nn.Linear(100,2)
        self.BatchNorm1 = nn.BatchNorm2d(30)

        self.BatchNorm2 = nn.BatchNorm1d(30)
        self.BatchNorm3 = nn.BatchNorm1d(60)
        self.BatchNorm4 = nn.LazyBatchNorm2d()
        self.BatchNorm5 = nn.LazyBatchNorm2d()
        self.BatchNorm6 = nn.LazyBatchNorm2d()
        self.dropout = nn.Dropout(0.6)  # best=0.3
        self.max = nn.MaxPool2d(kernel_size=(2, 1), stride=2)

    def forward(self, input):
        input = input.reshape(input.shape[0], input.shape[2], input.shape[1], input.shape[3])
        # food_conv1 = self.maxpool1(self.dropout(self.BatchNorm(self.relu(self.food_conv1(input)))).squeeze(3))
        conv1 = self.conv1(input)
        conv1 = self.relu(conv1)
        conv1 = self.BatchNorm1(conv1)
        conv1 = self.dropout(conv1)
        conv1 = conv1.reshape(conv1.shape[0], conv1.shape[1], conv1.shape[2])

        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        conv2 = self.BatchNorm2(conv2)
        conv2 = self.dropout(conv2)
        conv2 = torch.cat([conv1, conv2], 1)

        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        conv3 = self.BatchNorm3(conv3)
        conv3 = self.dropout(conv3)

        conv3 = self.Flatten(conv3)

        # print(conv3.shape)

        all = self.linear2(self.dropout(self.relu(self.linear1(conv3))))

        return all


class MDPR(nn.Module):

    def __init__(self, spatial_feature_extraction_module):
        super(MDPR, self).__init__()
        self.spatial_feature_extraction_module = spatial_feature_extraction_module
        # self.embedding =nn.Embedding(33, 1280)
        self.embedding = nn.Embedding.from_pretrained(torch.load("embedding_token"))

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(17, 13), stride=1)
        nn.init.xavier_normal_(self.conv1.weight, gain=1)

        self.conv2 = nn.Conv2d(1, 3, kernel_size=(17, 3), stride=1)
        nn.init.xavier_normal_(self.conv2.weight, gain=1)

        self.conv3 = nn.Conv2d(1, 3, kernel_size=(17, 5), stride=1)
        nn.init.xavier_normal_(self.conv3.weight, gain=1)

        self.conv4 = nn.Conv2d(1, 3, kernel_size=(17, 7), stride=1)
        nn.init.xavier_normal_(self.conv4.weight, gain=1)

        self.conv5 = nn.Conv2d(1, 3, kernel_size=(17, 9), stride=1)
        nn.init.xavier_normal_(self.conv5.weight, gain=1)
        self.conv6 = nn.Conv2d(1, 3, kernel_size=(17, 11), stride=1)
        nn.init.xavier_normal_(self.conv6.weight, gain=1)
        self.Flatten = nn.Flatten()

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(11457, 4500)
        nn.init.xavier_normal_(self.linear1.weight, gain=1)

        self.linear2 = nn.Linear(4500, 2)
        self.linear3 = nn.Linear(4, 2)
        nn.init.xavier_normal_(self.linear1.weight, gain=1)
        # self.linear3=nn.Linear(100,2)
        self.BatchNorm1 = nn.LazyBatchNorm2d()

        self.BatchNorm2 = nn.LazyBatchNorm2d()
        self.BatchNorm3 = nn.LazyBatchNorm2d()
        self.BatchNorm4 = nn.LazyBatchNorm2d()
        self.BatchNorm5 = nn.LazyBatchNorm2d()
        self.BatchNorm6 = nn.LazyBatchNorm2d()
        self.dropout = nn.Dropout(0.6)  # best=0.3
        self.max = nn.MaxPool2d(kernel_size=(1, 2), stride=2)

    def forward(self, input, xyz):
        input = self.embedding(input)
        input = input.reshape(input.shape[0], 1, input.shape[1], input.shape[2])
        # food_conv1 = self.maxpool1(self.dropout(self.BatchNorm(self.relu(self.food_conv1(input)))).squeeze(3))
        conv1 = self.conv1(input)
        conv1 = self.relu(conv1)
        conv1 = self.BatchNorm1(conv1)
        conv1 = self.dropout(conv1)
        conv1 = self.max(conv1)
        conv1 = self.Flatten(conv1)

        conv2 = self.conv2(input)
        conv2 = self.relu(conv2)
        conv2 = self.BatchNorm2(conv2)
        conv2 = self.dropout(conv2)
        conv2 = self.max(conv2)
        conv2 = self.Flatten(conv2)

        conv3 = self.conv3(input)
        conv3 = self.relu(conv3)
        conv3 = self.BatchNorm3(conv3)
        conv3 = self.dropout(conv3)
        conv3 = self.max(conv3)
        conv3 = self.Flatten(conv3)

        conv4 = self.conv4(input)
        conv4 = self.relu(conv4)
        conv4 = self.BatchNorm4(conv4)
        conv4 = self.dropout(conv4)
        conv4 = self.max(conv4)
        conv4 = self.Flatten(conv4)

        conv5 = self.conv5(input)
        conv5 = self.relu(conv5)
        conv5 = self.BatchNorm5(conv5)
        conv5 = self.dropout(conv5)
        conv5 = self.max(conv5)
        conv5 = self.Flatten(conv5)

        conv6 = self.conv6(input)
        conv6 = self.relu(conv6)
        conv6 = self.BatchNorm6(conv6)
        conv6 = self.dropout(conv6)
        conv6 = self.max(conv6)
        conv6 = self.Flatten(conv6)

        all = torch.cat([conv2, conv3, conv4, conv5, conv6, conv1], 1)

        all1 = self.linear2(self.dropout(self.relu(self.linear1(all))))
        all2 = self.spatial_feature_extraction_module(xyz)
        all = torch.cat([all2, all1], 1)

        return self.linear3(all)



def caculateAUC(AUC_outs, AUC_labels):
    ROC = 0
    outs = []
    labels = []
    for (index, AUC_out) in enumerate(AUC_outs):
        softmax = nn.Softmax(dim=1)
        out = softmax(AUC_out).detach().numpy()
        out = out[:, 1]
        for out_one in out.tolist():
            outs.append(out_one)
        for AUC_one in AUC_labels[index].tolist():
            labels.append(AUC_one)

    outs = np.array(outs)

    labels = np.array(labels)
    np.save("duomotai_label", labels)
    np.save("duomotai_outs", outs)
    fpr, tpr, thresholds = metrics.roc_curve(labels, outs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(labels, outs)

    return auc, aupr


class Mydata(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x, z, y = self.data[index]
        gene = torch.from_numpy(x)
        z = torch.from_numpy(z)
        # gene=gene.float()

        return gene.to(torch.int), z.to(torch.float), int(y)

    def __len__(self):
        return self.data.shape[0]

feature = np.load("data/m_yes.npy", allow_pickle=True)
feature_number = feature.shape[0]
print(feature_number)
vn_idx = range(0, feature_number)

count = 0
nn_s = int(np.ceil(feature_number * (1 - 0.33)))

feature2 = np.load("data/m_no.npy", allow_pickle=True)
feature_number2 = feature2.shape[0]
print(feature_number2)
vn_idx2 = range(0, feature_number2)

nn_s2 = int(np.ceil(feature_number2 * (1 - 0.33)))

PredictClassList = []
PredictLabelList = []
AUCDictList = []
n = 1
all_acc = 0.0
all_auc = 0.0
while count < n:
    print("==============Training cycle %d.=============" % (count))
    ID = str(count)
    count = count + 1
    vn_train = np.random.choice(vn_idx, nn_s, replace=False)

    vn_test = [x for x in vn_idx if x not in vn_train]

    vn_train2 = np.random.choice(vn_idx2, nn_s2, replace=False)
    vn_test2 = [x for x in vn_idx2 if x not in vn_train2]
    np.save("vn_train",arr=vn_train)
    np.save("vn_test",arr=vn_test)
    np.save("vn_train2",arr=vn_train2)
    np.save("vn_test2",arr=vn_test2)
    train_data = np.array(feature)[vn_train]
    test_data = np.array(feature)[vn_test]

    train_data2 = np.array(feature2)[vn_train2]
    test_data2 = np.array(feature2)[vn_test2]

    my_train = Mydata(train_data) + Mydata(train_data2)
    my_test = Mydata(test_data) + Mydata(test_data2)

    dataloader_train = DataLoader(dataset=my_train, batch_size=200, shuffle=True)
    dataloader_test = DataLoader(dataset=my_test, batch_size=200, shuffle=True)
    model_spatial_feature_extraction_module = spatial_feature_extraction_module()
    model_spatial_feature_extraction_module = model_spatial_feature_extraction_module.cuda()
    # optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': 0.00001}], lr=0.00001,
    #                             momentum=0.99,
    #                             weight_decay=5e-3)
    optimizer = torch.optim.AdamW(params=model_spatial_feature_extraction_module.parameters(), lr=0.0001, weight_decay=5e-3)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)
    # optimizer = torch.optim.Adam(params=model.parameters(),lr=0.0001)
    loss = torch.nn.CrossEntropyLoss()
    for i in range(600):  # 600
        test_acc = 0.0
        train_acc = 0.0
        model_spatial_feature_extraction_module.train()
        for data in dataloader_train:
            input, xyz, label = data
            input = input.cuda()
            xyz = xyz.cuda()
            label = label.cuda()
            out = model_spatial_feature_extraction_module(xyz)
            out = out.cuda()
            optimizer.zero_grad()
            result_loss = loss(out, label)

            result_loss = result_loss.cuda()
            result_loss.backward()
            optimizer.step()
            train_acc = train_acc + ((out.argmax(1) == label).sum())
    model = MDPR(spatial_feature_extraction_module=model_spatial_feature_extraction_module)
    model = model.cuda()
    optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': 0.00001}], lr=0.00001,
                                momentum=0.99,
                                weight_decay=5e-3)
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.00001, weight_decay=5e-3)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 3000], gamma=0.1)
    # optimizer = torch.optim.Adam(params=model.parameters(),lr=0.0001)
    loss = torch.nn.CrossEntropyLoss()
    for i in range(4000):  # 4000
        test_acc = 0.0
        train_acc = 0.0
        model.train()
        for data in dataloader_train:
            input, xyz, label = data
            input = input.cuda()
            xyz = xyz.cuda()
            label = label.cuda()
            out = model(input, xyz)
            out = out.cuda()
            optimizer.zero_grad()
            result_loss = loss(out, label)

            result_loss = result_loss.cuda()
            result_loss.backward()
            optimizer.step()
            train_acc = train_acc + ((out.argmax(1) == label).sum())
        # print(i)
        # print("train")
        train_scheduler.step()
        # print(float(train_acc / my_train.__len__()))
        model.eval()
        auc_label = []
        auc_out = []
        test_acc = 0.0
        with torch.no_grad():

            for data in dataloader_test:
                input, xyz, label = data
                input = input.cuda()

                xyz = xyz.cuda()

                label = label.cuda()
                out = model(input, xyz)
                out = out.cuda()

                result_loss = loss(out, label)
                result_loss = result_loss.cuda()

                auc_label.append(label.cpu().numpy())
                auc_out.append(out.cpu())

                test_acc = test_acc + ((out.argmax(1) == label).sum())
            print(i)
        # #print("test")

            auc_number, aupr = caculateAUC(auc_out, auc_label)
            print("acc:{},auc:{}".format(float(test_acc / my_test.__len__()), auc_number))



