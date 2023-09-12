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