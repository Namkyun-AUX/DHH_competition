import torch
import torch.nn as nn
import torch.nn.functional as F

class base_net(nn.Module):

    def __init__(self):
        super(base_net, self).__init__()
        # Conv1d
        self.conv1 = nn.Conv1d(1, 4, 3, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, stride=1)
        self.conv3 = nn.Conv1d(8, 8, 3, stride=1)
        self.conv4 = nn.Conv1d(8, 8, 3, stride=1)
        self.conv5 = nn.Conv1d(8, 8, 3, stride=1)
        # fullyconnected
        self.fc1_1 = nn.Linear(7 * 8, 64)
        self.fc2_1 = nn.Linear(64, 32)
        self.fc3_1 = nn.Linear(32, 2)
        
        self.fc1_2 = nn.Linear(7 * 8, 64)
        self.fc2_2 = nn.Linear(64, 32)
        self.fc3_2 = nn.Linear(32, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = x5.view(-1, 7 * 8)
        
        x6_1 = F.relu(self.fc1_1(x5))
        x7_1 = F.relu(self.fc2_1(x6_1))
        x8_1 = self.fc3_1(x7_1)
        
        x6_2 = F.relu(self.fc1_2(x5))
        x7_2 = F.relu(self.fc2_2(x6_2))
        x8_2 = self.fc3_2(x7_2)
        return x8_1, x8_2 #self.conv(x)

class mid3conv_net(nn.Module):

    def __init__(self):
        super(mid3conv_net, self).__init__()
        # Conv1d
        self.conv1 = nn.Conv1d(1, 8, 3, stride=1)
        self.batch1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, 3, stride=1)
        self.batch2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 16, 3, stride=1)
        self.batch3 = nn.BatchNorm1d(16)
        self.conv4 = nn.Conv1d(16, 16, 3, stride=1)
        self.batch4 = nn.BatchNorm1d(16)
        self.conv5 = nn.Conv1d(16, 16, 3, stride=1)
        self.batch5 = nn.BatchNorm1d(16)

        # fullyconnected
        self.fc1_1 = nn.Linear(4 * 16, 128)
        self.batch6_1 = nn.BatchNorm1d(128)
        self.fc2_1 = nn.Linear(128, 64)
        self.batch7_1 = nn.BatchNorm1d(64)
        self.fc3_1 = nn.Linear(64, 2)
        
        self.fc1_2 = nn.Linear(4 * 16, 128)
        self.batch6_2 = nn.BatchNorm1d(128)
        self.fc2_2 = nn.Linear(128, 64)
        self.batch7_2 = nn.BatchNorm1d(64)
        self.fc3_2 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = F.relu(self.batch1(self.conv1(x)))
        x2 = F.relu(self.batch2(self.conv2(x1)))
        x3 = F.relu(self.batch3(self.conv3(x2)))
        x4 = F.relu(self.batch4(self.conv4(x3)))
        x5 = F.relu(self.batch5(self.conv5(x4)))
        x5 = x5.view(-1, 4 * 16)
        
        x6_1 = F.relu(self.batch6_1(self.fc1_1(x5)))
        x7_1 = F.relu(self.batch7_1(self.fc2_1(x6_1)))
        x8_1 = self.fc3_1(x7_1)
        
        x6_2 = F.relu(self.batch6_2(self.fc1_2(x5)))
        x7_2 = F.relu(self.batch7_2(self.fc2_2(x6_2)))
        x8_2 = self.fc3_2(x7_2)
        return x8_1, x8_2

class mid4conv_net(nn.Module):

    def __init__(self):
        super(mid4conv_net, self).__init__()
        # Conv1d
        self.conv1 = nn.Conv1d(1, 8, 4, stride=1)
        self.batch1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, 4, stride=1)
        self.batch2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 16, 4, stride=1)
        self.batch3 = nn.BatchNorm1d(16)
        self.conv4 = nn.Conv1d(16, 16, 4, stride=1)
        self.batch4 = nn.BatchNorm1d(16)
        self.conv5 = nn.Conv1d(16, 16, 4, stride=1)
        self.batch5 = nn.BatchNorm1d(16)
        
        # fullyconnected
        self.fc1_1 = nn.Linear(2 * 16, 128)
        self.batch6_1 = nn.BatchNorm1d(128)
        self.fc2_1 = nn.Linear(128, 64)
        self.batch7_1 = nn.BatchNorm1d(64)
        self.fc3_1 = nn.Linear(64, 2)
        
        self.fc1_2 = nn.Linear(2 * 16, 128)
        self.batch6_2 = nn.BatchNorm1d(128)
        self.fc2_2 = nn.Linear(128, 64)
        self.batch7_2 = nn.BatchNorm1d(64)
        self.fc3_2 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = F.relu(self.batch1(self.conv1(x)))
        x2 = F.relu(self.batch2(self.conv2(x1)))
        x3 = F.relu(self.batch3(self.conv3(x2)))
        x4 = F.relu(self.batch4(self.conv4(x3)))
        x5 = F.relu(self.batch5(self.conv5(x4)))
        x5 = x5.view(-1, 2 * 16)
        
        x6_1 = F.relu(self.batch6_1(self.fc1_1(x5)))
        x7_1 = F.relu(self.batch7_1(self.fc2_1(x6_1)))
        x8_1 = self.fc3_1(x7_1)
        
        x6_2 = F.relu(self.batch6_2(self.fc1_2(x5)))
        x7_2 = F.relu(self.batch7_2(self.fc2_2(x6_2)))
        x8_2 = self.fc3_2(x7_2)
        return x8_1, x8_2

class EDecoder_net(nn.Module):

    def __init__(self):
        super(EDecoder_net, self).__init__()
        # Encoder
        self.fc_e1 = nn.Linear(15, 64)
        self.batch_e1 = nn.BatchNorm1d(64)
        self.fc_e2 = nn.Linear(64, 128)
        self.batch_e2 = nn.BatchNorm1d(128)
        self.fc_e3 = nn.Linear(128, 256)
        self.batch_e3 = nn.BatchNorm1d(256)
        self.fc_e4 = nn.Linear(256, 128)
        self.batch_e4 = nn.BatchNorm1d(128)
        self.fc_e5 = nn.Linear(128, 64)
        self.batch_e5 = nn.BatchNorm1d(64)

        # fullyconnected
        # self.fc1_1 = nn.Linear(64, 128)
        # self.batch6_1 = nn.BatchNorm1d(128)
        # self.fc2_1 = nn.Linear(128, 64)
        # self.batch7_1 = nn.BatchNorm1d(64)
        # self.fc3_1 = nn.Linear(64, 2)

        self.fc1_2 = nn.Linear(64, 128)
        self.batch6_2 = nn.BatchNorm1d(128)
        self.fc2_2 = nn.Linear(128, 64)
        self.batch7_2 = nn.BatchNorm1d(64)
        self.fc3_2 = nn.Linear(64, 1)

        # self.fc1_3 = nn.Linear(3, 128)
        # self.batch6_3 = nn.BatchNorm1d(128)
        # self.fc2_3 = nn.Linear(128, 64)
        # self.batch7_3 = nn.BatchNorm1d(64)
        # self.fc3_3 = nn.Linear(64, 2)

    def forward(self, x):
        x1 = F.relu(self.batch_e1(self.fc_e1(x)))
        x2 = F.relu(self.batch_e2(self.fc_e2(x1)))
        x3 = F.relu(self.batch_e3(self.fc_e3(x2)))
        x4 = F.relu(self.batch_e4(self.fc_e4(x3)))
        x5 = F.relu(self.batch_e5(self.fc_e5(x4)))
        
        # x6_1 = F.relu(self.batch6_1(self.fc1_1(x5)))
        # x7_1 = F.relu(self.batch7_1(self.fc2_1(x6_1)))
        # x8_1 = self.fc3_1(x7_1)

        x6_2 = F.relu(self.batch6_2(self.fc1_2(x5)))
        x7_2 = F.relu(self.batch7_2(self.fc2_2(x6_2)))
        x8_2 = self.fc3_2(x7_2)

        # x6_3 = F.relu(self.batch6_3(self.fc1_3(torch.cat((x8_1, x8_2), 1))))
        # x7_3 = F.relu(self.batch7_3(self.fc2_3(x6_3)))
        # x8_3 = self.fc3_3(x7_3)

        return x8_2, #x8_1, x8_3

class EDecoder_net_large(nn.Module):

    def __init__(self):
        super(EDecoder_net_large, self).__init__()
        # Encoder
        self.fc_e1 = nn.Linear(17, 128)
        self.batch_e1 = nn.BatchNorm1d(128)
        self.fc_e2 = nn.Linear(128, 256)
        self.batch_e2 = nn.BatchNorm1d(256)
        self.fc_e3 = nn.Linear(256, 512)
        self.batch_e3 = nn.BatchNorm1d(512)
        self.fc_e4 = nn.Linear(512, 256)
        self.batch_e4 = nn.BatchNorm1d(256)
        self.fc_e5 = nn.Linear(256, 128)
        self.batch_e5 = nn.BatchNorm1d(128)

        self.fc1_2 = nn.Linear(128, 256)
        self.batch6_2 = nn.BatchNorm1d(256)
        self.fc2_2 = nn.Linear(256, 128)
        self.batch7_2 = nn.BatchNorm1d(128)
        self.fc3_2 = nn.Linear(128, 1)

    def forward(self, x):
        x1 = F.relu(self.batch_e1(self.fc_e1(x)))
        x2 = F.relu(self.batch_e2(self.fc_e2(x1)))
        x3 = F.relu(self.batch_e3(self.fc_e3(x2)))
        x4 = F.relu(self.batch_e4(self.fc_e4(x3)))
        x5 = F.relu(self.batch_e5(self.fc_e5(x4)))
        
        x6_2 = F.relu(self.batch6_2(self.fc1_2(x5)))
        x7_2 = F.relu(self.batch7_2(self.fc2_2(x6_2)))
        x8_2 = self.fc3_2(x7_2)

        return x8_2, #x8_1, x8_3
