import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
def to_np(x):
    return x.data.cpu().numpy()
class VA_LSTM(nn.Module):
    def __init__(self):
        super(VA_LSTM, self).__init__()  #64 120 1024
        self.lstm_v1_out = 64
        self.lstm_v1 = nn.LSTM(1024, self.lstm_v1_out, num_layers=2, batch_first=True, dropout=0.5)
        self.lstm_v2 = nn.LSTM(self.lstm_v1_out, self.lstm_v1_out, num_layers=2, batch_first=True, dropout=0.5)

        self.lstm_a1_out = 64
        self.lstm_a1 = nn.LSTM(128, self.lstm_a1_out, num_layers=2, batch_first=True, dropout=0.5)
        self.lstm_a2 = nn.LSTM(self.lstm_a1_out, self.lstm_a1_out, num_layers=2, batch_first=True, dropout=0.5)  
        
        self.conv1_out = 4
        self.conv1_1 = nn.Sequential(  #64 1 120 64
            nn.Conv2d(in_channels=1, out_channels=self.conv1_out, kernel_size=3, stride=1, padding=2), 
            nn.BatchNorm2d(self.conv1_out, affine=False),
            nn.PReLU(self.conv1_out),  
            nn.MaxPool2d(2),
        ) 

        self.conv2_out = 8
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv1_out, out_channels=self.conv2_out, kernel_size=5, stride=3, padding=2),  
            nn.BatchNorm2d(self.conv2_out, affine=False),        
            nn.PReLU(self.conv2_out),
            nn.MaxPool2d(2),
        )   #64 64 15 16

        self.conv3_out = 16
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2_out, out_channels=self.conv3_out, kernel_size=7, stride=5, padding=2),
            nn.BatchNorm2d(self.conv3_out, affine=False),           
            nn.PReLU(self.conv3_out),
        )   #64 128 5 4
        
        self.fc_out1 = 64
        self.fc_out2 = 64
        self.fc = nn.Sequential(
            nn.Linear(self.fc_out2, 1),
        )
        self.step = 0
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
            if isinstance(m, nn.LSTM):
                nn.init.xavier_normal(m.all_weights[0][0])
                nn.init.xavier_normal(m.all_weights[0][1])
                nn.init.xavier_normal(m.all_weights[1][0])
                nn.init.xavier_normal(m.all_weights[1][1])
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight)


    def forward(self,vfeat,afeat):
        self.step = self.step + 1
        vfeat, (h_vn, c_vn) = self.lstm_v1(vfeat)  #64 120 32
        vfeat1, (h_vn, c_vn) = self.lstm_v2(vfeat)
        vfeat = vfeat + vfeat1
        afeat,  (h_an, c_an) = self.lstm_a1(afeat)  #64 120 32
        afeat1, (h_an, c_an) = self.lstm_a2(afeat)
        afeat = afeat + afeat1
        
        feat = torch.cat((vfeat, afeat), 2)               #64 120 64
        feat = feat.unsqueeze(1)                          #64 1 120 64

        feat = self.conv1_1(feat)
        feat = self.conv2_1(feat)
        feat = self.conv3_1(feat)        

        feat = feat.view(feat.size(0), -1)

        feat = self.fc(feat)
        feat = F.sigmoid(feat)
 
        feat = feat.view(-1)
        return feat
