import torch
from torch import nn

class Conv1dArch(nn.Module):
    def __init__(self, num_classes):
        super(Conv1dArch, self).__init__()

        # 1 x 2560
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=2, stride=1, padding=0),
            nn.LayerNorm([128, 2559]),
            nn.GELU())
        
        # 1 x 2559
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.LayerNorm([128, 2558]),
            nn.GELU(),
            nn.AvgPool1d(3, stride=2))

        # 1 x 1278
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([256, 1278]),
            nn.GELU(),
            nn.AvgPool1d(3, stride=2))
        
        # 1 x 638
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([256, 638]),
            nn.GELU(),
            nn.AvgPool1d(3, stride=2))
        
        # 1 x 318
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=0),
            nn.LayerNorm([512, 158]),
            nn.GELU(),
            nn.AvgPool1d(3, stride=3))
        
        # 1 x 52
        self.conv6 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([256, 52]),
            nn.GELU(),
            nn.AvgPool1d(3, stride=2))
        
        # 1 x 25
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=2, stride=1, padding=1),
            nn.LayerNorm([128, 26]),
            nn.GELU(),
            nn.AvgPool1d(3, stride=2))
        
        # 1 x 12
        self.conv8 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.LayerNorm([128, 11]),
            nn.GELU())
        
        self.fc_1 = nn.Linear(11*128, 512)
        self.act_1 = nn.LeakyReLU()
        self.fc_2 = nn.Linear(512, num_classes)
        self.act_2 = nn.Sigmoid()
    
    def forward(self, x):
        # expected conv1d input : minibatch_size x num_channel x width

        x = x.view(x.shape[0], 1,-1)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc_1(out)
        logit = self.act_1(logit)
        logit = self.fc_2(logit)
        logit = self.act_2(logit)

        return logit


class WholeConv1dArch(nn.Module):
    def __init__(self, model_1, model_2, conv1d):
        super(WholeConv1dArch, self).__init__()

        self.model_1 = model_1
        self.model_2 = model_2
        self.conv1d  = conv1d

    
    def forward(self, x):
        # expected conv1d input : minibatch_size x num_channel x width

        out_1 = self.model_1(x)
        out_2 = self.model_2(x)

        out = torch.cat((out_1, out_2), 1)
        logit = self.conv1d(out)

        return logit