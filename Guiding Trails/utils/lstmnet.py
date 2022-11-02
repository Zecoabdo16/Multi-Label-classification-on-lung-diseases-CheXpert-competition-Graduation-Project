import torch
from torch import nn

class LSTMArch(nn.Module):
    def __init__(self, num_classes):
        super(LSTMArch, self).__init__()

        self.encoder  = nn.Sequential( 
                            nn.Linear(112*112, 5120), 
                            nn.GELU(),
                            nn.Linear(5120, 2048),
                            nn.GELU(),
                            nn.Linear(2048, 1024),
                            nn.GELU(),
                            )       

        self.lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=1, 
                            bias=True, batch_first=True, dropout=0, bidirectional=True, proj_size=0)
        
        self.top  = nn.Sequential( 
                            nn.Linear(512, num_classes), 
                            nn.Sigmoid()
                            )   

    def forward(self, x):
        # expected input : Sequence x minibatch_size x patch_height*patch_width x 1

        enc1 = self.encoder(x[:,0,].unsqueeze(axis=1))
        enc2 = self.encoder(x[:,1,].unsqueeze(axis=1))
        enc3 = self.encoder(x[:,2,].unsqueeze(axis=1))
        enc4 = self.encoder(x[:,3,].unsqueeze(axis=1))

        enc  = torch.cat((enc1, enc2, enc3, enc4), 1)

        self.lstm.flatten_parameters()
        out  = self.lstm(enc)[0][:, 3, :]

        logit = self.top(out)


        return logit