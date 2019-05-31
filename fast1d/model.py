import torch.nn as nn
import torch.nn.functional as F
import torch

class Fast1D(nn.Module):
    def __init__(self, outputs):
        super(Fast1D, self).__init__()
        self.conv1_x = nn.Conv1d(1, 50, 3)
        self.conv2_x = nn.Conv1d(50, 50, 3)
        self.conv3_x = nn.Conv1d(50, 50, 3)
        self.dp1_x = nn.Dropout(0.1)
        self.dp2_x = nn.Dropout(0.1)
        self.dp3_x = nn.Dropout(0.1)

        self.conv1_y = nn.Conv1d(1, 50, 3)
        self.conv2_y = nn.Conv1d(50, 50, 3)
        self.conv3_y = nn.Conv1d(50, 50, 3)
        self.dp1_y = nn.Dropout(0.1)
        self.dp2_y = nn.Dropout(0.1)
        self.dp3_y = nn.Dropout(0.1)

        self.dp4_conc = nn.Dropout(0.3)

        self.fc1 = nn.Linear(1000, 50)
        self.fc2 = nn.Linear(50, outputs)

    def forward(self, x, y):
        # Max pooling over a (2, 2) window
        x = F.max_pool1d(F.relu(self.conv1_x(x)), 2)
        x = self.dp1_x(x)
        x = F.max_pool1d(F.relu(self.conv2_x(x)), 2)
        x = self.dp2_x(x)
        x = F.max_pool1d(F.relu(self.conv3_x(x)), 2)
        x = self.dp3_x(x)

        y = F.max_pool1d(F.relu(self.conv1_y(y)), 2)
        y = self.dp1_y(y)
        y = F.max_pool1d(F.relu(self.conv2_y(y)), 2)
        y = self.dp2_y(y)
        y = F.max_pool1d(F.relu(self.conv3_y(y)), 2)
        y = self.dp3_y(y)

        conc = torch.cat((x, y), 1)
        conc = conc.reshape(conc.size(0), -1)
        conc = F.relu(self.fc1(conc))
        conc = self.dp4_conc(conc)
        conc = self.fc2(conc)
        if self.training == False:
            _, conc = torch.max(conc, 1)
        return conc
