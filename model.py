import cv2
from rlsa import detect_blocks, compute_projections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1_x = nn.Conv1d(1, 50, 3)
        self.conv2_x = nn.Conv1d(50, 50, 3)
        self.conv3_x = nn.Conv1d(50, 50, 3)
        self.dp1_x = nn.Dropout(0.3)
        self.dp2_x = nn.Dropout(0.3)
        self.dp3_x = nn.Dropout(0.3)
        self.dp4_x = nn.Dropout(0.3)

        self.conv1_y = nn.Conv1d(1, 50, 3)
        self.conv2_y = nn.Conv1d(50, 50, 3)
        self.conv3_y = nn.Conv1d(50, 50, 3)
        self.dp1_y = nn.Dropout(0.3)
        self.dp2_y = nn.Dropout(0.3)
        self.dp3_y = nn.Dropout(0.3)
        self.dp4_conc = nn.Dropout(0.3)

        self.fc1 = nn.Linear(1000, 50)
        self.fc2 = nn.Linear(50, 3)

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

        conc = torch.cat((x, y), 1).flatten()
        print(conc.size())
        conc = F.relu(self.fc1(conc))
        conc = self.dp4_conc(conc)
        conc = F.softmax(self.fc2(conc))

        return conc


if __name__ == "__main__":
    image = cv2.imread('data/cropped_pdf_img/out_0.jpg')
    parts = detect_blocks(image, 300)
    x, y = compute_projections(parts[0])
    x = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
    y = torch.Tensor(y).unsqueeze(0).unsqueeze(0)
    model = Net()
    res = model(x, y)
    print(res.size())
    print(model)
