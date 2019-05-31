import torch
import torch.nn as nn
import torch.optim as optim

from datasets import TrainProjectionDataset, TestProjectionDataset
from useless_staff.model import Fast1D

def train(epochs=100):
    net = Fast1D(3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    trainset = TrainProjectionDataset("../out/projections/1120-2163-1-PB")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10)
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            x, y, labels = data
            optimizer.zero_grad()
            outputs = net(x, y)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('{} epoch loss: {}'.format(epoch + 1, running_loss / len(trainloader)))
    print('Finished Training')
    return net

def eval(net, path="../out/projections/1108-2162-1-PB"):
    testset = TestProjectionDataset(path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10)

    net.eval()
    results = [[0] * 3 for i in range(3)]
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        x, y, labels = data
        outputs = net(x, y)
        for i in range(x.size(0)):
            results[labels[i]][outputs[i]] += 1
    return results


if __name__ == "__main__":
    net = train(epochs=100)
    print("train", eval(net, path="../out/projections/1120-2163-1-PB"))
    print("test", eval(net))
