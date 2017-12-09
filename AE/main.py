# imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# Network Definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,50)
        self.fc2 = nn.Linear(50,784)

    def forward(self, x):
        x = self.flatten(x)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

    def flatten(self, x):
        # torch.Size([1, 28, 28]) -> torch.Size([1, 784])
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return x.view(-1, num_features)

net = Net()

# loading MNIST datasets
transform = transforms.Compose(
    [transforms.ToTensor(), # PILImage to Tensor
     transforms.Normalize((0.1307,), (0.3081,)) # Normalization
     ])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train
n_epochs = 200
for epoch in range(n_epochs):
    print('EPOCH: %d' %epoch)
    running_loss = 0.0
    for i, (inputs, _) in enumerate(trainloader):
        inputs = Variable(inputs)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, net.flatten(inputs))
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if i % 100 == 99:
            print(i + 1)
            print('loss: %f' %(running_loss / 100))
            running_loss = 0.0

# Test
loss_sum = 0
total = 0
for (images, labels) in testloader:
    images = Variable(images)
    outputs = net(images)
    total += labels.size(0)
    loss_sum += 100 * criterion(outputs, net.flatten(images)).data[0]

print('Test Loss: %f' %(loss_sum / total))
