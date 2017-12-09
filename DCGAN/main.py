# pytorch imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils

# package imports
import os, time

# file imports
import util

# loads MNIST dataset
transform = transforms.Compose(
    [transforms.Scale(util.img_size), # change image size to 64x64
     transforms.ToTensor(), # PILImage to Tensor
     transforms.Normalize((0.1307,), (0.3081,)) # normalize image
    ])

# dataset is under ./data
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=util.batch_size, shuffle=True)

# Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            # (1 x 1) x 100 -> 4 x 4 x 1024
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # 4 x 4 x 1024 -> 8 x 8 x 512
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 8 x 8 x 512 -> 16 x 16 x 256
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 16 x 16 x 256 -> 32 x 32 x 128
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 32 x 32 x 128 -> 64 x 64 x 1
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        ) 

    def forward(self, input):
        return self.main(input)

# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(
            # 64 x 64 x 1 -> 32 x 32 x 128
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 32 x 128 -> 16 x 16 x 256
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 16 x 256 -> 8 x 8 x 512
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 x 8 x 512 -> 4 x 4 x 1024
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 x 4 x 1024 -> 1 x 1 x 1
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# weight initialization for Conv2d and ConvTranspose2d
# mean: 0.0, std: 0.02
def init_weight(m):
   if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
       m.weight.data.normal_(util.mean, util.std)


# network initialization
netG = generator()
netG.apply(init_weight)

netD = discriminator()
netD.apply(init_weight)

# Loss criterion
criterion = nn.BCELoss()

# optimizer initialization
optimizerG = optim.Adam(netG.parameters(), lr=util.lr, betas=(util.beta, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=util.lr, betas=(util.beta, 0.999))

# fixed z
fixed_z = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
fixed_z = Variable(fixed_z)

# saving folder for image generated from fixed_z 
if not os.path.isdir("MNIST_fixed_results"):
    os.mkdir("MNIST_fixed_results")

# train
for epoch in range(util.train_epoch):
    G_losses = []
    D_losses = []
    print("%d epoch" % (epoch))
    for i, (inputs, _) in enumerate(trainloader):
        print("%d/%d" % (i, len(trainloader)))
        # train discriminator
        netD.zero_grad()
        mini_batch = inputs.size()[0]
        real_label = torch.ones(mini_batch)
        fake_label = torch.zeros(mini_batch)

        inputs, real_label, fake_label = Variable(inputs), Variable(real_label), Variable(fake_label)

        # train with real image        
        outputs = netD(inputs)
        D_real_loss = criterion(outputs, real_label)

        # train with fake image
        z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z = Variable(z)
        outputs = netD(netG(z))
        D_fake_loss = criterion(outputs, fake_label)

        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        optimizerD.step()

        D_losses.append(D_train_loss.data[0])

        # train generator
        netG.zero_grad()

        z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z = Variable(z)

        outputs = netD(netG(z))
        G_train_loss = criterion(outputs, real_label)
        G_train_loss.backward()
        optimizerG.step()

        G_losses.append(G_train_loss.data[0])

    # print losses
    print("G_loss: %.3f, D_loss: %.3f" % (torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(D_losses))))

    # generate image
    outputs = netG(fixed_z)
    vutils.save_image(outputs.data, "MNIST_fixed_results/fake_image_epoch_%03d.png" % (epoch), normalize=True)
    

