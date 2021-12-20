# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import time

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device： {device}')

mnist_root = '../data/mnist'
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),]) 
trainset = torchvision.datasets.MNIST(root=mnist_root, download=False, train=True, transform=transform)
testset = torchvision.datasets.MNIST(root=mnist_root, download=False, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 64)
        self.fc31 = nn.Linear(64, 20)
        self.fc32 = nn.Linear(64, 20)
        self.fc4 = nn.Linear(20, 64)
        self.fc5 = nn.Linear(64, 400)
        self.fc6 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(trainloader):
        optimizer.zero_grad()
        data = data.to(device)
        recon_batch, mu, logvar, z = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 200 == 0 and batch_idx!=0:
            print(f'[Epoch: {epoch}][{batch_idx * len(data)}/{len(trainloader.dataset)}'\
                  f'({100. * batch_idx / len(trainloader):.0f}%)] Train Loss: {loss.item() / len(data):.6f}')

    print(f'[Epoch: {epoch}] Average loss: {train_loss / len(trainloader.dataset):.4f}')

def SampleEncode(rnd):
    sample = testset.data[rnd]/255.
    sample = sample.to(device)
    with torch.no_grad():
        mu, logvar = vae.encode(sample.view(-1, 784))
        code = vae.reparameterize(mu, logvar)
        code = code.to('cpu')
    return code.data.numpy()

def MNISTInterpo(c1, c2):
    interpo = []
    for i in range(1,8):
        interpo.append(c1+(c2-c1)*i/8)
    return interpo

def SampleDecode(interpo):
    interpo_img = []
    for code in interpo:
        with torch.no_grad():
            img = vae.decode(transform(code).to(device)).to('cpu')
            interpo_img.append(img.reshape((28,28)))
    return interpo_img

def CreateLabelDict():
    test_labels = {}
    for i in range(len(testset)):
        label = testset[i][1]
        if label in test_labels:
            test_labels[label].append(i)
        else:
            test_labels[label] = [i]
    return test_labels

def SameDigitPlot():
    f, a = plt.subplots(10, 9, figsize=(20, 20))
    for digit in range(10):
        rnd = np.random.randint(len(test_labels[digit]),size = 2)
        sample = [test_labels[digit][rnd[0]],test_labels[digit][rnd[1]]]
        code1 = SampleEncode(sample[0])
        code2 = SampleEncode(sample[1])
        interpo = MNISTInterpo(code1,code2)
        imgs = SampleDecode(interpo)
        a[digit][0].imshow(testset.data.numpy()[sample[0]],cmap = 'gray')
        for i in range(1,8):
            a[digit][i].imshow(imgs[i-1],cmap = 'gray')
        a[digit][8].imshow(testset.data.numpy()[sample[1]],cmap = 'gray') 

def DifferentDigitPlot():
    f, a = plt.subplots(10, 9, figsize=(20, 20))
    for digit in range(10):
        sample = np.random.randint(len(testset.data),size = 2)
        code1 = SampleEncode(sample[0])
        code2 = SampleEncode(sample[1])
        interpo = MNISTInterpo(code1,code2)
        imgs = SampleDecode(interpo)
        a[digit][0].imshow(testset.data.numpy()[sample[0]],cmap = 'gray')
        for i in range(1,8):
            a[digit][i].imshow(imgs[i-1],cmap = 'gray')
        a[digit][8].imshow(testset.data.numpy()[sample[1]],cmap = 'gray') 

if __name__ == '__main__':
    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.005)
    
    EPOCH = 16
    start = time.time()
    for epoch in range(1, EPOCH + 1):
        train(epoch)
    end = time.time()
    print(f'Total train time: {end-start:.4}')
    test_labels = CreateLabelDict()
    
    SameDigitPlot()
    DifferentDigitPlot()