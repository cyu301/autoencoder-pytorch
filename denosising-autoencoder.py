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

# linear autoencoder
class DenoisingAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoEncoder, self).__init__()
        
        # encoder fully connected
        self.enfc1 = nn.Linear(28*28,256)
        self.enfc2 = nn.Linear(256,128)
        self.enfc3 = nn.Linear(128,64)
        self.enfc4 = nn.Linear(64,32)
        self.enfc5 = nn.Linear(32,16)
        self.enfc6 = nn.Linear(16,8)
        
        # decoder fully connected
        self.defc1 = nn.Linear(8,16)
        self.defc2 = nn.Linear(16,32)
        self.defc3 = nn.Linear(32,64)
        self.defc4 = nn.Linear(64,128)
        self.defc5 = nn.Linear(128,256)
        self.defc6 = nn.Linear(256,28*28)
        
    def encoder(self, x):
        h1 = F.relu(self.enfc1(x))
        h2 = F.relu(self.enfc2(h1))
        h3 = F.relu(self.enfc3(h2))
        h4 = F.relu(self.enfc4(h3))
        h5 = F.relu(self.enfc5(h4))
        return F.relu(self.enfc6(h5))
    
    def decoder(self, z):
        h7 = F.relu(self.defc1(z))
        h8 = F.relu(self.defc2(h7))
        h9 = F.relu(self.defc3(h8))
        h10 = F.relu(self.defc4(h9))
        h11 = F.relu(self.defc5(h10))
        return torch.sigmoid(self.defc6(h11))
        

    def forward(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        return z, decoded

def add_noise(img, factor):
    noise = torch.randn(img.size()) * factor
    noisy_img = img + noise
    return noisy_img

def train(epoch):
    autoencoder.train()
    avg_loss = 0
    for step, (x, label) in enumerate(trainloader):
        noisy_x = add_noise(x, 0.3) 
        noisy_x = noisy_x.view(-1, 28*28).to(device)
        y = x.view(-1, 28*28).to(device)

        encoded, decoded = autoencoder(noisy_x)

        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        if (step % 200 == 0) and step!=0:
            print(f'[Epoch {epoch}][{step*len(x)}/{len(trainloader.dataset)}] loss: {avg_loss/step:.6f}')
    return avg_loss / len(trainloader)

def test(epoch):
    autoencoder.eval()
    running_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for _, (test, _) in enumerate(testloader):
            #add noise
            noisy_test = add_noise(test, 0.3)
            noisy_test = noisy_test.view(-1, 28*28).to(device)
            origin = test.view(-1, 28*28).to(device)
            #autoencode
            
            _, recovered_test = autoencoder(noisy_test)
            running_loss += criterion(recovered_test, origin)
        print(f'[Epoch {epoch}] Test Loss:{running_loss:.6}')
    return 

def Visualize():
    rnd = np.random.randint(len(testset))
    #get data and normalize
    sample_data = testset.data[rnd].view(-1, 28*28)
    sample_data = sample_data.type(torch.FloatTensor)/255.
    original_x = sample_data[0]
    #add noise
    noisy_x = add_noise(original_x, 0.3).to(device)
    #autoencode
    with torch.no_grad():
        _, recovered_x = autoencoder(noisy_x)
    # back to cpu
    noisy_x = noisy_x.to('cpu')
    recovered_x = recovered_x.to('cpu')
    
    #plot
    f, a = plt.subplots(1, 3, figsize=(6, 3))
    original_img = np.reshape(sample_data[0].data.numpy(), (28, 28))
    noisy_img = np.reshape(noisy_x.data.numpy(), (28, 28))
    recovered_img = np.reshape(recovered_x.data.numpy(), (28, 28))
    a[0].set_title('Original')
    a[0].imshow(original_img, cmap='gray')
    a[1].set_title('Noisy')
    a[1].imshow(noisy_img, cmap='gray')
    a[2].set_title('Recovered')
    a[2].imshow(recovered_img, cmap='gray')
    plt.show()
    print(f'Test Item: {rnd}. Real label: {testset[rnd][1]}')


if __name__ == '__main__':

    autoencoder = DenoisingAutoEncoder()
    autoencoder = autoencoder.to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    EPOCH = 16
    start = time.time()
    for epoch in range(1, EPOCH+1):
        train(epoch)
        test(epoch)
    end = time.time()
    print(f'Total train and test time: {end-start}:.4')
    
    for i in range(3):
        Visualize()