import torch
import torch.nn as nn
import torchvision
from model import CNNClassifier
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm import tqdm

def load_model():
    # load model
    model = CNNClassifier(1,32,10,3)
    model.load_state_dict(torch.load("savemodel.pth"))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev)
    model.eval()
    return model, dev

def predict(model,img,device):
    # model, device = load_model()
    # convert from PIL image to tensor
    img = torchvision.transforms.ToTensor()(img)
    img = torchvision.transforms.Normalize(0.5, 0.5)(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device)
    predicted = model(img).squeeze()
    # get top 5 predictions
    _, index = torch.topk(predicted,5)
    # get probability
    prob = torch.softmax(predicted, 0)[index]
    return index, prob

def train():
    batch_size = 32
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)]
    )
    trainset = MNIST('./mnist', train=True, download=True,transform=transform)
    testset = MNIST('./mnist', train=False, download=True,transform=transform)

    trainloader = DataLoader(trainset, shuffle=True, num_workers=2, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNClassifier(1,32,10,3).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in tqdm(range(10)):
        for idx, data in enumerate(trainloader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()

    model.eval()
    total_correct = 0
    for idx, data in enumerate(testloader):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        out = out.argmax(dim=1)
        total_correct += (out == y).to(int).sum()

    print((total_correct/len(testset)).item())
    torch.save(model.state_dict(), 'savemodel.pth')

