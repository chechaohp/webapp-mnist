import torch
import torchvision
from PIL import Image
from model import CNNClassifier

# load model
print("load model")
model = CNNClassifier(1,32,10,3)
model.load_state_dict(torch.load("savemodel.pth"))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(dev)

print("load image")
img = Image.open("test.png")
img = torchvision.transforms.ToTensor()(img)
img = torchvision.transforms.Normalize(0.5, 0.5)(img)
img = torch.unsqueeze(img, dim=0)
img = img.to(dev)
predicted = model(img).squeeze()
# get top 5 predictions
_, index = torch.topk(predicted,5)
# get probability
prob = torch.softmax(predicted, 0)[index]


