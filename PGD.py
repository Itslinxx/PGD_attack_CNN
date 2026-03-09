

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from google.colab import files
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4 * 4 * 10, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

files.upload()

epsilons = [.1, .2, .3]
alphas = [0.01]
BATCH_SIZE = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name()}")
model = torch.load("my_cnn_model.pth/my_cnn_model.pth", weights_only=False, map_location=device)
model.eval()
torch.manual_seed(42)

def PGD_attack(model, image, label, epsilon, alpha, num_steps):

  label = label.to(device)
  perturbed_image = image.clone().detach().to(device)
  perturbed_image.requires_grad=True

  for i in range(num_steps):
    output = model(perturbed_image)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()

    sign_data_grad = perturbed_image.grad.sign()
    with torch.no_grad():
      perturbed_image = perturbed_image + alpha*sign_data_grad
      perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)
      perturbed_image = torch.clamp(perturbed_image, 0, 1)
    perturbed_image.requires_grad = True
  return perturbed_image

def test (model, device, test_loader, epsilon, alpha, num_steps):
  correct = total = 0
  for data, target in tqdm(test_loader, desc='test: '):
    data = data.to(device)
    target = target.to(device)
    data.requires_grad = True
    output = model(data)
    init_pred = output.max(1, keepdim = True)[1]
    if init_pred.item() != target.item():
      continue
    perturbed_data = PGD_attack(model, data, target, epsilon, alpha, num_steps)
    output = model (perturbed_data)

    final_pred = output.max(1, keepdim=True)[1]
    if final_pred.item() == target.item():
      correct+=1
    total += 1
  final_accuracy = correct/total if total > 0 else 0
  print(f"Epsilon: {epsilon:.2f} Alpha: {alpha:.2f} Final accuracy: {final_accuracy:.3f}")
  return final_accuracy

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

accuracies = []
examples = []

for eps in epsilons:
  for alp in alphas:
    acc = test(model, device, test_loader, eps, alp, 30)
    accuracies.append(acc)
