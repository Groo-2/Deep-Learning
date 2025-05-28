import tqdm
import torch
import torch.nn as nn
from torchinfo import summary

from torchvision.models import densenet121
from torchvision.datasets import CIFAR100  # ✅ CIFAR10 → CIFAR100
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data import DataLoader
from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 사전 학습된 DenseNet121 모델 준비
model = densenet121(pretrained=True)
num_output = 100  # ✅ CIFAR-100 클래스 수로 변경
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_output)
print(model)

model.to(device)
summary(model, input_size=(32, 3, 224, 224))  # torchinfo용

# ✅ CIFAR-100 평균/표준편차
transforms = Compose([
    Resize(224),
    RandomCrop((224, 224), padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2761))
])

# ✅ CIFAR-100 데이터 로더
training_data = CIFAR100(root="./data", train=True, download=True, transform=transforms)
test_data = CIFAR100(root="./data", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ✅ fc layer만 학습하고 나머지는 freeze
params_to_update = []
for name, param in model.named_parameters():
    if 'classifier' in name:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

# ✅ 옵티마이저
lr = 1e-4
optim = Adam(params_to_update, lr=lr)

# ✅ 학습 루프
for epoch in range(5):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()
        preds = model(data.to(device))
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()
        iterator.set_description(f"epoch:{epoch+1:05d} loss:{loss.item():05.2f}")

# ✅ 모델 저장
torch.save(model.state_dict(), "CIFAR100_pretrained_DenseNet.pth")

# ✅ 모델 평가
model.load_state_dict(torch.load("CIFAR100_pretrained_DenseNet.pth", map_location=device))
model.eval()
num_corr = 0
with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        _, preds = output.data.max(1)
        num_corr += preds.eq(label.to(device)).sum().item()

print(f"Accuracy: {num_corr / len(test_data):.4f}")
