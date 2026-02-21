import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# Codespaces helper: save plots
# -----------------------------
# Codespaces is usually headless, so figures won't pop up. We save them to ./outputs/.
def save_plot(filename: str, dpi: int = 150):
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", filename)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[Plot saved] {out_path}")
    plt.close()

import os

# 1. 下载并加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='/workspaces/ai101-starter/data_local/data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='/workspaces/ai101-starter/data_local/data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)

# 2. 定义神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 检查是否已有训练好的模型
model_path = "mnist_model.pth"

if os.path.exists(model_path):
    print("🔄 检测到已有模型，正在加载...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    print("✅ 模型加载成功！可以直接进行预测。")
else:
    print("⏳ 没有找到模型，开始训练...")

    # 训练神经网络
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")

    print("🎉 训练完成！")

    # 5. 训练完成后保存模型
    torch.save(model.state_dict(), model_path)
    print(f"💾 模型已保存至 {model_path}，下次可直接加载使用！")

# 6. 可视化预测结果
import numpy as np

def imshow(img):
    img = img * 0.5 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.axis('off')

# 获取测试集的 10 张图片
dataiter = iter(testloader)
images, labels = next(dataiter)

# 预测
images = images.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)  # 取最大值作为预测类别

# 显示图片和预测结果
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    imshow(images[i].cpu())  # 转回 CPU 并显示
    plt.title(f"预测: {predicted[i].item()}")
save_plot("phase1_predictions.png")