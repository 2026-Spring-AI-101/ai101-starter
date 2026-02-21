# -*- coding: utf-8 -*-
"""
MNIST 跨模态语义对齐（共享向量空间）示例
------------------------------------------------
把你原来的 MNIST 分类（image -> digit）升级为：
1) Image -> Text: 看图找“文本标签”（0-9）
2) Text -> Image: 输入“数字文本”检索最匹配的图片（Top-K）

核心思想：
- 图像编码器 f_img 输出 D 维 embedding
- 文本编码器 f_txt（这里用最简单的 10 类 embedding lookup）输出 D 维 embedding
- 用余弦相似度（向量点积）做匹配；用交叉熵训练“对齐”

运行方式：
    python mnist_shared_embedding.py

第一次运行会训练并保存模型；后续运行会自动加载。
"""

import os
import numpy as np
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

# -----------------------------
# 1) 数据加载
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 把像素约束到 [-1, 1]，与原脚本保持一致
])

trainset = torchvision.datasets.MNIST(root="/workspaces/ai101-starter/data_local/data", train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root="/workspaces/ai101-starter/data_local/data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=256, shuffle=False)


# -----------------------------
# 2) 模型：共享向量空间（双编码器）
# -----------------------------
class ImageEncoder(nn.Module):
    """把图片编码成 D 维向量"""
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, emb_dim)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # (B, D)
        return x


class SharedEmbeddingAligner(nn.Module):
    """
    - 图像编码器：Image -> embedding (B, D)
    - 文本编码器：digit id (0-9) -> embedding (10, D)
    - 相似度：cosine / dot product
    """
    def __init__(self, emb_dim: int = 32, num_text_tokens: int = 10, temperature: float = 0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(emb_dim=emb_dim)
        self.text_embed = nn.Embedding(num_text_tokens, emb_dim)  # “文本”这里简化成 10 个 token
        self.temperature = temperature

    @staticmethod
    def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        z = self.image_encoder(images)
        return self.l2_normalize(z)

    def encode_text_all(self) -> torch.Tensor:
        # 直接取 embedding table 的所有 token（0-9）
        z = self.text_embed.weight  # (10, D)
        return self.l2_normalize(z)

    def logits_image_to_text(self, images: torch.Tensor) -> torch.Tensor:
        """
        返回 (B, 10) 的相似度 logits：每张图对 10 个文本 token 的匹配程度
        """
        img_z = self.encode_image(images)          # (B, D)
        txt_z = self.encode_text_all()             # (10, D)
        logits = (img_z @ txt_z.t()) / self.temperature
        return logits  # (B, 10)


# -----------------------------
# 3) 训练 / 载入
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SharedEmbeddingAligner(emb_dim=32, temperature=0.07).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model_path = "mnist_shared_embedding.pth"

def evaluate_accuracy(model: SharedEmbeddingAligner, dataloader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model.logits_image_to_text(images)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


if os.path.exists(model_path):
    print("🔄 检测到已有模型，正在加载...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ 模型加载成功！")
else:
    print("⏳ 没有找到模型，开始训练（共享向量空间对齐）...")

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model.logits_image_to_text(images)          # (B, 10)
            loss = criterion(logits, labels)                     # 让正确数字的文本向量与图像向量更接近
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_acc = evaluate_accuracy(model, trainloader)
        test_acc = evaluate_accuracy(model, testloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(trainloader):.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    torch.save(model.state_dict(), model_path)
    print(f"💾 模型已保存至 {model_path}")


# -----------------------------
# 4) 可视化：Image -> Text（像分类一样预测）
# -----------------------------
def imshow(img_tensor_1x28x28):
    # 反归一化到 [0, 1]
    img = img_tensor_1x28x28 * 0.5 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(npimg.squeeze(0), cmap="gray")
    plt.axis("off")

model.eval()

# 取测试集前 10 张展示
sample_images = torch.stack([testset[i][0] for i in range(10)], dim=0).to(device)  # (10,1,28,28)
with torch.no_grad():
    logits = model.logits_image_to_text(sample_images)
    predicted = torch.argmax(logits, dim=1).cpu().numpy()

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    imshow(sample_images[i])
    plt.title(f"预测: {predicted[i]}")
plt.suptitle("Image -> Text（共享向量空间：看图预测数字）", fontsize=12)
plt.tight_layout()
save_plot("phase2_image_to_text.png")


# -----------------------------
# 5) 可视化：Text -> Image 检索（输入数字，返回 Top-K 最像的图片）
# -----------------------------
def build_test_image_embeddings(model: SharedEmbeddingAligner, dataloader):
    model.eval()
    all_embs = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            emb = model.encode_image(images)  # (B, D)
            all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)  # (N, D)

print("📦 正在为测试集构建图像 embedding（用于检索）...")
test_image_embs = build_test_image_embeddings(model, testloader)  # (10000, D)

# 选择一个查询数字（你也可以改成 input() 交互）
query_digit = 7
with torch.no_grad():
    # 取对应文本 embedding（这里文本就是 digit id）
    txt_emb_all = model.encode_text_all().cpu()  # (10, D)
    query_emb = txt_emb_all[query_digit:query_digit+1]  # (1, D)

# 相似度：每张图与 query_emb 的点积（等价余弦相似度，因为都归一化了）
sims = (test_image_embs @ query_emb.t()).squeeze(1)  # (N,)
topk = 10
topk_vals, topk_idx = torch.topk(sims, k=topk)

# 取出 Top-K 图片展示
retrieved_imgs = [testset[i][0] for i in topk_idx.tolist()]
retrieved_labels = [testset[i][1] for i in topk_idx.tolist()]

plt.figure(figsize=(10, 3))
for i in range(topk):
    plt.subplot(2, 5, i + 1)
    imshow(retrieved_imgs[i])
    plt.title(f"lbl:{retrieved_labels[i]}")
plt.suptitle(f'Text -> Image 检索：输入 "{query_digit}" 返回 Top-{topk}', fontsize=12)
plt.tight_layout()
save_plot(f"phase2_text_to_image_query{query_digit}_top{topk}.png")

print("✅ 完成：你已经得到一个最小可运行的跨模态语义对齐（共享向量空间）实验。")