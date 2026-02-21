# -*- coding: utf-8 -*-
"""
MNIST 跨模态语义对齐（共享向量空间） + 解码器
=============================================
路线2：对比学习对齐 embedding + 每个通道自带 decoder

架构:
  - Image Encoder (CNN): 图像 -> embedding
  - Image Decoder (CNN): embedding -> 图像
  - Text Encoder: digit id (0-9) -> embedding (nn.Embedding lookup)
  - 对齐: 图像 embedding 与文本 embedding 在归一化空间中对齐

三个损失函数联合训练:
  (1) 对齐损失: CrossEntropy(image_emb @ text_emb.T, label) — 让同类对齐
  (2) 图像重建损失: MSE(Decoder(Encoder(image)), image) — 让 encoder 保留重建信息
  (3) 文本解码损失: MSE(Decoder(TextEmbed(label)), image) — ★ 关键！
      让 decoder 学会从文本 embedding 解码图像，否则跨模态生成不可用

训练完成后可以:
  - Image -> Text: 图像编码后与10个文本embedding比较，取最相似的 = 看图识数字
  - Text -> Image: 文本embedding直接送入decoder = 从数字生成图像（条件生成）
  - Image -> Image: 图像编码再解码 = 自编码器重建

评估方式:
  - 分类准确率: Image -> Text 的识别准确率
  - 重建 MSE: 图像自编码重建误差
  - 外部分类器准确率: 生成的图像能否被独立分类器识别 (客观指标，避免自评偏差)
  - 外部分类器置信度: 独立分类器对正确类别的 softmax 概率 (越高越好)

运行方式:
    python mnist_shared_embedding_with_decoder.py
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

# =====================================================================
# 1) 超参数
# =====================================================================
EMB_DIM    = 128       # embedding 维度 (越大信息瓶颈越宽，重建越好)
EPOCHS     = 20        # 训练轮次 (重建比分类需要更多轮)
LR         = 1e-3
TEMP       = 0.07      # 余弦相似度的温度参数
LAMBDA_RECON      = 1.0   # 图像重建损失权重
LAMBDA_TEXT_RECON  = 1.0   # 文本解码损失权重
MODEL_PATH = "mnist_shared_with_decoder.pth"


# =====================================================================
# 2) 数据加载
# =====================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 像素归一化到 [-1, 1]
])

trainset = torchvision.datasets.MNIST(root="/workspaces/ai101-starter/data_local/data", train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root="/workspaces/ai101-starter/data_local/data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,  shuffle=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=256, shuffle=False)


# =====================================================================
# 3) 模型定义: CNN 编码器 + CNN 解码器 + 文本 Embedding
# =====================================================================
class ImageEncoder(nn.Module):
    """
    CNN 编码器: (B, 1, 28, 28) -> (B, emb_dim)
    通过 stride=2 的卷积逐步下采样: 28 -> 14 -> 7 -> 4
    BatchNorm 加速收敛，ReLU 引入非线性
    """
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,  32, 3, stride=2, padding=1),  # (B,32,14,14)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (B,64,7,7)
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # (B,64,4,4)
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 4 * 4, emb_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # (B, 1024)
        return self.fc(x)           # (B, emb_dim)


class ImageDecoder(nn.Module):
    """
    CNN 解码器: (B, emb_dim) -> (B, 1, 28, 28)
    通过 ConvTranspose2d 逐步上采样: 4 -> 7 -> 14 -> 28
    最后用 Tanh 输出 [-1, 1]，与输入数据范围对齐
    """
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 64 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=0),  # (B,64,7,7)
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # (B,32,14,14)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32,  1, 3, stride=2, padding=1, output_padding=1),  # (B,1,28,28)
            nn.Tanh(),
        )

    def forward(self, z):
        z = self.fc(z)               # (B, 1024)
        z = z.view(-1, 64, 4, 4)     # (B, 64, 4, 4)
        return self.deconv(z)         # (B, 1, 28, 28)


class SharedSpaceWithDecoder(nn.Module):
    """
    完整模型: 图像编码器 + 图像解码器 + 文本Embedding
    - 对齐在 L2 归一化后的空间完成 (余弦相似度)
    - 重建在 raw embedding 空间完成 (保留完整信息)
    """
    def __init__(self, emb_dim: int = 128, num_text_tokens: int = 10, temperature: float = 0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(emb_dim=emb_dim)
        self.image_decoder = ImageDecoder(emb_dim=emb_dim)
        self.text_embed = nn.Embedding(num_text_tokens, emb_dim)
        self.temperature = temperature

    @staticmethod
    def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """图像 -> L2 归一化后的 embedding (用于对齐/分类)"""
        return self.l2_normalize(self.image_encoder(images))

    def encode_text_all(self) -> torch.Tensor:
        """获取全部 10 个文本 embedding (L2 归一化)"""
        return self.l2_normalize(self.text_embed.weight)

    def logits_image_to_text(self, images: torch.Tensor) -> torch.Tensor:
        """Image -> Text: 返回 (B, 10) 的相似度 logits"""
        img_z = self.encode_image(images)     # (B, D)
        txt_z = self.encode_text_all()        # (10, D)
        return (img_z @ txt_z.t()) / self.temperature

    def reconstruct(self, images: torch.Tensor):
        """Image -> Encoder -> Decoder -> Image (自编码重建)"""
        z_raw = self.image_encoder(images)      # (B, D) 未归一化
        return self.image_decoder(z_raw)        # (B, 1, 28, 28)

    def decode_from_text(self, digit_id: torch.Tensor):
        """Text -> Image: 文本 embedding 直接送入 decoder 生成图像"""
        z = self.text_embed(digit_id)  # (B, D)
        return self.image_decoder(z)   # (B, 1, 28, 28)


# =====================================================================
# 4) 独立分类器 (用于客观评估生成图像质量)
# =====================================================================
# 为什么需要独立分类器?
# 如果用同一个模型的 logits_image_to_text 来评判自己生成的图像，
# 存在"自评偏差"——模型可能对自己的输出特别"宽容"。
# 独立分类器是完全独立训练的，能更客观地评判生成图像是否像对应数字。

class IndependentClassifier(nn.Module):
    """简单的 MLP 分类器，独立于主模型"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)


# =====================================================================
# 5) 训练 / 载入
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

model = SharedSpaceWithDecoder(emb_dim=EMB_DIM, temperature=TEMP).to(device)

criterion_align = nn.CrossEntropyLoss()
criterion_recon = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def evaluate_accuracy(model, dataloader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            pred = model.logits_image_to_text(images).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


if os.path.exists(MODEL_PATH):
    print(f"🔄 检测到已有模型，正在加载 {MODEL_PATH} ...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ 模型加载成功！")
else:
    print("⏳ 没有找到模型，开始训练（对齐 + 图像重建 + 文本解码）...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = running_align = running_recon = running_text = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # (1) 对齐损失: 图像 embedding 与对应文本 embedding 对齐
            logits = model.logits_image_to_text(images)
            loss_align = criterion_align(logits, labels)

            # (2) 图像重建损失: Image -> Encoder -> Decoder -> Image
            recon = model.reconstruct(images)
            loss_recon = criterion_recon(recon, images)

            # (3) ★ 文本解码损失: TextEmbed(label) -> Decoder -> Image
            #     这是让跨模态生成可用的关键！
            #     如果不加这个损失，decoder 从未见过文本 embedding，
            #     推理时 Text->Image 就会生成垃圾。
            text_recon = model.decode_from_text(labels)
            loss_text_recon = criterion_recon(text_recon, images)

            loss = loss_align + LAMBDA_RECON * loss_recon + LAMBDA_TEXT_RECON * loss_text_recon
            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            running_align += loss_align.item()
            running_recon += loss_recon.item()
            running_text  += loss_text_recon.item()

        scheduler.step()
        n = len(trainloader)
        cur_lr = scheduler.get_last_lr()[0]
        test_acc = evaluate_accuracy(model, testloader)
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | lr={cur_lr:.6f} | loss={running_loss/n:.4f} "
              f"(align={running_align/n:.4f} img_recon={running_recon/n:.4f} txt_recon={running_text/n:.4f}) "
              f"| test_acc={test_acc*100:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 模型已保存至 {MODEL_PATH}")


# =====================================================================
# 6) 训练独立分类器 (作为外部裁判)
# =====================================================================
print("\n📊 训练独立分类器（用于客观评估生成质量）...")
ext_clf = IndependentClassifier().to(device)
ext_opt = optim.Adam(ext_clf.parameters(), lr=1e-3)
ext_ce = nn.CrossEntropyLoss()
for ep in range(5):
    ext_clf.train()
    for imgs, lbls in trainloader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        ext_opt.zero_grad()
        loss = ext_ce(ext_clf(imgs), lbls)
        loss.backward(); ext_opt.step()
ext_clf.eval()
ext_correct = ext_total = 0
with torch.no_grad():
    for imgs, lbls in testloader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        ext_correct += (ext_clf(imgs).argmax(1) == lbls).sum().item()
        ext_total += lbls.numel()
print(f"   独立分类器准确率: {ext_correct/ext_total*100:.2f}%\n")


# =====================================================================
# 7) 综合评估
# =====================================================================
model.eval()

# (a) Image -> Text 分类准确率
test_acc = evaluate_accuracy(model, testloader)

# (b) Image 重建 MSE
sample_images = torch.stack([testset[i][0] for i in range(10)]).to(device)
with torch.no_grad():
    recon_images = model.reconstruct(sample_images)
    recon_mse = nn.functional.mse_loss(recon_images, sample_images).item()

# (c) Text -> Image 生成质量 (独立分类器评估)
#     对每个数字 d (0-9):
#       1. text_embed(d) -> decoder -> 生成图像
#       2. 生成图像 -> 独立分类器 -> 预测 & 置信度
ext_gen_correct = 0
ext_gen_conf_sum = 0.0
gen_results = {}  # 存储每个数字的详细结果

with torch.no_grad():
    for d in range(10):
        digit_id = torch.tensor([d], device=device)
        gen_img = model.decode_from_text(digit_id)

        # 独立分类器判断
        ext_logits = ext_clf(gen_img)
        ext_pred = ext_logits.argmax(1).item()
        ext_probs = torch.softmax(ext_logits, dim=1)
        ext_conf = ext_probs[0, d].item()  # P(correct class)

        if ext_pred == d:
            ext_gen_correct += 1
        ext_gen_conf_sum += ext_conf

        gen_results[d] = {
            'image': gen_img[0].cpu(),
            'pred': ext_pred,
            'conf': ext_conf,
            'correct': ext_pred == d,
        }

ext_gen_acc = ext_gen_correct / 10
ext_gen_conf = ext_gen_conf_sum / 10

# 打印评估结果
print(f"{'='*60}")
print(f"  综合评估结果")
print(f"{'='*60}")
print(f"  Image->Text 分类准确率:       {test_acc*100:.2f}%")
print(f"  Image 重建 MSE:               {recon_mse:.4f}")
print(f"  Text->Image 外部分类器准确率: {ext_gen_acc*100:.0f}%  (10个数字中{ext_gen_correct}个被正确识别)")
print(f"  Text->Image 外部分类器置信度: {ext_gen_conf*100:.1f}% (对正确类别的平均概率)")
print(f"{'='*60}")
print(f"\n  逐数字生成质量:")
for d in range(10):
    r = gen_results[d]
    status = "✓" if r['correct'] else "✗"
    print(f"    数字 {d}: 预测={r['pred']} {status}  置信度={r['conf']*100:.1f}%")


# =====================================================================
# 8) 可视化
# =====================================================================
def to_img(tensor):
    """[-1, 1] -> [0, 1] 的灰度图"""
    return tensor.detach().cpu().squeeze() * 0.5 + 0.5

# --- (a) Image 重建对比 ---
fig, axes = plt.subplots(2, 10, figsize=(12, 2.5))
for i in range(10):
    axes[0, i].imshow(to_img(sample_images[i]), cmap='gray'); axes[0, i].axis('off')
    axes[1, i].imshow(to_img(recon_images[i]),  cmap='gray'); axes[1, i].axis('off')
axes[0, 0].text(-0.15, 0.5, 'Original', transform=axes[0, 0].transAxes,
                fontsize=9, va='center', ha='right', fontweight='bold')
axes[1, 0].text(-0.15, 0.5, f'Recon\n(MSE={recon_mse:.4f})', transform=axes[1, 0].transAxes,
                fontsize=7, va='center', ha='right')
fig.suptitle('Image Reconstruction: Image -> Encoder -> Decoder -> Image', fontsize=11)
plt.tight_layout(rect=[0.08, 0, 1, 0.93])
save_plot("phase3_reconstruction.png")

# --- (b) Text -> Image 生成 (0-9) 与真实样本对比 ---
real_examples = {}
for j in range(len(testset)):
    l = testset[j][1]
    if l not in real_examples:
        real_examples[l] = testset[j][0]
    if len(real_examples) == 10:
        break

fig, axes = plt.subplots(2, 10, figsize=(12, 2.5))
for d in range(10):
    r = gen_results[d]
    axes[0, d].imshow(to_img(r['image']), cmap='gray'); axes[0, d].axis('off')
    color = 'green' if r['correct'] else 'red'
    axes[0, d].set_title(f"{d} ({r['conf']*100:.0f}%)", fontsize=8, color=color)
    axes[1, d].imshow(to_img(real_examples[d]), cmap='gray'); axes[1, d].axis('off')
axes[0, 0].text(-0.15, 0.5, 'Generated', transform=axes[0, 0].transAxes,
                fontsize=9, va='center', ha='right', fontweight='bold')
axes[1, 0].text(-0.15, 0.5, 'Real', transform=axes[1, 0].transAxes,
                fontsize=9, va='center', ha='right', fontweight='bold')
fig.suptitle(f'Text -> Image Decoding  |  ExtAcc={ext_gen_acc*100:.0f}%  ExtConf={ext_gen_conf*100:.1f}%',
             fontsize=11)
plt.tight_layout(rect=[0.08, 0, 1, 0.93])
save_plot("phase3_text_to_image.png")

print("\n✅ 完成！这是经过优化的跨模态语义对齐 + 解码器生成方案。")