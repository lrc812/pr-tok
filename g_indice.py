import os
import pandas as pd
from PIL import Image
import torch
from taming.models.vqgan import VQModel
from torchvision import transforms
from tqdm import tqdm

# 加载训练好的 VQ-GAN 模型
def load_vqgan_model(config_path, ckpt_path):
    from omegaconf import OmegaConf
    from taming.models.vqgan import VQModel

    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return model
# 定义图像预处理
def get_image_transform(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 将像素值归一化到 [-1, 1]
    ])
# 处理 test_captions.csv 文件
def process_test_captions(csv_path, image_dir, output_csv, vqgan_model, device):
    # 读取 CSV 文件
    data = pd.read_csv(csv_path)

    # 添加 Image_Indices 列
    image_transform = get_image_transform()
    image_indices = []

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        image_id = row["ID"]
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping...")
            image_indices.append(None)
            continue
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0).to(device)  # [1, 3, 256, 256]
        # 使用 VQ-GAN 编码图像
        with torch.no_grad():
            _, _, info = vqgan_model.encode(image_tensor)
            # print(info[-1].shape)
            quantized_indices = info[-1].view(-1).cpu().numpy().tolist()  # 展平为 1D 列表
        # print(len(quantized_indices))
        # 保存量化索引
        image_indices.append(quantized_indices)

    # 将量化索引添加到 DataFrame
    data["Image_Indices"] = image_indices
    # 保存到新的 CSV 文件
    data.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

if __name__ == "__main__":
    # 配置路径
    config_path = "/home/disk1/lihaoran/vq-gan/taming-transformers/logs/2025-11-14T05-48-50_custom_vqgan/configs/2025-11-14T05-48-50-project.yaml"  # 替换为 VQ-GAN 配置文件路径
    ckpt_path = "/home/disk1/lihaoran/vq-gan/taming-transformers/logs/2025-11-14T05-48-50_custom_vqgan/checkpoints/last.ckpt"  # 替换为 VQ-GAN 权重文件路径
    csv_path = "/home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/train_captions.csv"
    image_dir = "/home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/train_images/train/"  # 替换为图片所在目录
    output_csv = "/home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/train_captions_with_indices.csv"
    # 加载 VQ-GAN 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqgan_model = load_vqgan_model(config_path, ckpt_path).to(device)
    # 处理数据
    process_test_captions(csv_path, image_dir, output_csv, vqgan_model, device)