from argparse import Namespace
from dataset.t2i import Text2ImgDataset
from torchvision import transforms

# 定义 args 为 Namespace 对象
args = Namespace(
    data_path="/home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/train_captions_modified.csv",
    t5_feat_path="/home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/train_images/train",
    image_size=256,
    downsample_size=16
)

# 定义 transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 创建数据集
dataset = Text2ImgDataset(args, transform)

# 测试数据集
for i in range(5):
    data = dataset[i]
    print(data)