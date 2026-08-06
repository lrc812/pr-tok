import os
import glob

def extract_image_filenames(folder_path, output_txt_path, prefix_path):

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        # 获取当前扩展名的所有文件
        files = glob.glob(os.path.join(folder_path, ext))
        files += glob.glob(os.path.join(folder_path, ext.upper()))  # 也查找大写扩展名
        image_files.extend(files)
    
    # 提取文件名（不含路径）并添加前缀
    full_paths = []
    for file_path in image_files:
        # 获取纯文件名
        filename = os.path.basename(file_path)
        # 添加前缀路径
        full_path = os.path.join(prefix_path, filename)
        full_paths.append(full_path)
    
    # 写入txt文件
    with open(output_txt_path, 'w') as f:
        for path in full_paths:
            f.write(path + '\n')
    
    print(f"成功提取 {len(full_paths)} 个图像文件名")
    print(f"结果已保存到: {output_txt_path}")
    print(f"示例前5行:")
    for i, path in enumerate(full_paths[:5], 1):
        print(f"  {i}. {path}")

# 使用示例
if __name__ == "__main__":
    # 配置参数 - 请根据实际情况修改这些路径
    image_folder = "/home/disk1/lihaoran/vq-gan/taming-transformers/data/openimage/validation"  # 替换为你的图像文件夹路径
    output_file = "openimage.txt"  # 输出的txt文件名 
    # 指定的前缀路径
    prefix = "/home/disk1/lihaoran/vq-gan/taming-transformers/data/openimage/validation/"
    # 确保前缀路径以斜杠结尾
    if not prefix.endswith('/'):
        prefix += '/'
    # 执行提取
    extract_image_filenames(image_folder, output_file, prefix)