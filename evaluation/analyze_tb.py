import os
import argparse
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
# print("日志中的标量标签：", ea.Tags()["scalars"])
tags =["train/total_loss_epoch", "train/total_loss_step","train/aeloss_step","train/quant_loss_step","train/logits_fake_step"]  # 替换为你要查看的标签列表
def visual(log_path):
    ea = event_accumulator.EventAccumulator(
    log_path,
    size_guidance={  # 按需配置，只加载标量（最关键），其他类型可注释
        event_accumulator.SCALARS: 0,  # 0表示加载所有标量数据
        # event_accumulator.HISTOGRAMS: 0,  # 不需要可注释
        # event_accumulator.IMAGES: 0,      # 不需要可注释
    }
)
    img_save_dir = os.path.dirname(log_path)
    ea.Reload()  # 加载数据
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        steps = [event.step for event in events]
        print(f"正在处理标签: {tag}")
        print(f"数据点数量: {len(events)}")
        # print(type(events))
        # print(events)
        values = [event.value  for event in events]
        # 如果value有负数打印输出
        if any(v < 0 for v in values):
            print(f"标签 {tag} 包含负值。")
        plt.figure()
        plt.plot(steps, values, label=tag)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(f'TensorBoard Scalar: {tag}')
        plt.legend()
        plt.grid()
        plt.show()
        safe_tag = tag.replace('/', '_').replace('\\', '_')  # 替换路径分隔符为下划线
        img_save_path = os.path.join(img_save_dir, f"{safe_tag}.png")  # 拼接保存路径
        plt.savefig(img_save_path, dpi=150, bbox_inches='tight')  # dpi 控制图片清晰度
        print(f"已保存图片：{img_save_path}")
    plt.close()
        
if __name__ == "__main__":
    log_path = "/data1/lhran/proj/vq-gan/taming-transformers/logs/2026-07-29T11-55-57_my_vqgan_experiment/tensorboard/version_0"
    visual(log_path)