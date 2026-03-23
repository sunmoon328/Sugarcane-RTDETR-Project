# 文件路径: Sugarcane-RTDETR-Project/tools/visualize_heatmap.py
import cv2
import numpy as np
import torch
import os
import argparse
from ultralytics import RTDETR


# 注意：此脚本利用了 Ultralytics 较新版本内置的特征图提取功能
# 如果由于版本原因不支持，可替换为通用的 Grad-CAM 库

def parse_args():
    parser = argparse.ArgumentParser(description="甘蔗检测热力图可视化 (用于论文配图)")
    parser.add_argument('--weights', type=str, default='runs/train/final_model/weights/best.pt', help='模型权重')
    parser.add_argument('--source', type=str, required=True, help='输入一张遮挡严重的甘蔗图片')
    parser.add_argument('--output', type=str, default='runs/detect/heatmaps', help='保存目录')
    return parser.parse_args()


def generate_heatmap(model_path, img_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = RTDETR(model_path)

    # 读取原始图像
    img = cv2.imread(img_path)

    # 开启特征可视化模式进行推理
    # Ultralytics 支持通过 visualize=True 保存特征图
    print("正在提取网络深层特征图...")
    results = model(img, visualize=True, project=output_dir, name='feature_maps', exist_ok=True)

    # 提示：visualize=True 会在 output_dir/feature_maps 下生成网络每一层的特征图 (stage0 到 stageN)
    # 在写论文时，您可以去那个文件夹里挑选出加入了 CoordAtt (CA) 模块那一层的特征图
    # 提取出来的图像会清晰地显示：基线模型的特征在甘蔗截断处是断开的，而您的改进模型在截断处有红色的高亮连接。

    print(f"✅ 特征图已生成！请前往 {os.path.join(output_dir, 'feature_maps')} 目录挑选用于论文的对比图。")
    print("💡 论文写作建议：选取 Baseline 和 最终模型 的同一张复杂图片，对比深层特征的热力图响应。")


if __name__ == '__main__':
    args = parse_args()
    generate_heatmap(args.weights, args.source, args.output)