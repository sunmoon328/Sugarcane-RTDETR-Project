# 文件路径: Sugarcane-RTDETR-Project/tools/visualizer.py
import os
import cv2
import argparse
from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description="甘蔗实例分割结果与特征热力图可视化")
    parser.add_argument('--weights', type=str, default='runs/train/final_model/weights/best.pt',
                        help='训练好的模型权重路径')
    parser.add_argument('--source', type=str, required=True, help='测试图片路径 (建议选择遮挡严重的图片)')
    parser.add_argument('--output_dir', type=str, default='runs/detect/visualize_results', help='结果保存目录')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🔄 正在加载模型: {args.weights}")
    model = RTDETR(args.weights)

    img_name = os.path.basename(args.source)
    img = cv2.imread(args.source)
    if img is None:
        raise ValueError(f"无法读取图像: {args.source}")

    # 1. 生成检测与实例分割对比图
    print("📸 正在生成分割掩码对比图...")
    results = model.predict(source=img, conf=args.conf, save=False)
    annotated_img = results[0].plot(labels=True, boxes=True, masks=True)

    mask_output_path = os.path.join(args.output_dir, f"mask_pred_{img_name}")
    cv2.imwrite(mask_output_path, annotated_img)
    print(f"✅ 分割结果图已保存至: {mask_output_path}")

    # 2. 生成特征热力图 (Feature Map Visualization)
    # 利用 Ultralytics 内置的 visualize 参数，提取网络中间层的特征图
    print("🔥 正在提取深层特征热力图...")
    # 这会在 output_dir 下自动创建一个基于图片名的文件夹，里面包含各个 stage 的特征响应图
    model.predict(source=img, conf=args.conf, visualize=True, project=args.output_dir,
                  name=f"heatmap_{img_name.split('.')[0]}")

    print(f"✅ 特征热力图已生成！请前往 {args.output_dir}/heatmap_{img_name.split('.')[0]} 目录。")
    print(
        "💡 论文配图建议：在生成的文件夹中，挑选深层（如 stage3/stage4）的特征图，对比 Baseline 和 加入创新点后的模型在遮挡处的响应差异。")


if __name__ == "__main__":
    main()