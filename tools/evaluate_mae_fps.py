# 文件路径: Sugarcane-RTDETR-Project/tools/evaluate_mae_fps.py
import json
import os
import time
import argparse
from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description="评估模型的 MAE (计数误差) 和 FPS (推理速度)")
    parser.add_argument('--weights', type=str, default='runs/train/final_model/weights/best.pt')
    parser.add_argument('--test_json', type=str, default='../datasets/sugarcane_coco/annotations/instances_test.json',
                        help='测试集 COCO 标注文件')
    parser.add_argument('--img_dir', type=str, default='../datasets/sugarcane_coco/images/test', help='测试集图片目录')
    parser.add_argument('--conf', type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    model = RTDETR(args.weights)

    # 读取真实的计数标签 (Ground Truth)
    with open(args.test_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    gt_counts = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        gt_counts[img_id] = gt_counts.get(img_id, 0) + 1

    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    total_absolute_error = 0
    total_images = len(id_to_filename)
    total_time = 0.0

    print(f"🚀 开始评估 MAE 和 FPS，共计 {total_images} 张图片...")

    for img_id, filename in id_to_filename.items():
        img_path = os.path.join(args.img_dir, filename)
        if not os.path.exists(img_path):
            continue

        gt_count = gt_counts.get(img_id, 0)

        # 测算 FPS (仅计算推理时间)
        start_time = time.time()
        results = model.predict(source=img_path, conf=args.conf, verbose=False)
        end_time = time.time()

        total_time += (end_time - start_time)

        # 获取预测计数
        pred_count = len(results[0].masks.data) if results[0].masks is not None else 0

        # 计算绝对误差
        error = abs(pred_count - gt_count)
        total_absolute_error += error

    # 计算最终指标
    mae = total_absolute_error / total_images if total_images > 0 else 0
    avg_time_per_img = total_time / total_images if total_images > 0 else 0
    fps = 1.0 / avg_time_per_img if avg_time_per_img > 0 else 0

    print("-" * 30)
    print(f"📊 评估结果:")
    print(f"模型权重: {args.weights}")
    print(f"平均绝对误差 (MAE): {mae:.4f} (越低越好)")
    print(f"平均推理帧率 (FPS): {fps:.2f} 帧/秒")
    print("-" * 30)


if __name__ == "__main__":
    main()