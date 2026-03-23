# 文件路径: Sugarcane-RTDETR-Project/tools/count_system.py
import cv2
import os
import argparse
from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description="甘蔗实例分割与自动计数系统")
    parser.add_argument('--weights', type=str, default='runs/train/final_model/weights/best.pt',
                        help='训练好的模型权重路径')
    parser.add_argument('--source', type=str, default='../datasets/sugarcane_coco/images/test',
                        help='测试图片目录或视频路径')
    parser.add_argument('--output', type=str, default='runs/detect/count_results', help='结果保存目录')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    return parser.parse_args()


def process_image(img_path, model, output_dir, conf_thres):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        return

    # 进行推理
    results = model(img, conf=conf_thres, verbose=False)[0]

    # 获取掩码数量 (即甘蔗实例数)
    sugarcane_count = 0
    if results.masks is not None:
        sugarcane_count = len(results.masks.data)

    # 获取画好分割结果的图像 (Ultralytics自带的plot方法)
    annotated_img = results.plot()

    # 在图像左上角绘制计数结果面板
    overlay = annotated_img.copy()
    cv2.rectangle(overlay, (20, 20), (450, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated_img, 0.4, 0, annotated_img)  # 半透明效果

    text = f"Sugarcane Count: {sugarcane_count}"
    cv2.putText(annotated_img, text, (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # 保存结果
    base_name = os.path.basename(img_path)
    save_path = os.path.join(output_dir, base_name)
    cv2.imwrite(save_path, annotated_img)
    print(f"Processed: {base_name} | Count: {sugarcane_count}")


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading model from {args.weights}...")
    model = RTDETR(args.weights)

    if os.path.isdir(args.source):
        # 批量处理文件夹中的图片
        img_files = [os.path.join(args.source, f) for f in os.listdir(args.source) if
                     f.endswith(('.jpg', '.png', '.jpeg'))]
        for img_file in img_files:
            process_image(img_file, model, args.output, args.conf)
    else:
        # 处理单张图片或视频 (此处仅展示图片处理，视频可自行用 cv2.VideoCapture 扩展)
        process_image(args.source, model, args.output, args.conf)

    print(f"✅ 所有计数结果已保存至: {args.output}")


if __name__ == '__main__':
    main()