# 文件路径: Sugarcane-RTDETR-Project/tools/dataset_checker.py
import json
import os
import cv2
import numpy as np
import random


def check_coco_dataset(json_path, img_dir, output_dir, num_samples=5):
    """
    随机抽取 COCO 数据集中的图片，并将 BBox 和 Mask 绘制在图片上，用于验证转换是否正确。
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # 随机抽取几张图片
    sample_images = random.sample(images, min(num_samples, len(images)))

    for img_info in sample_images:
        img_id = img_info['id']
        img_name = img_info['file_name']
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"警告：找不到图片 {img_path}")
            continue

        image = cv2.imread(img_path)

        # 查找该图片的所有标注
        img_anns = [ann for ann in annotations if ann['image_id'] == img_id]

        for ann in img_anns:
            # 绘制 BBox (绿色)
            x, y, w, h = [int(v) for v in ann['bbox']]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制 Mask 多边形 (红色)
            for seg in ann['segmentation']:
                pts = np.array(seg, np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

            # 加上类别标签
            cat_name = categories[ann['category_id']]
            cv2.putText(image, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out_path = os.path.join(output_dir, f"checked_{img_name}")
        cv2.imwrite(out_path, image)
        print(f"已保存可视化结果至: {out_path}")


if __name__ == "__main__":
    # 指向你刚刚用 labelme2coco.py 生成的路径
    JSON_FILE = "../datasets/sugarcane_coco/annotations/instances_train.json"
    IMG_DIR = "../datasets/sugarcane_coco/images/train"
    OUT_DIR = "../datasets/checker_output"

    print("开始进行数据集可视化检查...")
    check_coco_dataset(JSON_FILE, IMG_DIR, OUT_DIR, num_samples=10)
    print("检查完成！请前往 checker_output 文件夹查看标注是否对齐。")