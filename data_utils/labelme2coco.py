import os
import json
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm


class Labelme2COCO:
    def __init__(self, labelme_json_dir, labelme_image_dir, output_dir, classes):
        self.labelme_json_dir = labelme_json_dir
        self.labelme_image_dir = labelme_image_dir
        self.output_dir = output_dir
        self.classes = classes
        self.class_name_to_id = {cls: i + 1 for i, cls in enumerate(classes)}

        # 初始化 COCO 结构
        self.coco_dataset = {
            "info": {"description": "Sugarcane Dataset", "version": "1.0"},
            "images": [],
            "annotations": [],
            "categories": [{"id": v, "name": k} for k, v in self.class_name_to_id.items()]
        }
        self.image_id = 1
        self.annotation_id = 1

    def process_jsons(self, json_files, output_json_path, image_dest_dir):
        dataset = self.coco_dataset.copy()
        dataset["images"] = []
        dataset["annotations"] = []

        os.makedirs(image_dest_dir, exist_ok=True)

        for json_file in tqdm(json_files, desc=f"Processing {os.path.basename(output_json_path)}"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 1. 添加图片信息 (带智能后缀探测)
            original_img_name = data['imagePath'].split('\\')[-1].split('/')[-1]
            base_name = os.path.splitext(original_img_name)[0]  # 提取纯文件名，例如 "0044"

            # 自动探测实际的图片后缀名，兼容 Labelme 存错后缀的情况
            possible_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']
            img_path = None
            actual_img_name = original_img_name

            for ext in possible_extensions:
                temp_path = os.path.join(str(self.labelme_image_dir), base_name + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    actual_img_name = base_name + ext
                    break

            # 复制图片到目标文件夹 (加入防错机制，并用 str() 消除 IDE 警告)
            if img_path and os.path.exists(img_path):
                dest_path = os.path.join(str(image_dest_dir), str(actual_img_name))
                shutil.copy(str(img_path), dest_path)
            else:
                print(f"\n警告: 找不到纯文件名为 {base_name} 的任何格式图片(.jpg/.png)，请检查 labelme_images 文件夹！")
                continue

            # 这里的 file_name 必须使用真实找到的实际文件名
            image_info = {
                "id": self.image_id,
                "file_name": actual_img_name,
                "height": data['imageHeight'],
                "width": data['imageWidth']
            }
            dataset["images"].append(image_info)

            # 2. 添加标注信息
            for shape in data['shapes']:
                label = shape['label']
                if label not in self.class_name_to_id:
                    continue

                points = shape['points']
                # 将点集拉平为 [x1, y1, x2, y2, ...]
                segmentation = [coord for point in points for coord in point]

                # 计算 BBox 和 Area
                pts = np.array(points, np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                area = cv2.contourArea(pts)

                annotation = {
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": self.class_name_to_id[label],
                    "segmentation": [segmentation],
                    "area": float(area),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "iscrowd": 0
                }
                dataset["annotations"].append(annotation)
                # 这一步会自动为你未编组的甘蔗生成独立的实例 ID
                self.annotation_id += 1

            self.image_id += 1

        # 保存为 COCO JSON
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 配置你的双文件夹路径
    JSON_DIR = "../datasets/raw_labelme/labelme_jsons"
    IMAGE_DIR = "../datasets/raw_labelme/labelme_images"
    OUT_DIR = "../datasets/sugarcane_coco"
    CLASSES = ["sugarcane"]

    # 获取所有 json 文件并划分
    all_jsons = glob.glob(os.path.join(JSON_DIR, "*.json"))

    if len(all_jsons) == 0:
        print(f"错误：在 {JSON_DIR} 中没有找到任何 JSON 文件，请检查路径！")
        exit()

    # 划分 8:1:1 (Train : Val : Test)
    train_jsons, test_val_jsons = train_test_split(all_jsons, test_size=0.2, random_state=42)
    val_jsons, test_jsons = train_test_split(test_val_jsons, test_size=0.5, random_state=42)

    converter = Labelme2COCO(JSON_DIR, IMAGE_DIR, OUT_DIR, CLASSES)

    print(f"找到 {len(all_jsons)} 个标注文件，开始转换数据集...")
    converter.process_jsons(train_jsons, os.path.join(OUT_DIR, "annotations", "instances_train.json"),
                            os.path.join(OUT_DIR, "images", "train"))
    converter.process_jsons(val_jsons, os.path.join(OUT_DIR, "annotations", "instances_val.json"),
                            os.path.join(OUT_DIR, "images", "val"))
    converter.process_jsons(test_jsons, os.path.join(OUT_DIR, "annotations", "instances_test.json"),
                            os.path.join(OUT_DIR, "images", "test"))
    print(f"转换完成！数据集已完美就绪，保存在 {OUT_DIR}")