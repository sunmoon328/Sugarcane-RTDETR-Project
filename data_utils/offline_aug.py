import cv2
import os
import glob
from tqdm import tqdm


def apply_clahe(img_path, output_dir):
    img = cv2.imread(img_path)
    if img is None:
        return

    # 将图像转换到 LAB 颜色空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 限制对比度的自适应直方图均衡化 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 合并通道并转回 BGR
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 保存图像
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, f"clahe_{filename}")
    cv2.imwrite(save_path, final_img)


if __name__ == "__main__":
    # 配置输入输出路径
    INPUT_DIR = "../datasets/sugarcane_coco/images/train"
    OUTPUT_DIR = "../datasets/sugarcane_coco/images/train_clahe"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))

    for img_path in tqdm(image_paths, desc="Applying CLAHE"):
        apply_clahe(img_path, OUTPUT_DIR)

    print("✅ 离线数据增强完成！")