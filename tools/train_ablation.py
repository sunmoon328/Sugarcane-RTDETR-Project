# 文件路径: tools/train_ablation.py
import argparse
from ultralytics import RTDETR
import os


def parse_args():
    parser = argparse.ArgumentParser(description="甘蔗实例分割：RT-DETR 消融实验训练脚本")
    # 实验配置
    parser.add_argument('--exp_name', type=str, default='baseline',
                        choices=['baseline', 'add_asc', 'add_ca', 'final_model'],
                        help='当前消融实验的名称')
    # 模型与数据路径
    parser.add_argument('--model_cfg', type=str, default='rtdetr-l.yaml',
                        help='模型配置文件路径 (如果引入创新点，请指向修改后的 yaml 文件)')
    parser.add_argument('--data_cfg', type=str, default='../datasets/sugarcane.yaml',
                        help='数据集配置文件路径')
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小 (GTX 1650 建议设为 2)')
    parser.add_argument('--img_size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0', help='GPU 设备编号')

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"🚀 开始执行实验: {args.exp_name}")
    print(f"加载模型配置: {args.model_cfg}")

    # 1. 初始化模型
    # 如果是 baseline，直接加载官方结构；如果是其他实验，加载你修改过网络结构的 YAML
    model = RTDETR(args.model_cfg)

    # 2. 如果之前有预训练权重，可以加载以加速收敛 (可选)
    # model.load('rtdetr-l.pt')

    # 3. 开始训练
    # 开启 mosaic 和 mixup 等在线数据增强
    results = model.train(
        data=args.data_cfg,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        project='runs/train',  # 结果保存的主目录
        name=args.exp_name,  # 本次实验保存的子目录名
        optimizer='AdamW',
        lr0=0.0001,
        mosaic=1.0,  # 启用 Mosaic 增强
        mixup=0.1  # 启用 Mixup 增强
    )

    print(f"✅ 实验 {args.exp_name} 训练完成！权重已保存在 runs/train/{args.exp_name}/weights/ 中。")


if __name__ == '__main__':
    main()