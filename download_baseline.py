import os
import shutil
from ultralytics import RTDETR

def download_baseline():
    os.makedirs('baseline', exist_ok=True)
    print("⏳ 正在下载 RT-DETR-L 预训练权重...")
    model = RTDETR('rtdetr-l.pt')
    
    if os.path.exists('rtdetr-l.pt'):
        shutil.move('rtdetr-l.pt', 'models/baseline/rtdetr-l.pt')
        print("✅ 基线模型已成功下载并移动到 baseline/rtdetr-l.pt")
    elif os.path.exists('models/baseline/rtdetr-l.pt'):
         print("✅ 基线模型已存在于 baseline/rtdetr-l.pt")
    else:
        print("❌ 下载可能失败，请检查网络连接。")

if __name__ == '__main__':
    download_baseline()
