import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

# --- ⚠️ 确保这两条路径与你电脑上的真实路径完全一致 ---
#KNOWLEDGE_BASE_PATH = r"E:\PythonD\ERA3\scripts\V9\scripts\unified_medical_kb.json"  # 你的 json 绝对路径
KNOWLEDGE_BASE_PATH = r"E:\PythonD\ERA3\scripts\V9\3wd2\unified_medical_kb.json"
OUTPUT_DIR = r"E:/PythonD/ERA3/scripts/V9/scripts/outputs_medical/"

def build_feature_matrix():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======================================================")
    print("     BiomedCLIP 医疗知识库特征提取脚本")
    print("======================================================")
    print(f"使用设备: {device}")

    print("\n加载 BiomedCLIP (医疗专精视觉语言模型)...")
    # BiomedCLIP 权重会自动从 HuggingFace 下载 (需要科学上网环境)
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
    model = model.to(device)
    model.eval()

    knowledge_base_path = Path(KNOWLEDGE_BASE_PATH)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    features_output_path = output_path / "medical_features.npz"

    if not knowledge_base_path.exists():
        print(f"错误: 知识库文件 '{knowledge_base_path}' 未找到。")
        sys.exit(1)

    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    all_features = []
    valid_kb_ids = []

    print(f"开始为 {len(knowledge_base)} 个医疗图像提取 512 维高阶特征...")
    with torch.no_grad():
        for entry in tqdm(knowledge_base, desc="提取特征"):
            try:
                image_path = entry.get('image_path')
                if not image_path or not os.path.exists(image_path):
                    continue

                image = Image.open(image_path).convert('RGB')
                img_tensor = preprocess_val(image).unsqueeze(0).to(device)

                # 提取全局图像特征并归一化 (BiomedCLIP 原生输出)
                image_feature = model.encode_image(img_tensor)
                image_feature = torch.nn.functional.normalize(image_feature, dim=-1)

                all_features.append(image_feature.cpu().numpy().squeeze())
                valid_kb_ids.append(str(Path(image_path).as_posix()))

            except Exception as e:
                tqdm.write(f"\n警告: 处理 {image_path} 失败: {e}")
                continue

    if not all_features:
        print("\n错误: 未能提取任何特征。")
        sys.exit(1)

    features_matrix = np.array(all_features, dtype=np.float32)
    print(f"\n特征提取完成，共 {features_matrix.shape[0]} 个特征。特征维度: {features_matrix.shape[1]}")
    np.savez(features_output_path, features=features_matrix, ids=np.array(valid_kb_ids))
    print(f"✅ BiomedCLIP 特征矩阵保存成功: {features_output_path}")

if __name__ == '__main__':
    build_feature_matrix()