# @Author  : Franz Liszt
# @Time    : 2025/6/10 22:35
# @Email   : News53231323@163.com
# @File    : download_models.py
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

# 下载并缓存模型
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

save_path = Path(__file__).resolve().parent.parent / "local_models" / "sentence-transformers" / "all-mpnet-base-v2"

if not save_path.exists():
    save_path.mkdir(parents=True, exist_ok=True)
# 保存到本地文件夹
model.save(str(save_path))


