from pathlib import Path

from modelscope import snapshot_download

# 魔搭上的模型仓库名
model_id = "LLM-Research/Meta-Llama-3.1-8B-Instruct"

# 本地保存位置
ROOT_DIR = Path(__file__).resolve().parent
save_dir = str(ROOT_DIR / "Llama-3.1-8B-Instruct")

if __name__ == "__main__":
    model_dir = snapshot_download(
        model_id,
        local_dir=save_dir,
        ignore_file_pattern=[
            "*.pth",
            "original/*",
        ],
    )
    print(f"模型下载完成，保存位置：{model_dir}")
