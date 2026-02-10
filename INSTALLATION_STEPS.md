# Pipeline 环境安装步骤（从0开始）

**完整的操作命令列表，按顺序执行即可。**

---

## 前提条件

- 操作系统: Ubuntu 20.04/22.04 (推荐)
- GPU: NVIDIA GPU (8GB+ 显存)
- 磁盘空间: 50GB 以上

---

## 第一步：系统准备

### 1.1 安装 NVIDIA 驱动和 CUDA

```bash
# 检查是否已安装
nvidia-smi

# 如果未安装，执行以下命令
# Ubuntu 22.04
sudo apt update
sudo apt install nvidia-driver-525
sudo apt install nvidia-cuda-toolkit

# 重启
sudo reboot

# 重启后验证
nvidia-smi
nvcc --version
```

### 1.2 安装系统依赖

```bash
sudo apt update
sudo apt install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    build-essential
```

---

## 第二步：克隆项目

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 或者如果已有项目文件夹
cd /root/program2
```

---

## 第三步：创建 Python 虚拟环境

```bash
# 创建虚拟环境
python3.10 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel
```

---

## 第四步：安装 PyTorch

### 4.1 确定 CUDA 版本

```bash
nvcc --version
# 记下 CUDA 版本，例如 11.8 或 12.1
```

### 4.2 安装对应版本的 PyTorch

**CUDA 11.8:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

**验证安装:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

预期输出:
```
PyTorch: 2.1.0+cu118
CUDA available: True
```

---

## 第五步：安装其他 Python 依赖

```bash
# 安装 requirements.txt 中的所有包
pip install -r requirements.txt

# 如果某些包安装失败，单独安装
pip install transformers==4.40.0
pip install accelerate==0.25.0
pip install qwen-vl-utils
pip install paddlepaddle-gpu==2.5.0
pip install paddleocr==2.7.0
pip install Pillow opencv-python numpy
pip install sentencepiece protobuf regex
pip install tqdm pandas requests
pip install huggingface-hub
```

**验证安装:**
```bash
pip list | grep -E "torch|transformers|paddleocr|qwen"
```

---

## 第六步：下载模型文件

### 6.1 安装 HuggingFace CLI

```bash
pip install huggingface-hub[cli]

# 验证
huggingface-cli --version
```

### 6.2 创建模型目录

```bash
mkdir -p models
mkdir -p checkpoints
```

### 6.3 下载 Phi-3.5-Vision (8GB)

**模型名称**: `microsoft/Phi-3.5-vision-instruct`

```bash
huggingface-cli download microsoft/Phi-3.5-vision-instruct \
  --local-dir models/phi-3_5_vision \
  --local-dir-use-symlinks False
```

**验证:**
```bash
ls -lh models/phi-3_5_vision/
# 应该看到 model-00001-of-00002.safetensors (4.7GB)
#          model-00002-of-00002.safetensors (3.2GB)
#          config.json
#          等其他文件
```

### 6.4 下载 Qwen2-VL-2B (5GB)

**模型名称**: `Qwen/Qwen2-VL-2B-Instruct`

```bash
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct \
  --local-dir models/Qwen2-VL-2B-Instruct \
  --local-dir-use-symlinks False
```

**验证:**
```bash
ls -lh models/Qwen2-VL-2B-Instruct/
# 应该看到 model-00001-of-00002.safetensors (3.8GB)
#          model-00002-of-00002.safetensors (410MB)
#          config.json
#          等其他文件
```

### 6.5 下载 TrOCR (2.5GB)

**模型名称**: `microsoft/trocr-base-handwritten`

```bash
huggingface-cli download microsoft/trocr-base-handwritten \
  --local-dir trocr-base-handwritten \
  --local-dir-use-symlinks False
```

**验证:**
```bash
ls -lh trocr-base-handwritten/
# 应该看到 model.safetensors (1.3GB)
#          pytorch_model.bin (1.3GB)
#          config.json
#          等其他文件

---

## 第七步：获取分类器 Checkpoint

### 方法 1: 自己训练（需要训练数据）

```bash
# 如果有训练数据和训练脚本
python printed_vs_hand_main.py --train
```

### 方法 2: 从提供的位置下载（如果有）

```bash
# 示例：从网盘下载
wget YOUR_DOWNLOAD_LINK -O checkpoints/printed_vs_hand_best.pth

# 或使用 curl
curl -L YOUR_DOWNLOAD_LINK -o checkpoints/printed_vs_hand_best.pth
```

### 方法 3: 临时跳过（仅用于测试）

```bash
# 创建一个占位文件（不推荐，但可用于测试）
touch checkpoints/printed_vs_hand_best.pth
```

**验证:**
```bash
ls -lh checkpoints/
# 应该看到 printed_vs_hand_best.pth (43MB)
```

---

## 第八步：验证安装

### 8.1 检查 Python 环境

```bash
python --version
# 输出: Python 3.10.x
```

### 8.2 检查 PyTorch 和 CUDA

```bash
python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
EOF
```

**预期输出:**
```
PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU count: 1
GPU name: NVIDIA GeForce RTX 3090
GPU memory: 24.0 GB
```

### 8.3 检查依赖包

```bash
python << EOF
packages = {
    'transformers': 'transformers',
    'accelerate': 'accelerate', 
    'paddleocr': 'paddleocr',
    'qwen_vl_utils': 'qwen-vl-utils',
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
}

for module_name, package_name in packages.items():
    try:
        if module_name == 'PIL':
            import PIL
            print(f"✓ {package_name}: {PIL.__version__}")
        elif module_name == 'cv2':
            import cv2
            print(f"✓ {package_name}: {cv2.__version__}")
        else:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'installed')
            print(f"✓ {package_name}: {version}")
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
EOF
```

### 8.4 检查模型文件

```bash
python << EOF
from pathlib import Path

models = [
    ('models/phi-3_5_vision', 'Phi-3.5-Vision'),
    ('models/Qwen2-VL-2B-Instruct', 'Qwen2-VL-2B'),
    ('trocr-base-handwritten', 'TrOCR'),
    ('checkpoints/printed_vs_hand_best.pth', 'Classifier'),
]

print("Model files check:")
for path, name in models:
    p = Path(path)
    if p.exists():
        if p.is_dir():
            size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
            size_gb = size / 1024**3
            print(f"✓ {name}: {size_gb:.1f} GB")
        else:
            size_mb = p.stat().st_size / 1024**2
            print(f"✓ {name}: {size_mb:.1f} MB")
    else:
        print(f"✗ {name}: NOT FOUND")
EOF
```

**预期输出:**
```
Model files check:
✓ Phi-3.5-Vision: 8.0 GB
✓ Qwen2-VL-2B: 5.0 GB
✓ TrOCR: 2.5 GB
✓ Classifier: 43.0 MB
```

### 8.5 测试运行

```bash
# 创建测试图片（如果没有）
mkdir -p test_images

# 测试 Phi3.5 Pipeline
python case2/pipeline.py \
  --image test_images/sample.jpg \
  --query "What is written in the image?" \
  --output test_output.json \
  --limit 1

# 检查输出
cat test_output.json
```

**预期**: 应该成功生成 JSON 文件，包含识别结果。

---

## 第九步：首次运行注意事项

### 9.1 PaddleOCR 模型自动下载

**第一次运行 pipeline 时，PaddleOCR 会自动下载以下模型:**

1. **PP-OCRv5_server_det** (~50 MB) - 文字检测模型
2. **en_PP-OCRv5_mobile_rec** (~10 MB) - 英文识别模型

**下载位置:** `~/.paddlex/official_models/`

**首次运行命令:**
```bash
python case2/pipeline.py \
  --image test_images/sample.jpg \
  --query "What is written?" \
  --output test.json
```

**预期行为:**
```
Creating model: ('PP-OCRv5_server_det', None)
Downloading model files...  [下载进度条]
Creating model: ('en_PP-OCRv5_mobile_rec', None)
Downloading model files...  [下载进度条]
```

**等待下载完成即可，下次运行就不会再下载。**

### 9.2 布局检测模型自动下载

**如果 pipeline 需要布局检测，会自动下载:**

**PP-DocLayout_plus-L** (~124 MB)

**下载位置:** `layoutModel/` 或 PaddlePaddle 缓存目录

---

## 完整的模型清单

### 需要手动下载的模型

| 模型名称 | HuggingFace 仓库 | 本地路径 | 大小 | 必需 |
|---------|-----------------|---------|------|------|
| Phi-3.5-Vision | `microsoft/Phi-3.5-vision-instruct` | `models/phi-3_5_vision/` | 8 GB | ✅ Phi3.5 Pipeline |
| Qwen2-VL-2B | `Qwen/Qwen2-VL-2B-Instruct` | `models/Qwen2-VL-2B-Instruct/` | 5 GB | ✅ Qwen Pipeline |
| Qwen2-VL-7B | `Qwen/Qwen2-VL-7B-Instruct` | `models/Qwen2-VL-7B-Instruct/` | 15 GB | ❌ 可选 |
| TrOCR | `microsoft/trocr-base-handwritten` | `trocr-base-handwritten/` | 2.5 GB | ✅ 手写识别 |
| 分类器 | (需自己训练或获取) | `checkpoints/printed_vs_hand_best.pth` | 43 MB | ✅ 分类器 |

### 自动下载的模型

| 模型名称 | 下载时机 | 下载位置 | 大小 |
|---------|---------|---------|------|
| PP-OCRv5_server_det | 首次运行 | `~/.paddlex/official_models/` | 50 MB |
| en_PP-OCRv5_mobile_rec | 首次运行 | `~/.paddlex/official_models/` | 10 MB |
| PP-DocLayout_plus-L | 首次布局检测 | `layoutModel/` | 124 MB |

---

## 快速安装脚本

将以下内容保存为 `quick_install.sh`:

```bash
#!/bin/bash
set -e

echo "================================"
echo "Pipeline 环境快速安装"
echo "================================"

# 1. 创建虚拟环境
echo "Step 1: 创建虚拟环境..."
python3.10 -m venv venv
source venv/bin/activate

# 2. 升级 pip
echo "Step 2: 升级 pip..."
pip install --upgrade pip setuptools wheel

# 3. 安装 PyTorch (CUDA 11.8)
echo "Step 3: 安装 PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu118

# 4. 安装依赖
echo "Step 4: 安装依赖包..."
pip install -r requirements.txt

# 5. 创建目录
echo "Step 5: 创建模型目录..."
mkdir -p models checkpoints

# 6. 下载模型
echo "Step 6: 下载模型 (这将需要一些时间)..."

echo "  - 下载 Phi-3.5-Vision (8GB)..."
huggingface-cli download microsoft/Phi-3.5-vision-instruct \
  --local-dir models/phi-3_5_vision \
  --local-dir-use-symlinks False

echo "  - 下载 Qwen2-VL-2B (5GB)..."
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct \
  --local-dir models/Qwen2-VL-2B-Instruct \
  --local-dir-use-symlinks False

echo "  - 下载 TrOCR (2.5GB)..."
huggingface-cli download microsoft/trocr-base-handwritten \
  --local-dir trocr-base-handwritten \
  --local-dir-use-symlinks False

echo ""
echo "================================"
echo "✓ 安装完成！"
echo "================================"
echo ""
echo "⚠️  注意事项:"
echo "1. 分类器 checkpoint 需要单独获取"
echo "   位置: checkpoints/printed_vs_hand_best.pth"
echo ""
echo "2. PaddleOCR 模型会在首次运行时自动下载"
echo ""
echo "3. 测试安装:"
echo "   python case2/pipeline.py --image test.jpg --query 'What is this?'"
echo ""
```

运行脚本:
```bash
chmod +x quick_install.sh
./quick_install.sh
```

---

## 故障排除

### 问题 1: huggingface-cli 下载很慢

**解决方案: 使用镜像**
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download microsoft/Phi-3.5-vision-instruct \
  --local-dir models/phi-3_5_vision
```

### 问题 2: CUDA out of memory

**解决方案: 使用 Qwen2-VL-2B 而不是 7B**
```bash
# 确保下载的是 2B 版本
ls -lh models/Qwen2-VL-2B-Instruct/
```

### 问题 3: ImportError: cannot import name 'XXX'

**解决方案: 重新安装依赖**
```bash
pip install -r requirements.txt --force-reinstall
```

### 问题 4: PaddleOCR 下载模型失败

**解决方案: 手动下载**
```bash
# PaddleOCR 会在首次运行时自动下载
# 如果失败，等待重试或检查网络
```

---

## 完成检查清单

安装完成后，检查以下项目:

- [ ] Python 3.10+ 已安装
- [ ] CUDA 11.8+ 可用 (`nvidia-smi`)
- [ ] 虚拟环境已创建并激活
- [ ] PyTorch 已安装 (`torch.cuda.is_available()` 返回 True)
- [ ] requirements.txt 所有包已安装
- [ ] Phi-3.5-Vision 已下载 (8GB)
- [ ] Qwen2-VL-2B 已下载 (5GB)
- [ ] TrOCR 已下载 (2.5GB)
- [ ] 分类器 checkpoint 已获取 (43MB)
- [ ] 测试运行成功

---

## 下一步

安装完成后，可以:

1. **运行 Phi3.5 Pipeline:**
   ```bash
   python case2/pipeline.py --image your_image.jpg --query "your question"
   ```

2. **运行 Qwen Pipeline:**
   ```bash
   python qwen_version/pipeline.py --image your_image.jpg --query "your question"
   ```

3. **批量处理:**
   ```bash
   python case2/pipeline.py --json dataset.json --output results.json
   ```

---

## 总结

**安装时间估算:**
- 依赖包安装: 10-20 分钟
- 模型下载: 1-2 小时 (取决于网速)
- 总计: 1.5-2.5 小时

**磁盘空间使用:**
- Python 环境: ~5 GB
- 模型文件: ~16 GB
- 总计: ~21 GB

**显存需求:**
- Phi3.5 Pipeline: 8-10 GB
- Qwen2-VL-2B Pipeline: 6-8 GB
- 同时运行: 不建议

祝安装顺利！🚀
