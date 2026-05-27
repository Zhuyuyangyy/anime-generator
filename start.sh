#!/bin/bash
# anime-generator 启动脚本
# Stable Diffusion多后端集成 - SD + ComfyUI + DreamStudio

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  anime-generator - AI动漫生成系统"
echo "  Stable Diffusion + ComfyUI + DreamStudio"
echo "=============================================="
echo

VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install -q openai pillow python-dotenv 2>/dev/null

echo "[启动] 运行主程序..."
python3 main.py "$@"