# 🤖 MiniGPT  

The simplest, cleanest repository for **building and training a GPT model from scratch** ✨  
This project is my hands-on practice to understand **Transformer architecture**, **tokenization**, **training loops**, and **text generation**, inspired by nanoGPT but implemented with my own learning journey in mind.  

The goal: make GPT **approachable, hackable, and fully customizable** for small- to medium-scale experiments on a single GPU or CPU ⚡  
The code is minimal, readable, and beginner-friendly — with `train.py` as the ~300-line training loop and `model.py` as the ~300-line Transformer definition 🛠️  

---

## 📌 Highlights
- 🏗️ **From scratch** implementation of a GPT-like Transformer model  
- 🔤 **Character-level & token-level training** support  
- 📂 **Easy dataset preparation** (Tiny Shakespeare, custom text)  
- ⚙️ **Configurable hyperparameters** for scaling up/down  
- ✍️ **Sampling script** for text generation from checkpoints  
- 🧩 Modular, clean, and **hacker-friendly** code  

---

## 📦 Installation  

```bash
pip install torch numpy transformers datasets tqdm
```
Dependencies 📜

🐍 pytorch ❤️
➕ numpy ❤️
🤗 transformers (HuggingFace GPT-2 tools) ❤️
📊 datasets (HuggingFace dataset utilities) ❤️
⏳ tqdm (progress bars) ❤️

🚀 Quick Start
1️⃣ Prepare your dataset
📜 Train a character-level GPT on Shakespeare:

```bash
python data/shakespeare_char/prepare.py
```
Generates: train.bin & val.bin 📦

2️⃣ Train a small GPT
💻 On GPU:

```bash
python train.py config/train_shakespeare_char.py
```
📏 Context: 256
📐 Embedding: 384
🧠 6 Layers × 6 Heads
⏱️ ~3 minutes on A100 GPU

🖥️ On CPU:
```bash
python train.py config/train_shakespeare_char.py \
    --device=cpu --compile=False --block_size=64 \
    --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 \
    --max_iters=2000 --dropout=0.0
```
Runs in ~3–5 minutes 🕒

3️⃣ Generate text
```bash
python sample.py --out_dir=out-shakespeare-char
```
📜 Example:
```
vbnet
```
DUKE:
I thank your eyes against it.

ANGELO:
And cowards it be strawn to my bed.
🏋️‍♂️ Reproducing GPT-2 Scale (Optional)
Train on OpenWebText:

```bash
python data/openwebtext/prepare.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```
🔧 Finetuning
```bash
python train.py config/finetune_shakespeare.py
```
✅ Loads GPT-2 weights → trains with small LR → adapts to new data
🧪 Sampling from Pretrained Models
```bash
python sample.py \
    --init_from=gpt2-xl \
    --start="The meaning of life is" \
    --num_samples=5 --max_new_tokens=100
```
⚡ Efficiency Tips
🚀 Use torch.compile() (PyTorch 2.0) for faster training
🍏 On Apple Silicon: --device=mps for GPU acceleration

📅 To-Do List
 🔄 Rotary Embeddings & Flash Attention
 🧮 Mixed-Precision Training (fp16/bf16)
 🖥️ FSDP for large model scaling
 🌐 Web-based text generation UI






