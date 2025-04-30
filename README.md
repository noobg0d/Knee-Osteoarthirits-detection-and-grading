# 🦵 Joint Vision: Vision-Language Model for Knee Osteoarthritis Detection and Grading

Joint Vision is a multimodal deep learning application for automatic detection and Kellgren-Lawrence (KL) grading of knee osteoarthritis from X-ray images. It integrates an ensemble of vision models with a language module powered by LLaMA-3 to generate clinical reports.

---

## 📂 Features

- ✅ Detect presence of osteoarthritis from knee X-rays
- 📊 Perform KL grading (0–4 scale)
- 🧠 Uses ensemble of ViT, ResNet50, DenseNet-121, EfficientNet-B8, and VGG16
- 💬 Generates clinical suggestions using LLaMA-3
- ⚡ Built with PyTorch + Streamlit + Ollama (local LLM deployment)

---

## 📁 Folder Structure

knee_OA_dl_app/ ├── models/ # Pretrained models (excluded from Git) ├── xray_images/ # Input images ├── app.py # Streamlit app ├── utils.py # Helper functions ├── requirements.txt # Dependencies ├── README.md # This file
---

## ⚙️ Installation

1. Clone the repository
```bash
git clone https://github.com/noobg0d/Knee-Osteoarthirits-detection-and-grading.git
cd Knee-Osteoarthirits-detection-and-grading
2.Create and activate a virtual environment
conda create -n jointvision python=3.10
conda activate jointvision
pip install -r requirements.txt
ollama run llama3
streamlit run app.py

