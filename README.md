# 🦵 Joint Vision

**Advanced Vision-Language Model for Knee Osteoarthritis Detection and Grading**



## Overview

Joint Vision is a state-of-the-art multimodal deep learning application designed for healthcare professionals to automatically detect and grade knee osteoarthritis from X-ray images. The system utilizes an ensemble of vision models combined with a language module powered by LLaMA-3 to generate comprehensive clinical reports.

## Key Features

- ✅ **Automatic Detection**: Identify presence of osteoarthritis from knee X-rays with high accuracy
- 📊 **KL Grading**: Perform Kellgren-Lawrence grading (0-4 scale) for disease severity assessment
- 🧠 **Model Ensemble**: Leverages multiple architectures (ViT, ResNet50, DenseNet-121, EfficientNet-B8, VGG16) for robust predictions
- 💬 **Clinical Reporting**: Generates natural language clinical suggestions using LLaMA-3
- 📈 **Visualization**: Interactive heatmaps highlighting affected regions
- ⚡ **Performance**: Optimized for speed with local LLM deployment via Ollama
- 🔒 **Privacy**: Process images locally with no data sent to external servers
- 🛠️ **DIY Training**: All models must be trained by users with their own datasets

## User Flow

1. **Upload X-ray Image**: Users can drag and drop knee X-ray images into the application
2. **Automatic Analysis**: The system processes the image through the ensemble of vision models
3. **View Results**: KL grade and detection confidence are displayed with visual explanations
4. **Clinical Report**: A comprehensive natural language report is generated with clinical suggestions
5. **Export Results**: Save or share analysis results and reports

## Components

### Image Analysis Engine

The core of Joint Vision is a sophisticated image analysis pipeline that:

- Preprocesses and normalizes knee X-ray images
- Passes them through an ensemble of five different vision models
- Aggregates predictions for robust KL grading and detection
- Generates visual explanations using Grad-CAM heatmaps

### LLM-Powered Clinical Assistant

Joint Vision integrates LLaMA-3 to:

- Translate technical findings into comprehensible clinical language
- Generate contextual recommendations based on detected KL grade
- Provide relevant treatment and follow-up suggestions
- Answer questions about the analysis in natural language

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/noobg0d/Knee-Osteoarthirits-detection-and-grading.git
   cd Knee-Osteoarthirits-detection-and-grading
   ```

2. Create and activate a virtual environment:
   ```bash
   conda create -n jointvision python=3.10
   conda activate jointvision
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the models:
   ```bash
   # Models are not provided - you must train your own models
   # See training scripts in the training/ directory
   python training/train_models.py
   ```

5. Install and run Ollama for local LLM deployment:
   ```bash
   # Install Ollama following instructions at https://ollama.ai/
   ollama run llama3
   ```

6. Launch the application:
   ```bash
   streamlit run app.py
   ```

7. Open the application in a browser at http://localhost:8501

## Technical Architecture

Joint Vision is built on a modular architecture:

- **Frontend**: Streamlit for interactive web interface
- **Vision Models**: PyTorch implementation of ViT, ResNet50, DenseNet-121, EfficientNet-B8, and VGG16 (all models require training by users)
- **Language Model**: LLaMA-3 deployed locally via Ollama
- **Integration Layer**: Custom middleware for vision-language coordination
- **Training Module**: Scripts for training your own models on knee X-ray datasets

## Model Performance

| Model | Balanced Accuracy |
|-------|-----------------|
| ResNet-50 | 63% |
| VGG-16 | 59% |
| Vision Transformer (ViT) | 69% |
| DenseNet-121 | 65% |
| EfficientNet | 66% |
| Xception | 72% |

## Project Structure

```
knee_OA_dl_app/
├── models/                # Model weights directory (users must train their own models)
│   ├── vit_model.pth       # You will need to train this yourself
│   ├── resnet50_model.pth  # You will need to train this yourself
│   ├── densenet121_model.pth # You will need to train this yourself
│   ├── efficientnetb8_model.pth # You will need to train this yourself
│   └── vgg16_model.pth     # You will need to train this yourself
├── xray_images/           # Sample and uploaded images
├── assets/                # UI assets and images
├── app.py                 # Streamlit application
├── utils/
│   ├── preprocessing.py   # Image preprocessing utilities
│   ├── inference.py       # Model inference functions
│   ├── visualization.py   # Heatmap generation
│   └── llm_interface.py   # LLaMA-3 integration
├── training/              # Model training scripts
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Future Enhancements

- Integration with PACS systems for direct image acquisition
- Support for additional joint types (hip, shoulder, etc.)
- Longitudinal tracking of patient progression
- Enhanced explainability with more detailed visual annotations
- Mobile application for point-of-care use



## License

This project is licensed under the MIT License - see the LICENSE file for details.
