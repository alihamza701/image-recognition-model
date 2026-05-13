# 🖼️ Image Recognition Model — CIFAR-10

A **CNN-based image classifier** that recognizes 10 categories of objects including animals, birds, and vehicles. Built with PyTorch, featuring data augmentation, batch normalization, dropout regularization, and a cosine annealing learning rate scheduler.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Overview

This project trains a deep convolutional neural network on the **CIFAR-10** dataset to classify images into 10 categories:

| Label | Class | Label | Class |
|---|---|---|---|
| 0 | ✈️ Airplane | 5 | 🐕 Dog |
| 1 | 🚗 Automobile | 6 | 🐸 Frog |
| 2 | 🐦 Bird | 7 | 🐴 Horse |
| 3 | 🐱 Cat | 8 | 🚢 Ship |
| 4 | 🦌 Deer | 9 | 🚛 Truck |

---

## ✨ Features

- ✅ **Data Augmentation** — Random crop, horizontal flip, color jitter for better generalization
- ✅ **Batch Normalization** — Stable training across all conv layers
- ✅ **Dropout Regularization** — Spatial dropout in conv blocks + FC dropout
- ✅ **AdamW + Cosine LR Scheduler** — Modern optimizer with learning rate annealing
- ✅ **Label Smoothing** — Reduces overconfidence, improves generalization
- ✅ **Early Stopping** — Automatically stops when validation accuracy plateaus
- ✅ **Best Model Checkpointing** — Saves the best-performing weights
- ✅ **Confusion Matrix & Classification Report** — Per-class performance breakdown
- ✅ **Visual Inference** — Sample predictions displayed with confidence scores

---

## 🏗️ Model Architecture

```
Input: (3, 32, 32)
│
├── Block 1: Conv(3→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout2d
├── Block 2: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool → Dropout2d
├── Block 3: Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool → Dropout2d
│
└── Classifier: Flatten → FC(4096→1024) → ReLU → Dropout
                         → FC(1024→512) → ReLU → Dropout
                         → FC(512→10)
```

**Total Parameters:** ~7.2M

---

## 📦 Requirements

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn jupyter
```

Tested with Python 3.10+, PyTorch 2.x. GPU recommended but not required.

---

## 🚀 Quickstart

**1. Clone the repo:**
```bash
git clone https://github.com/alihamza701/image-recognition-model.git
cd image-recognition-model
```

**2. Launch the notebook:**
```bash
jupyter notebook image_recognition_model.ipynb
```

**3. Run all cells.** CIFAR-10 downloads automatically (~170 MB). Training takes ~10–15 min on GPU, ~45–60 min on CPU.

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~82–85% |
| Best Val Accuracy | ~83–86% |
| Epochs (typical) | ~25–30 |

> Results may vary slightly depending on hardware and random seed.

---

## 📁 Project Structure

```
image-recognition-model/
│
├── image_recognition_model.ipynb   # Main training notebook
├── best_model.pth                  # Best checkpoint (generated after training)
├── image_recognition_checkpoint.pth # Full deployment checkpoint
├── training_curves.png             # Loss & accuracy plots (generated)
├── confusion_matrix.png            # Confusion matrix (generated)
└── README.md
```

---

## 🔮 Inference Example

```python
import torch
from torchvision import transforms
from PIL import Image

# Load checkpoint
checkpoint = torch.load('image_recognition_checkpoint.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess your image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

img = Image.open('your_image.jpg')
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    logits = model(tensor)
    pred = logits.argmax(1).item()

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print(f'Predicted: {CLASSES[pred]}')
```

---

## 🛠️ Possible Improvements

- Replace custom CNN with **ResNet-18** via transfer learning for ~93%+ accuracy
- Add **Grad-CAM** visualizations to see what the model focuses on
- Deploy as a **Gradio** or **Streamlit** web app
- Experiment with **MixUp** or **CutMix** augmentation

---

## 👤 Author

**Ali Hamza**
- GitHub: [@alihamza701](https://github.com/alihamza701)
- LinkedIn: [alihamzarasheed](https://www.linkedin.com/in/alihamzarasheed/)
- Kaggle: [alihamzarasheed](https://www.kaggle.com/alihamzarasheed)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
