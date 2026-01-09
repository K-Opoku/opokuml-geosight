# 🌍 OpokuML-GeoSight: SOTA Terrain Intelligence

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/Inference-ONNX-005ced.svg)](https://onnx.ai/)
[![Docker Tag](https://img.shields.io/badge/Docker_Tag-opokuml--geosight-2496ed.svg)](https://www.docker.com/)

**OpokuML-GeoSight** is an end-to-end geospatial classification service that transforms raw satellite imagery into actionable environmental intelligence. Built with a **ConvNext Tiny** backbone and optimized for serverless inference, the system achieves a peak accuracy of **98.86%**.

---



## 1. 🎯 Business Logic & Project Objective

In environmental monitoring and urban planning, a simple classification label is rarely enough. The objective of **GeoSight** is to bridge the gap between "Pixels" and "Policy."

**The Problem:** Satellite imagery is often massive, blurry, or lacks context. Analysts need a system that filters out low-quality data and provides immediate recommendations based on what is detected.

**The GeoSight Solution:**
* **Quality Gatekeeping:** Uses OpenCV to automatically reject blurry images, saving compute costs and preventing false results.
* **High-Precision Mapping:** Differentiates between 10 complex terrain types (e.g., Annual Crops vs. Permanent Crops).
* **Actionable Intelligence:** Every prediction triggers a "Domain Insight" providing specific recommendations for agriculture, conservation, or infrastructure.

---

## 2. 📈 Final Model Performance

The model far surpassed the initial goal of 90% accuracy through strategic architectural shifts and hardware-specific tuning.

| Metric | Target | Result |
| :--- | :--- | :--- |
| **Accuracy** | > 90.00% | **98.86%** |
| **Inference Latency** | < 200ms | **Optimized (Single-Threaded)** |
| **Model Size** | Lightweight | **ONNX Optimized** |

---

## 3. 🔬 Deep Learning Pipeline (The "Grit")

### The Architecture Choice: ConvNext Tiny
I chose **ConvNext Tiny** after researching State-of-the-Art (SOTA) models for satellite imagery. It competes with Vision Transformers (ViTs) in accuracy but maintains the efficiency of a CNN, making it perfect for high-resolution satellite tiles.

### The Training Struggle & Pivot
1.  **The Failure:** I initially trained with **frozen weights** (ImageNet pre-training). Accuracy hit a ceiling of **<30%** because satellite features (top-down, multispectral) differ fundamentally from ImageNet (side-view objects like dogs or cars).
2.  **The Solution:** I unfroze the backbone, and trained the model entirely on the Eurosat dataset. I started getting the expected metric score.
3.  **The Compute Pivot:** Local training was taking hours without visible output. I migrated the dataset temporarily to **Google Colab** to leverage high-speed GPU acceleration, which allowed the model to converge rapidly.

### Hyperparameters & Preprocessing
* **Optimizer:** `AdamW` (Learning Rate: 1e-4, Weight Decay: 1e-4)
* **Batch Size:** 32
* **Resizing:** Used **Bicubic Interpolation** specifically to preserve edge sharpness in terrain boundaries.
* **Augmentation:** Random Rotation (15°), Color Jitter, and Horizontal Flips to ensure the model is invariant to satellite orientation.

---

## 4. 🛠️ Engineering & Production Optimizations

### 1. ONNX Inference Tuning
To prepare for serverless deployment, I converted the model to **ONNX**. So that the model will be light and safe to deploy serverlessly with no problem of latency. 
* **The Problem:** Default inference exhibited high latency spikes due to multi-threading overhead.
* **The Fix:** I manually forced the runtime to use single-threaded execution, ensuring stable performance in resource-constrained environments like AWS Lambda.
```python
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
```
### 2. The Hybrid Docker Strategy
I faced a "dependency hell" where the **AWS Lambda Python 3.12** environment conflicted with specific `onnxruntime` and `OpenCV` versions.

* **The Innovation:** I implemented a **Hybrid Dockerfile** strategy. In my dockerfile, aws wasn't compatible with my onnx version but when i upgrade the version to suit it, another conflict rise between open cv and the upgraded version of onnx.
* **The Execution:** I used `uv` and `uv.lock` for lightning-fast, reproducible installs of 95% of the environment to ensure consistency. Then, I used `pip` to surgically install the specific, compatible versions of the remaining libraries (onnx) required to run smoothly on the AWS Lambda base image. This hybrid approach still ensures reproducibility and compatibility if neccessary.

---

### 5. 📨 Feature Spotlight: Intelligent Insights
GeoSight doesn't just provide a label; it provides an automated strategy based on the prediction to assist in real-world decision-making.

| Terrain Class | Insight Description | Actionable Recommendation |
| :--- | :--- | :--- |
| **Annual Crop** | Detected seasonal farming land. | Monitor soil moisture; check for seasonal pest outbreaks. |
| **Forest** | High-density vegetation detected. | Monitor for deforestation or wildfire risks in dry seasons. |
| **Industrial** | Man-made structures/factories. | Evaluate urban heat island effect and runoff management. |

 ### 6. Streamlit app: 
After completion and containerization, I built a streamlit app on top of it. This app is an interactive user interface that lets users upload images easily and get a prediction with explanation of the prediction, confidence score and a visual representation of the prediction
---

### 🚀 How to Run Locally

#### 1. Build & Run via Docker
This builds the hybrid environment optimized for AWS Lambda compatibility:

```bash
# clone the repository
git clone https://github.com/K-Opoku/opokuml-geosight.git
cd opokuml-geosight

# Build the image
docker build -t opokuml-geosight .

# Run the container (Mapping Streamlit port)
docker run -p 9090:8080 opokuml-geosight
```



