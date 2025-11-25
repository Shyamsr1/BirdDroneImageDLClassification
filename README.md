# 🕊️🛸 Bird vs Drone Classification & Detection (DL + YOLOv8)  
### Aerial Object Classification System – Streamlit Web App

This project is a **Deep Learning–powered Aerial Object Classification System** capable of:
- **Bird vs Drone Image Classification** (CNN + MobileNetV2)
- **YOLOv8 Object Detection** for locating drones/birds in images
- **Interactive Streamlit Web App** for real-time predictions

The system is optimized for **production deployment** (HuggingFace Spaces / Streamlit Cloud) with lightweight models and a clean folder structure.

---

## 🚀 Key Features

### **✔ 1. Dual Model Classification**
- **Custom CNN model** (trained on 300+ aerial images)
- **MobileNetV2 Transfer Learning** model for higher accuracy and faster inference
- Supports **Bird**, **Drone**, and **Uncertain** predictions  

### **✔ 2. YOLOv8 Detection**
- Detects drones or birds inside uploaded images  
- Uses **Ultralytics YOLOv8n** for lightweight, fast inference  
- Only enabled when compatible images are uploaded  

### **✔ 3. Streamlit Application**
- Clean UI  
- Model selection (CNN / MobileNetV2 / YOLOv8)  
- Automatic clearing of images & results  
- Dynamic enabling/disabling of YOLO button  
- Model comparison table  

### **✔ 4. Production-Optimized**
- Small footprint: **< 50 MB**  
- Only inference models included  
- No large datasets or training notebooks  
- Requirements minimized for cloud deployment  

---

## 📁 Project Structure

```
BirdDroneImageDLClassification/
│
├── streamlit_app/
│   ├── app.py
│
├── models/
│   ├── cnn_best_model.h5
│   ├── mobilenet_best_model.h5
│
├── runs/
│   └── detect/
│       └── bird_drone_yolov8/
│
├── yolov8n.pt
├── requirements.txt
├── README.md

```

### ❗Not included (intentionally removed for production)
- Raw datasets  
- Training notebooks  
- Intermediate checkpoints  
- Large test sets  

---

## 🧠 Model Architecture Overview

### **1️⃣ CNN Model**
- Built using TensorFlow/Keras  
- Input: 224×224 RGB  
- Conv2D → MaxPool → Dropout → Dense  
- Softmax output for two classes  

### **2️⃣ MobileNetV2 Transfer Learning**
- Pretrained on ImageNet  
- Fine-tuned final layers  
- Faster + more accurate than CNN  
- Best for real-time inference  

### **3️⃣ YOLOv8n Detection Model**
- Ultralytics YOLOv8 nano model  
- Very fast, tiny (6MB), production-friendly  
- Detects:
  - Bird  
  - Drone  
- Outputs bounding boxes + labels + confidence  

---

## ▶️ How to Run Locally

### **1. Create Environment**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### **2. Install Requirements**
```bash
pip install -r requirements.txt
```

### **3. Run Streamlit App**
```bash
streamlit run streamlit_app/app.py
```

---

## 🌐 Deploying to HuggingFace / Streamlit Cloud

### **HuggingFace Deployment**
1. Upload the folder to a new HF Space  
2. Set Space Type = “Streamlit”  
3. Add the following in **`README.md`**:
   ```
   ---  
   title: Bird vs Drone Detection  
   emoji: 🕊️  
   colorFrom: blue  
   colorTo: red  
   sdk: streamlit  
   app_file: streamlit_app/app.py  
   pinned: false  
   ---  
   ```
4. Push code → Space auto-deploys

### **Streamlit Cloud**
1. Push project to GitHub  
2. Create new Streamlit Cloud app  
3. Select `streamlit_app/app.py`  
4. Deploy → Done  

---

## 📊 Evaluation & Metrics

| Model          | Accuracy | Speed | Notes |
|----------------|----------|--------|-------|
| CNN            | ~91%     | Fast   | Lightweight custom model |
| MobileNetV2    | ~96%     | Very Fast | Best classification accuracy |
| YOLOv8n        | High     | Real-time | Detects bird/drone regions |

---

## 🧪 Demo Flow

### Step 1 — Upload Image  
User uploads a bird/drone image.

### Step 2 — Select Model  
- CNN  
- MobileNetV2  
- YOLOv8  

### Step 3 — Get Output  
- If classification → display predicted class + confidence  
- If YOLO → show bounding box results  
- Model comparison table auto-updates  
- Image + outputs clear when new model chosen  

---

## 🗂 Requirements

Example `requirements.txt`:

```
streamlit==1.32.0
tensorflow==2.12.0
numpy==1.24.3
pillow==10.0.0
ultralytics==8.0.196
opencv-python-headless==4.8.1.78
```

---

## 📌 Future Improvements
- Add drone type classification  
- Add video inference  
- Add real-time webcam detection  
- Add dataset augmentation module  
- Deploy as API service  

---

## 👨‍💻 Author
**Shyam Sirugudi Ramaswamy**  
AI/ML Developer & Data Science Intern  
- GitHub: https://github.com/Shyamsr1  
- LinkedIn: https://www.linkedin.com/in/shyam-sirugudi-ramaswamy/  

---

## ⭐ If you like the project  
Please consider giving it a **GitHub star**!  
