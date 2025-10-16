# Real-time Detection of Urban Signs

### Overview
This project performs **real-time detection of urban signs** using the **YOLOv5n** model, optimized for **low power consumption** and **edge inference** on devices like the Jetson Nano or Raspberry Pi.  
It combines **OpenCV motion detection** with the YOLO model, allowing detection to run **only when movement is detected**, which greatly reduces **GPU and CPU usage** while maintaining high **frames per second (FPS)**.

---

### Features
- ⚡ **Lightweight YOLOv5n** model optimized for embedded hardware  
- 🎥 Real-time **video stream capture** via OpenCV  
- 🔍 **Motion-based triggering** to minimize unnecessary inference  
- 💾 Designed for **low-resource environments** with efficient memory management  

---

### Requirements
Install dependencies before running:
pip install ultralytics torch torchvision opencv-python imutils


---

### ▶️ How to Run
python SignDetector.py

### ⚙️ How It Works
1. **Frame Capture:** The system continuously captures frames from a camera (`cv2.VideoCapture(0)`).  
2. **Motion Detection:** Each new frame is compared to a reference frame using **grayscale conversion**, **Gaussian blur**, and **frame differencing** to detect movement.  
3. **Trigger:** When motion is detected (contour area ≥ 200), the YOLOv5n model performs inference only on that frame.  
4. **Detection:** Bounding boxes and class labels are drawn over detected urban signs.  
5. **Optimization:** If no movement is detected, inference is skipped — reducing **GPU/CPU load** and increasing **FPS**.  

You can include a diagram to visualize the workflow:
