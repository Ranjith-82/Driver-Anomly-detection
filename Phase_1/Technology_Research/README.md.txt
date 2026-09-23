# Technology Research

## 1. Overview

Technology research was carried out during the initial phase of the project to identify suitable computer vision and deep learning techniques for detecting abnormal driver behaviour.

The initial project direction focused on using deep learning approaches such as CNNs, LSTMs, and other anomaly detection techniques. Based on literature studies, dataset availability, technical feasibility, real-time requirements, and feedback received during project reviews, the technology direction was further refined.

The final system is designed as a modular real-time driver monitoring framework that combines facial analysis, object detection, feature extraction, temporal modelling, and alert generation.

---

## 2. Purpose of Technology Research

The main purposes of the technology research were:

- To identify suitable computer vision techniques for monitoring drivers.
- To study deep learning models applicable to driver behaviour analysis.
- To investigate methods for detecting drowsiness and fatigue.
- To study object detection techniques for identifying distraction-related objects.
- To understand video-based driver behaviour analysis.
- To identify technologies suitable for real-time processing.
- To study the computational limitations of running multiple models simultaneously.
- To select technologies that can later be integrated into a unified driver monitoring pipeline.

---

## 3. Technologies Investigated

The following technology areas were studied:

| Technology Area | Purpose |
|---|---|
| Computer Vision | Analyse visual information from the driver and camera frames |
| Deep Learning | Learn complex patterns related to driver behaviour |
| Driver Behaviour Detection | Identify abnormal or unsafe driver actions |
| Drowsiness Detection | Detect signs such as prolonged eye closure and yawning |
| Object Detection | Detect distraction-related objects such as mobile phones and bottles |
| Video-Based Analysis | Analyse driver behaviour across consecutive video frames |

Detailed research for each technology area is documented in the corresponding folders.

---

## 4. Computer Vision

Computer vision techniques were studied as the primary method for analysing visual information from the driver.

The research focused on facial landmark detection, facial feature analysis, head movement estimation, and real-time frame processing.

The selected approach uses facial landmarks to derive numerical features such as:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head pose
- Eye state
- Facial movement

These features provide useful information for identifying behaviours such as drowsiness, yawning, talking, and distraction.

---

## 5. Deep Learning

Different deep learning approaches were investigated as part of the project research.

The initial research considered models such as:

- CNN
- LSTM
- Autoencoder
- ResNet
- Other deep learning approaches reported in the literature

CNN-based approaches are useful for extracting spatial information from images, while LSTM models are suitable for analysing temporal sequences.

Based on the project's requirement to analyse driver behaviour over time, temporal modelling using LSTM is considered for the later stage of the system.

---

## 6. Driver Behaviour Detection

Driver behaviour detection focuses on identifying behaviours that may indicate unsafe or abnormal driving.

The project considers behaviours such as:

- Normal driving
- Drowsiness / fatigue
- Yawning
- Talking
- Distraction
- Mobile phone usage
- Other distraction-related activities

The research showed that individual video frames may not always be sufficient to determine whether a behaviour is truly abnormal.

Therefore, the refined approach considers both:

**Spatial information**

and

**Temporal information**

to improve behaviour understanding.

---

## 7. Drowsiness Detection

Drowsiness detection techniques were studied to identify signs of driver fatigue.

Facial features such as eye and mouth movements are particularly useful for this purpose.

The project uses facial landmarks to calculate:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Eye closure duration
- Blink-related information
- Head pose information

A key consideration is that a single frame showing closed eyes should not automatically be considered drowsiness.

Instead, eye closure needs to be analysed over a period of time. This motivates the use of temporal analysis and sequence modelling in the later stages of the project.

---

## 8. Object Detection

Object detection was investigated to identify objects that may indicate driver distraction.

The project focuses on detecting objects such as:

- Mobile phone
- Water bottle
- Coffee cup
- Other relevant objects

YOLO-based object detection was studied because the system requires object detection from live video while maintaining real-time performance.

YOLOv8 was selected for the implementation of the object detection module.

The object detection module is designed to operate independently from the facial monitoring module.

---

## 9. Video-Based Analysis

Since the project works with live camera/video input, video-based analysis is an important part of the system.

Instead of treating every frame as an independent observation, consecutive frames can be used to understand how driver behaviour changes over time.

The refined architecture therefore considers:

```text
Video Input
     ↓
Frame Processing
     ↓
Feature Extraction
     ↓
Feature Vector
     ↓
Temporal Sequence / Buffer
     ↓
LSTM
     ↓
Behaviour Classification