# Project Documentation

## 1. Overview

This folder contains the formal documentation of the Capstone project:

**Driver Behavior Anomaly Detection using Deep Learning for Accident Prevention**

The documentation describes the problem being addressed, project objectives, scope, research and technology gap, existing systems, literature survey, challenges, and the evolution of the project from Phase 1 to Phase 2.

---

## 2. Project Background

Road accidents are often associated with human factors such as driver drowsiness, distraction, inattention, and other unsafe behaviours.

The project aims to develop an intelligent driver monitoring system capable of analysing driver behaviour from real-time visual input and identifying potentially unsafe conditions.

The initial project concept focused broadly on deep learning-based driver anomaly detection. During the research and review process, the project direction was refined based on dataset availability, technology research, computational limitations, real-time requirements, and feedback received during project reviews.

---

## 3. Project Evolution

### Phase 1 – Research and Problem Understanding

The first phase focused primarily on:

- Understanding the driver behaviour anomaly problem.
- Studying existing driver monitoring systems.
- Conducting a literature survey.
- Exploring available datasets.
- Identifying suitable technologies and deep learning approaches.
- Studying research gaps and technical challenges.
- Defining the initial objectives and scope.

The initial technical direction considered deep learning approaches such as CNN, LSTM, Autoencoder, and other anomaly detection techniques.

---

### Phase 2 – Refined System Development

Based on further research, experimentation, and project guidance, the system was refined into a modular real-time driver monitoring architecture.

The refined approach focuses on combining multiple complementary components:

```text
Camera Input
     ↓
Frame Processing
     ↓
┌──────────────┬──────────────┬──────────────┐
│              │              │
Face           Face           YOLOv8
Recognition    Monitoring     Object Detection
│              │              │
Driver ID      EAR/MAR        Phone/Bottle/
               Head Pose      Cup/Objects
└──────────────┴──────────────┴──────────────┘
                    ↓
              Feature Extraction
                    ↓
               Feature Fusion
                    ↓
              Temporal Buffer
                    ↓
                   LSTM
                    ↓
          Driver Behaviour State
                    ↓
                 Alerting