# Dataset Exploration

## Overview

During Phase 1 of the project, we explored publicly available datasets
related to driver behavior, driver distraction, drowsiness, and anomalous
driving behavior.

The main objective of the dataset exploration was to identify datasets
that can support the development of our proposed system:

**Driver Behavior Anomaly Detection using Deep Learning for Accident Prevention**

The datasets were evaluated based on:

- Types of driver behaviors available
- Number and diversity of samples
- Camera viewpoints
- Day and night conditions
- Available modalities
- Dataset size
- Suitability for deep learning
- Relevance to driver distraction and anomaly detection
- Practicality of using the dataset for our project

During Phase 1, two major datasets were investigated:

1. State Farm Distracted Driver Detection Dataset
2. Driver monitoring/anomaly detection dataset (DAD / 3MDAD)

---

# 1. State Farm Distracted Driver Detection Dataset

## 1.1 Overview

The State Farm Distracted Driver Detection dataset was released as
part of a Kaggle competition focused on automatically identifying
distracted driving behavior from images captured inside a vehicle.

The dataset contains images of drivers performing different activities
while inside a car. The objective is to classify the driver's activity
into one of ten predefined classes.

The dataset is particularly relevant to our project because several
of its classes represent common forms of driver distraction.

## 1.2 Dataset Size

According to the Kaggle dataset information:

- Total files: approximately 102,152
- Dataset size: approximately 4.31 GB
- Image format: JPG
- Metadata/labels: CSV
- Training images: approximately 22,424 labeled images
- Test images: approximately 79,726 images

The Kaggle dataset also provides `driver_imgs_list.csv`, which contains
information about the training images, driver IDs, and class IDs.

## 1.3 Driver Behavior Classes

The dataset contains 10 classes:

| Class | Driver Behavior |
|-------|-----------------|
| c0 | Safe driving |
| c1 | Texting - right hand |
| c2 | Talking on the phone - right hand |
| c3 | Texting - left hand |
| c4 | Talking on the phone - left hand |
| c5 | Operating the radio |
| c6 | Drinking |
| c7 | Reaching behind |
| c8 | Hair and makeup |
| c9 | Talking to passenger |

These classes cover both normal driving and several forms of
distracted driving.

## 1.4 Camera Perspective

The dataset consists of images captured from a camera mounted inside
the vehicle.

The camera provides a relatively consistent view of the driver,
allowing the driver's posture, hands, face and activities to be
analyzed.

## 1.5 Dataset Characteristics

Important characteristics of the State Farm dataset include:

- RGB driver images
- Single primary camera perspective
- Driver activity classification
- 10 behavior classes
- Different forms of manual distraction
- Multiple drivers
- Training and test data separated by driver

The separation by driver is important because the same driver is not
intended to appear in both the training and test sets.

## 1.6 Advantages

### Advantages

- Relatively manageable dataset size compared with very large
  video datasets.
- Clearly defined behavior classes.
- Contains both safe and distracted driving.
- Contains several common distraction behaviors.
- Suitable for image-based deep learning classification.
- Useful for developing an initial driver distraction detection model.
- Labels are already provided for the training data.

## 1.7 Limitations

### Limitations

- Primarily based on a single camera viewpoint.
- Mainly focuses on visible driver activities.
- Does not provide multiple synchronized camera views.
- Does not directly provide physiological signals.
- Does not cover every type of abnormal driving behavior.
- The dataset was collected in a controlled environment rather than
  representing every possible real-world driving condition.
- It does not provide extensive night-time/multispectral coverage.

## 1.8 Relevance to Our Project

The State Farm dataset is highly relevant for detecting
**driver distraction**.

It can potentially support detection of behaviors such as:

- Phone usage
- Texting
- Drinking
- Looking/reaching behind
- Interacting with the radio
- Talking to passengers
- Other visible distractions

These behaviors form an important part of the broader driver behavior
anomaly detection problem.

However, the dataset alone may not be sufficient for our complete
project because our proposed system aims to consider a broader range
of driver states and behaviors.

---

# 2. Driver Anomaly Detection (DAD) Dataset

## 2.1 Overview

The Driver Anomaly Detection (DAD) dataset was developed specifically
for driver anomaly detection and open-set recognition.

The main motivation behind the dataset is to distinguish normal
driving behavior from anomalous behaviors, including anomalous
behaviors that were not necessarily seen during training.

This makes the dataset particularly relevant to our project because
our objective is not limited to recognizing a fixed set of activities.
We are interested in identifying abnormal or unsafe driver behavior.

## 2.2 Dataset Characteristics

The official DAD dataset has the following characteristics:

- 31 subjects
- Front and top camera views
- Depth and infrared modalities
- Synchronized multi-view recordings
- 45 frames per second
- Resolution of 224 × 171 pixels
- Normal and anomalous driving recordings
- Video-based rather than only individual image classification

The dataset was recorded using a driving simulator with two
Infineon CamBoard pico flexx cameras.

The front camera captures the driver's head, body and visible hand
movements, while the top camera focuses more strongly on hand
movements.

## 2.3 Training and Test Data

The DAD dataset contains:

### Training

- 25 subjects
- Approximately 550 minutes of normal driving
- Approximately 100 minutes of anomalous driving

### Test

- 6 subjects
- Approximately 88 minutes of normal driving
- Approximately 45 minutes of anomalous driving

An important characteristic of the test set is that it contains
anomalous actions that were not present in the training set.

This makes the dataset suitable for studying open-set anomaly
detection.

## 2.4 Dataset Size

The complete official DAD dataset is approximately **95 GB**.

Our locally downloaded copy/subset may be smaller depending on which
parts, modalities, views or samples were downloaded.

Therefore, the size of the local copy should not be confused with the
size of the complete official dataset.

## 2.5 Modalities

The DAD dataset provides:

- Depth data
- Infrared data

The use of infrared and depth information makes the dataset useful
for studying driver behavior under different lighting conditions.

## 2.6 Advantages

### Advantages

- Specifically designed for driver anomaly detection.
- Supports open-set anomaly detection.
- Contains both normal and anomalous driving.
- Multiple camera views.
- Depth and infrared modalities.
- High temporal resolution.
- Contains anomalous actions that are not present during training.
- More suitable for video-based behavioral analysis than a simple
  image classification dataset.

## 2.7 Limitations

### Limitations

- Very large dataset size.
- Higher storage requirements.
- More computationally demanding.
- Data is collected using a driving simulator.
- Depth and infrared processing introduces additional complexity.
- The dataset structure is more complicated than a simple image
  classification dataset.

## 2.8 Relevance to Our Project

The DAD dataset is relevant to our project because it focuses directly
on distinguishing normal driving from anomalous behavior.

It can potentially help us investigate:

- Temporal driver behavior
- Abnormal behavior detection
- Open-set recognition
- Multi-view driver monitoring
- Hand movement analysis
- Driver state changes

---

# 3. Important Dataset Identification Note

During our initial exploration, we also encountered a dataset commonly
referred to as **3MDAD (Multimodal Multiview and Multispectral Driver
Action Dataset)**.

This dataset should not be confused with the DAD dataset.

3MDAD has characteristics that are different from the official DAD
dataset.

## 3MDAD Characteristics

3MDAD contains:

- Daytime recordings
- Nighttime recordings
- Front view
- Side view
- RGB data
- Depth data
- Multiple driver activities
- 16 in-vehicle action classes

The 16 activities include:

1. Safe driving
2. Doing hair and makeup
3. Adjusting radio
4. GPS operation
5. Writing a message using the right hand
6. Writing a message using the left hand
7. Talking on the phone using the right hand
8. Talking on the phone using the left hand
9. Taking a picture
10. Talking to a passenger
11. Singing or dancing
12. Fatigue and somnolence
13. Drinking using the right hand
14. Drinking using the left hand
15. Reaching behind
16. Smoking

The combination of front/side views and day/night conditions makes
3MDAD particularly useful for studying driver behavior under changing
visual conditions.

## Important

Before finalizing our project documentation, we will verify whether
the dataset currently available to our team is:

- DAD (Driver Anomaly Detection), or
- 3MDAD (Multimodal Multiview and Multispectral Driver Action Dataset).

This distinction is important because the two datasets have different
camera configurations, modalities and objectives.

---

# 4. Dataset Comparison

| Feature | State Farm | DAD | 3MDAD |
|--------|------------|-----|-------|
| Main purpose | Distracted driver classification | Driver anomaly detection | Driver action/distraction analysis |
| Data type | Images | Video | Video/image sequences |
| Normal driving | Yes | Yes | Yes |
| Distracted behavior | Yes | Yes | Yes |
| Camera views | Mainly single view | Front + Top | Front + Side |
| Night data | Limited | Infrared modality | Yes |
| RGB | Yes | No primary RGB modality | Yes |
| Depth | No | Yes | Yes |
| Infrared | No | Yes | No |
| Number of classes/actions | 10 | Normal + anomalous actions | 16 actions |
| Temporal information | Limited | Strong | Strong |
| Approx. size | 4.31 GB | ~95 GB complete dataset | Large multimodal dataset |
| Main strength | Clear distraction classes | Open-set anomaly detection | Multiview + day/night analysis |
| Main limitation | Single-view image data | Large and computationally expensive | Large and complex |

---

# 5. Comparison Based on Our Project Requirements

Our project aims to develop a driver behavior anomaly detection system
capable of identifying unsafe or abnormal driver behavior.

The important requirements identified during Phase 1 are:

- Driver distraction detection
- Driver activity recognition
- Detection of abnormal behavior
- Robustness to different viewpoints
- Handling of different lighting conditions
- Potential real-time processing
- Deep learning compatibility
- Possibility of extending the system to multiple behavior classes

Based on these requirements, each dataset provides different
advantages.

### State Farm

State Farm is useful for:

- Initial driver distraction classification
- Image-based model development
- Recognizing clearly defined distraction classes
- Developing and testing baseline models

### DAD

DAD is useful for:

- Driver anomaly detection
- Normal vs anomalous behavior
- Temporal analysis
- Open-set recognition
- Multi-view analysis
- Depth and infrared based driver monitoring

### 3MDAD

3MDAD is useful for:

- Day/night driver monitoring
- Front/side view analysis
- Multimodal analysis
- Driver action recognition
- Studying changes caused by lighting and viewpoint

---

# 6. Initial Phase 1 Findings

From the dataset exploration, we observed that no single dataset
necessarily provides every characteristic required by our proposed
system.

The State Farm dataset provides clearly labeled distraction
categories and is comparatively easier to work with.

The DAD dataset provides a stronger focus on anomaly detection and
normal-versus-anomalous behavior, along with multiple modalities and
views.

The 3MDAD dataset provides useful multiview and day/night information
for driver action recognition.

Therefore, the final dataset strategy will be decided after further
evaluation of:

- Required behavior classes
- Data quality
- Available labels
- Camera viewpoint requirements
- Computational resources
- Model architecture
- Training time
- Real-time inference requirements

---

# 7. Dataset Selection Status

**Status: Under Evaluation**

At the current Phase 1 stage, the datasets have been explored and
compared, but the final dataset selection will be made after
considering the requirements of the proposed system and the available
computational resources.

The selected dataset(s) will be documented in a later project phase
after preprocessing and experimentation.

---

# 8. Sources

## State Farm Distracted Driver Detection

Dataset: State Farm Distracted Driver Detection  
Platform: Kaggle  
Purpose: Distracted driver behavior classification

## Driver Anomaly Detection (DAD)

Dataset: Driver Anomaly Detection Dataset  
Institution: Technical University of Munich  
Purpose: Driver anomaly detection and open-set recognition

## 3MDAD

Dataset: Multimodal Multiview and Multispectral Driver Action Dataset  
Purpose: Driver action/distraction analysis under different views,
modalities and lighting conditions.

---

# 9. Phase 1 Conclusion

The dataset exploration provided an understanding of the available
public datasets for driver monitoring and driver behavior analysis.

The State Farm dataset provides a strong starting point for
recognizing common distracted driving behaviors.

The DAD dataset provides a stronger foundation for anomaly detection
and normal-versus-anomalous behavior analysis.

The 3MDAD dataset provides additional advantages through multiple
views, modalities and day/night recordings.

These findings will be used to guide the dataset selection,
preprocessing strategy and model development in the subsequent phases
of the project.s