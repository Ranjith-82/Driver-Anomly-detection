## Dataset

This project uses the **YawDD (Yawning Detection Dataset)** for detecting driver drowsiness and yawning-related behavior.

The dataset was obtained from Kaggle and is used as part of the driver behavior analysis pipeline. It contains visual data that can be used to identify yawning behavior, which is an important indicator of driver fatigue and drowsiness.

### Dataset Source

The dataset used in this project is available on Kaggle:

**YawDD Dataset:**
https://www.kaggle.com/datasets/enider/yawdd-dataset

### Dataset Usage

The dataset is used in this project for:

* Detecting yawning behavior
* Extracting visual features related to driver fatigue
* Training and evaluating the drowsiness/anomaly detection pipeline
* Supporting real-time driver behavior analysis

### Dataset Structure

The dataset contains multiple files and supporting documentation. The complete dataset is **not included in this GitHub repository** because of its size.

To reproduce the experiments, download the dataset from the Kaggle source above and place it in the appropriate `data/` directory.

### Data Preprocessing

Before being used by the model, the dataset can be processed to:

1. Extract relevant frames/images.
2. Detect facial landmarks.
3. Calculate mouth-related features such as the **Mouth Aspect Ratio (MAR)**.
4. Identify frames associated with yawning behavior.
5. Generate feature sequences for temporal analysis.
6. Provide the processed features to the deep learning model.

### License

The Kaggle dataset page lists the dataset under the **MIT License**. Users should refer to the original dataset and Kaggle page for the applicable licensing and usage conditions.

### Note

The original dataset is not stored in this repository. Only the source and instructions for obtaining the dataset are provided to keep the GitHub repository lightweight and manageable.

**Source:** Kaggle – YawDD Dataset
