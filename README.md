# Project Description 

This project is a deep learning-based web application for classifying chicken feces into four categories: Coccidiosis, Healthy, Newcastle Disease, and Salmonella. The classifier achieves an impressive 97% accuracy using a fine-tuned VGG16 architecture, served through an intuitive Streamlit UI and fully containerized using Docker.

### 🔍 Project Overview
📊 1. **Data Collection**
Dataset was collected from [Kaggle](https://www.kaggle.com/](https://www.kaggle.com/datasets/efoeetienneblavo/chicken-disease-dataset), containing over 8000 labeled images of chicken feces. The four classes include:
- Coccidiosis
- Healthy
- Newcastle Disease
- Salmonella

### 🧠 2. Model Development
- Initially built a **custom CNN from scratch**, achieving around **95% accuracy**.
- Observed **class imbalance** in the dataset. Tried handling it using:
  ```python
  from sklearn.utils.class_weight import compute_class_weight

https://chicken-disease-classification.streamlit.app/



