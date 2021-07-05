# covid-19-detection
Detection of COVID-19 Pneumonia from chest X-Ray images. 

---

### Task
The goal is to classify Chest X-Ray images into 3 categories:
1. Normal
2. Viral Pneumonia
3. COVID-19

In essence, this is an **image classification problem**.

### Dataset
The dataset used for this task is the popular **COVID-19 Radiography Database** on [Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).

### Approach
The [ResNet-18](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/) CNN model is used for **Transfer Learning**. The network is pre-trained on the **ImageNet** database.

### Result
The model yielded > 97% accuracy when tested a large number of times.

