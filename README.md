# Breast-Cancer-detection
# 🧠 Breast Cancer Classification using ANN

This project implements an **Artificial Neural Network (ANN)** to classify breast cancer tumors as **benign or malignant** using the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset).

---

## 📝 Theory

Breast cancer is one of the most common cancers in women. Early and accurate diagnosis can significantly improve survival rates. Machine learning techniques are widely applied to medical datasets to assist in diagnosis.  

An **Artificial Neural Network (ANN)** is a machine learning model inspired by the human brain. It consists of layers of interconnected nodes (neurons):  
- **Input Layer** → takes features (e.g., cell size, texture, smoothness).  
- **Hidden Layers** → apply weights, biases, and activation functions to learn patterns.  
- **Output Layer** → predicts whether the tumor is benign (non-cancerous) or malignant (cancerous).  

This project uses TensorFlow/Keras to build an ANN for binary classification. The dataset is preprocessed, split into training and testing sets, and the model is evaluated using metrics like accuracy, precision, recall, and F1-score.  

---

## ✨ Features
- Implements an ANN using TensorFlow/Keras  
- Classifies tumors into **benign** or **malignant**  
- Evaluates model with accuracy, precision, recall, and F1-score  
- Generates confusion matrix and training/validation plots  
- Modular code structure for easy experimentation  

---

## 🛠️ Installation
Clone the repository:  
git clone https://github.com/your-username/breast-cancer-ann.git  
cd breast-cancer-ann  

Create and activate a virtual environment:  
python -m venv venv  
source venv/bin/activate   # Linux/Mac  
venv\Scripts\activate      # Windows  

Install dependencies:  
pip install -r requirements.txt  

---

## ▶️ Usage
Training the Model:  
python src/train.py  

Evaluating the Model:  
python src/evaluate.py  

Jupyter Notebook (interactive exploration):  
jupyter notebook notebooks/breast_cancer_ann.ipynb  

---

## 📊 Example Output
- **Accuracy:** ~97%  
- **Confusion Matrix:**  

|               | Predicted Benign | Predicted Malignant |  
|---------------|------------------|----------------------|  
| Actual Benign | 180              | 2                    |  
| Actual Malignant | 3             | 107                  |  

---

## 🔄 Workflow
Input Data → Preprocessing → Train/Test Split → ANN Model Definition → Training → Evaluation → Report Generation  

---

## 🤝 Contributing
Pull requests are welcome. For major changes, open an issue first to discuss what you’d like to change.  

---
