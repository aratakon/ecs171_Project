# ECS 171 Apple Classification Project

The goal of this project is to accurately predict the quality of an apple as 'good' or 'bad' based on the following features: size, weight, sweetness, crunchiness, juiciness, ripeness, acidity, and quality. This project involves the development of three supervised machine learning models: Multi-Layer Perceptron (MLP), Support Vector Machine (SVM), and Random Forest (RF). Each model is trained on an apple quality dataset obtained from Kaggle: <https://www.kaggle.com/datasets/markmedhat/apple-quality>. This is a highly rated, cleaned, and standardized dataset.

## Relevant Files
- **ECS 171 Project Data Preprocessing and EDA.ipynb**: A Jupyter notebook containing the data preprocessing, exporatory data analysis (EDA), and model development and evaluation steps.

- **apple_quality.csv**: A CSV file containing the complete apple quality dataset, including the following features: apple id, size, weight, sweetness, crunchiness, juiciness, ripeness, acidity, and quality.

- **html**: A folder containing the files for a simple HTML website that can be run locally.

## How to run the HTML website locally:
1. First download the 'html' folder, and move or copy the 'apple_quality.csv' file into the html folder. <br>
Now, move into the html folder.
   ```(bash)
   cd html
3. Create and activate a new virtual environment:
    ```(bash)
     pip install virtualenv
     python3 -m venv venv
     source venv/bin/activate
    ```
3. Install the necessary libraries:
   ```(bash)
   pip install tensorflow flask joblib pandas seaborn scikit-learn scikeras
   ```
4. Run the backend:
   ```(bash)
   python3 app.py
5. The frontend is live at the following link: <br>
    <http://127.0.0.1:5500/templates/index.html>
   
6. Enter values for all fields and make a prediction!
