import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.preprocessing import StandardScaler


# Load in dataset
df = pd.read_csv("apple_quality.csv")

df = df[df['Acidity'] != 'Created_by_Nidula_Elgiriyewithana']
df['Acidity'] = pd.to_numeric(df['Acidity'])



# Remove outliers
X = df.loc[:, ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']]
Q1 = X.quantile(0.25)
Q2 = X.quantile(0.5)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
min = X.min()
max = X.max()

outliers = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))

def boundaries(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    print(feature)
    print(lower_limit, upper_limit)
    outliers = df[(df[feature] < lower_limit) | (df[feature] > upper_limit)]

for i in df[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']]:
    boundaries(df, i)

def remove_outliers(df, features):
    mask = pd.Series([True] * len(df))

    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        mask &= (df[feature] >= lower_limit) & (df[feature] <= upper_limit)

    return df[mask]


features = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']


cleaned_df = remove_outliers(df, features)


outliers_removed = len(df) - len(cleaned_df)



# Mlp model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV


Mlp_dataset = cleaned_df.iloc[:,1:]

X = Mlp_dataset.drop('Quality', axis=1)  
y = Mlp_dataset['Quality']  

# spliting the data 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mlp = MLPClassifier(hidden_layer_sizes=(16, 32, 64), activation='tanh', solver='adam', max_iter=500, random_state=42) #0.9353562005277045

# Train the model
mlp.fit(X_train, y_train)

# Predict on the test set
y_pred = mlp.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy= accuracy_score(y_test, y_pred)
print(accuracy)

# dump model to file
dump(mlp, 'mlp_model.joblib')

# scale model 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
dump(scaler, 'scaler.joblib')