import numpy as np 
import pandas as pd 
import mlflow
import sys
import random 
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

warnings.filterwarnings('ignore')

# Ambil dari path file preprocessing dari parameter command line
path_preprocessing = sys.argv[sys.argv.index("--path_preprocessing") + 1]
df = pd.read_csv(path_preprocessing)


X = df.drop('condition', axis=1 )
y = df['condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = random.randint(50, 300)
max_depth = random.randint(10, 20)


input_example = X_train.iloc[:5]

with mlflow.start_run() :  
    mlflow.autolog()

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train,y_train)

    accuracy = model.score(X_test,y_test)
    print(f"Model trained with n_estimators={n_estimators}, max_depth={max_depth}")
    print(f"Accuracy on test set: {accuracy:.4f}")

   