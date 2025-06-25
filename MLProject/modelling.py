import numpy as np 
import pandas as pd 
import mlflow
import random 
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("HeartDiagnosa")

df = pd.read_csv('preprocessing/HeartDiagnosa_preprocessing.csv')

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
    mlflow.log_metric('accuracy', accuracy)

    mlflow.sklearn.log_model(model, name="RandomForestClassifier" , input_example=input_example)

    print(f"Model trained with n_estimators={n_estimators}, max_depth={max_depth}")
    print(f"Accuracy on test set: {accuracy:.4f}")


    