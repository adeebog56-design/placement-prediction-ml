# Importing required modules

import pandas as pd
import joblib

# Loading Models

Logistic_Regression = joblib.load("Models/Placement_Logestic_Regression.pkl")
KNN = joblib.load("Models/Placement_KNN.pkl")
Random_Forest = joblib.load("Models/Placement_Random_Forest.pkl")

print("==========Placement_Prediction==========")
print()

# Sample Data

sample = pd.DataFrame({
    "IQ": [125],
    "Prev_Sem_Result": [8.4],
    "CGPA": [8.8],
    "Academic_Performance": [9],
    "Internship_Experience": ["Yes"],
    "Extra_Curricular_Score": [7],
    "Communication_Skills": [8],
    "Projects_Completed": [5]
})

# Predicting Labels

lr_prediction = Logistic_Regression.predict(sample)[0]
knn_prediction = KNN.predict(sample)[0]
rf_prediction = Random_Forest.predict(sample)[0]

print("=============RESULT=============")
print()

print("Logistic Regression :", lr_prediction)
print("KNN Classifier :", knn_prediction)
print("Random Forest :", rf_prediction)
print()

print("============= END =============")