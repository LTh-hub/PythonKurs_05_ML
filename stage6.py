# stage6.py
#
# Onsdag 30/4 - Pythonprogrammering för AI-utveckling
# Video från förra kursomgången
# Dataset är en kopia från förra veckans YouTube jupyter 
# 

# pip install pandas scikit-learn numpy


# 1 Import data
# 2 Clean the data
# 3 Split data (70/30) - training / testing
# 4 Create the model
# 5 Train the model
# 6 Predictions
# 7 Evaluate and improve / SCORE
# 8 Create Presistant model - PERSISTANT MODEL



import pandas as pd                                     # hantering av data
import joblib



model = joblib.load('music-recommender.joblib')                      # färdigtränad modell läses upp från disk

input_data = pd.DataFrame([[21, 1]], columns=['age', 'gender'])     # tensorrepresentation av data 

prediction = model.predict(input_data)                              # använd modellen för att prediktera


print("#"*40, "\t", end="")
print(f"{prediction[0]  =  }")                                      # skriv ut prediktion



