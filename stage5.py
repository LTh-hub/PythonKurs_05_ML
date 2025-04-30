# stage5.py
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
from sklearn.tree import DecisionTreeClassifier         # Funktion av beslutsträd
import joblib

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])  # feature
y = music_data['genre']                 # labels

model = DecisionTreeClassifier()                    # Val av modell
model.fit(X, y)                                     # modellanpassning

joblib.dump(model, 'music-recommender.joblib')       # AI-modellen sparas på disk


