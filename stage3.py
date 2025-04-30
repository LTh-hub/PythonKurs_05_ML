# stage3.py
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
# 7 Evaluate and improve
# 8 Create Presistant model



import pandas as pd                             # hantering av data
from sklearn.tree import DecisionTreeClassifier # Funktion av beslutsträd

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])  # feature
y = music_data['genre']                 # labels

model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict([[21, 1], [23, 0]])
print("="*40)
print(f"{predictions = }")


