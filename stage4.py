# stage4.py
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
# 8 Create Presistant model



import pandas as pd                                     # hantering av data
from sklearn.tree import DecisionTreeClassifier         # Funktion av beslutsträd
from sklearn.model_selection import train_test_split    # 
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])  # feature
y = music_data['genre']                 # labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)

print("#"*40, "\t", end="")
print(f"{predictions = }")

print("="*40, "\t", end="")
print(f"{score = }")

