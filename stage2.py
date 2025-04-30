# stage2.py
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



import pandas as pd                     # hantering av data
music_data = pd.read_csv('music.csv')
#print(music_data)


X = music_data.drop(columns=['genre'])  # feature
y = music_data['genre']                 # labels


print("="*30  +  "\n"  +  f"   {X = } ")
print("="*30  +  "\n"  +  f"   {y = } ")




