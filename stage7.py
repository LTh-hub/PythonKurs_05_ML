# stage7.py
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
from sklearn.tree import DecisionTreeClassifier         # modell
from sklearn import tree                                # 
from graphviz import Source                             # beskriva hur trädet ser ut



music_data = pd.read_csv('music.csv')                   # läs in csv-fil
X = music_data.drop(columns=['genre'])                  # input data (träning)
y = music_data['genre']						            # output data (träning)


model = DecisionTreeClassifier()                        # model definition
model.fit(X, y)                                         # model anpassning



# Skapa grafisk representation av trädstrukturen - beskrivning sparad i txt-fil på disk
tree.export_graphviz(model, out_file='music-recommender.dot', 
                     feature_names=['age', 'gender'], 
                     class_names=sorted(y.unique()), 
                     label='all', 
                     rounded=True, 
                     filled=True)




# Läs från disk - innehållet från .dot-filen
with open("music-recommender.dot", "r") as f:
    dot_code = f.read()


# Skapa ett Graphviz-objekt från texten
graph = Source(dot_code)


# Visa grafen (fungerar både i Jupyter & VS Code)
graph.view()  # Skapar en .pdf och öppnar med standardvisare
graph.render("output-filename", format="png", view=True)  # Alternativt till PNG




