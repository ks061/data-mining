import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD
from scikeras.wrappers import KerasClassifier

names = ["target","alcohol","malic_acid","ash","alcalinity","Mg","tot_phenols","flavanoids",
         "non_flav_phenols","proanthocyanins","color_intensity",
         "hue","OD","proline"]

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None,names=names)

df_wine.target = pd.Categorical(df_wine.target, ordered=False)

for column in df_wine.columns:
    if df_wine[column].dtype==float:
        df_wine[column] = pd.to_numeric(df_wine[column], downcast='float')
    elif df_wine[column].dtype==int:
        df_wine[column] = pd.to_numeric(df_wine[column], downcast='integer')
        
sclr = StandardScaler()
X = sclr.fit_transform(df_wine.drop(columns="target"))
X = pd.DataFrame(X, 
                 columns=df_wine.drop(columns="target").columns)
y = df_wine["target"]
y_bin = pd.get_dummies(y)

def create_keras_model(num_hidden, activation):

    # wine has 13 predictor variables
    inputs = Input(shape=(13,))
    
    # our hidden layer is parameterized
    layer = Dense(num_hidden, activation=activation)(inputs)
    
    # we have a 3-class problem, and we'll hard-code softmax activation
    outputs = Dense(3,activation="softmax")(layer)
    
    # Build our model
    model = Model(inputs=inputs, outputs=outputs, name="model_1")

    return model

kclf = KerasClassifier()

clf = KerasClassifier(
      model=create_keras_model, 
      verbose=0, 
      epochs=5, 
      batch_size=4,
      optimizer='adam', # compile param
      loss='categorical_crossentropy', # compile param
      metrics=["accuracy"], # compile param
      num_hidden=10, # model param
      activation='relu' # model param
) 

# I want to use StratifiedKFold with shuffle to get
# randomly sorted data with stratified (representative)
# samples. It is my computer scientific artistic license!
y_pred = cross_val_predict(
    clf, X, y,
    cv=KFold(n_splits=10, shuffle=True),
    method='predict'
)

num_in_list = 3 # approximately (might be less due to
                # rounding and such below)
learning_rate = 0.001
momentum=0.45
param_grid = {
    "optimizer__" : 
    [
        SGD(learning_rate=learning_rate, 
            momentum=momentum),
        SGD(learning_rate=learning_rate, 
            momentum=momentum*2),
        SGD(learning_rate=learning_rate*2,
            momentum=momentum),
        SGD(learning_rate=learning_rate*2,
            momentum=momentum*2)
    ],
    "model__num_hidden" : 
    [i for i in range(14, 140, int((140-14)/num_in_list))],
    'model__activation' : 
    ["relu", "tanh"],
    'epochs' : 
    [i for i in range(50, 100, int((100-50)/num_in_list))],
    'batch_size' : 
    [i for i in range(1, y.size, int(y.size/num_in_list))]
}

grid = GridSearchCV(
    clf, 
    param_grid, 
    return_train_score=True, 
    cv=10, # question mentions 5 OR 10, so I'll choose 10!
    verbose=2,
    n_jobs=6
)
grid_result = grid.fit(X,y)