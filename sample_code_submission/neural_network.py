from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np

class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """
    def __init__(self, train_data):
        self.model = Sequential()

        n_dim = train_data.shape[1]
        #Vecteur de dimension n_dim en entrée

        self.model.add(Dense(10, input_dim=n_dim, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        #Simule la fonction de perte
        self.scaler = StandardScaler()
        #nouveau paramètres: 
        self.epochs = 5

    def fit(self, train_data, y_train, weights_train=None):

        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        self.model.fit(
            X_train, y_train, sample_weight=weights_train, epochs=self.epochs, verbose=2
        )
    #epochs = nombre d'entraînement qu'on va faire
    #verbose = débugs, informations sur les entraînements, fonction de coût.

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data).flatten().ravel()
    