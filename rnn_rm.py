# Recurrent neural networks
#=================================================================================================================
# Data preprocessing

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the training data
# tato funkcia nam loaduje vsetky ceny z csvcka
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# ale my chceme pracovat len s OPEN cenami takze ich takto vyfiltrujeme
training_set = dataset_train.iloc[:, 1:2].values

# Features scaling - je to vlastne zmena range pre jednotlive data, kedy ich chceme dat do range medzi 0 a 1
from sklearn.preprocessing import MinMaxScaler
# toto vytvori objekt ktory na ktory ked aplikujeme nase traningove data tak ich da do rozmedzia medzi 0 a 1, 
# to hovori ten parameter feature_range = (0,1) 
sc = MinMaxScaler(feature_range = (0,1))
# tato funckia aplikuje scaling a transformuje nase training data a ulozi vysledok do novej premennej
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# toto znamena ze zoberieme 60 hodnot z training setu a pouzijeme ich ako vstupne hodnoty, 
# pricom tu 61 pouzijeme ako vystupnu hodnotu, cize prvych 60 hodnotu je vstup a vystupna hodnota pre tychto 60 vstupov je 61 hodnota vystupna hodnota
# tu si inicializujem 2 premenne pre X a y
X_train = []
y_train = []
# tento for loop zacina od 60 a konci poslednou hodnotou v training data. Zacina 60 lebo mame 60 timesteps
for i in range(60, 1258):
    # Toto tu prida do pola 60 hodnot ktore budu pouzite pri uceni ako vstupne hodnoty X pri uceni
    X_train.append(training_set_scaled[i-60:i, 0])
    # Toto tu prida do pola jednu hodnotu, v prvom loope to bude hodnota z miesta 61, ktora bude pouzita ako vystupna hodnota y pri uceni
    y_train.append(training_set_scaled[i, 0])
# toto tu urobi z premennych tyu LIST numpy ARRAY polia, cize zmeni ich typ tak aby sa dalo s nimi dalej pracovat
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape
# do 2D pola X_train pridavame 3 dimenziu co je vlastne pocet indikatorov ktore mame,
# v nasom pripade je to jeden indikator lebo pracujeme len s Open datami
# keby sme mali Open a Close data tak by sme tam museli dat 2 ako posledny parameter vo vnorenej zatvorke
# X_train.shape[0] toto mi vracia pocet riadkov v poli
# X_train.shape[1] toto mi vracia pocet stlpcov v poli
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#=================================================================================================================
# Building the RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# inicializacia RNN
# toto volame teraz regressor namiesto clasifier lebo teraz chcem na vystupe dostat linearne hodnoty
# cize vlastne akekolvek hodnoty a nie len napriklad macku alebo psa, cize klasifikaciu
regressor = Sequential()

# Adding first LSTM layer together with Dropout regularization
# tato funkcia pridava LSTM vrstvu s 50 neuronmi (units = 50) dalej kedze chceme aby dalsia vrstva bola
# LSTM vstrva musime nastavit return_sequences na True
# posledny parameter je input shape vstupnych dat ako mame po reshape v premennej X_train ale zadava sa tam len casova os a pocet indikatorov
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# Toto prida pre pridanu vrstvu Dropout regulrizaciu aby sme sa vyhli overfittingu, 
# Drop out znamena ze pri kazdej epoche sa ignoruje urcite pocet neuronov, v nasom pripade sme to nastavili
# na 20 percent co sme nastavili ako 0.2 vstupnu hodnotu
regressor.add(Dropout(0.2))

# Adding the second LSTM layer with regularisation
# v tejto vrstve uz nemusime zadavat input_shape lebo to sa zadava len ked predchadzajuca vrstva je vstupna vrstva
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer with regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer with regularisation
# tuto uz nemame true return_sequences lebo dalsia vstva uz nebude LSTM vrstva ale vystupna vrstva
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
# units = 1 znamena ze budem mat jednu vystupnu hodnotu, zaujimave je ze to nezadavam aka funkcia ma byt pouzita
# pravdepodobne kvoli tomu ze je to regresson problem a nie classification problem.
regressor.add(Dense(units = 1))

# Compiling the RNN
# pri kompilovani nastavujem optimazer, cize ako sa bude robit back propagation a ako druhy parameter
# nastavujeme cost function
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fiiting and learn the RNN
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)





#=================================================================================================================
# Making the prediction and visualization the results


#=================================================================================================================