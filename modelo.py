from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,BatchNormalization,MaxPooling2D,Dropout
def criar_classificador_gatos_e_cachorros(input_shape=(64,64,3)) -> Sequential:
    classificador=Sequential()
    classificador.add(Conv2D(32,(3,3),input_shape=input_shape,activation='relu'))
    classificador.add(BatchNormalization())
    classificador.add(MaxPooling2D(pool_size=(2,2)))
    classificador.add(Conv2D(32,(3,3),activation='relu'))
    classificador.add(BatchNormalization())
    classificador.add(MaxPooling2D(pool_size=(2,2)))
    classificador.add(Flatten())
    classificador.add(Dense(units=128,activation='relu'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=128,activation='relu'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=1,activation='sigmoid'))
    return classificador
