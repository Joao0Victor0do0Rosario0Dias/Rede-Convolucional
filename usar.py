from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,BatchNormalization,MaxPooling2D,Dropout
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
import numpy as np
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
classificador=Sequential()
classificador.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
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
classificador.compile(optimizer='adam',loss='binary_crossentropy',
metrics=['accuracy'])
gerador_treinamento=ImageDataGenerator(rescale=1./255,#normalização 0 a 1
rotation_range=7,#rotação
horizontal_flip=True,#giros horizotais
shear_range=0.2, #mudança dos pixels para outra direção
height_shift_range=0.07,#faixa de mudança da altura
zoom_range=0.2)#zoom
gerador_teste=ImageDataGenerator(rescale=1./255)
base_Treinamento=gerador_treinamento.flow_from_directory('dataset/training_set',target_size=(64,64),batch_size=32,class_mode='binary')
base_Teste=gerador_teste.flow_from_directory('dataset/test_set',target_size=(64,64),batch_size=32,class_mode='binary')
classificador.fit(
    base_Treinamento,
    steps_per_epoch=int(4000/32),
    epochs=10,
    validation_data=base_Teste,
    validation_steps=int(1000/32)
)
#classificador.save_weights("classificador.h5")
while True:
    imagempath=input("Deseja classificar qual imagem? ")

    imagem_teste = load_img(imagempath,
                                  target_size = (64,64))
    imagem_teste = img_to_array(imagem_teste)
    imagem_teste /= 255
    imagem_teste = np.expand_dims(imagem_teste, axis = 0)
    previsao = classificador.predict(imagem_teste)
    previsao = (previsao > 0.5)#0= cachorro, gato = 1
    if previsao:
        print("Gato")
    else:
        print("Cachorro")
