import numpy as np
from keras.utils import load_img, img_to_array
import modelo
classificador = modelo.criar_classificador_gatos_e_cachorros()
classificador.load_weights("classificador.h5")
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