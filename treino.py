import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import modelo
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
classificador = modelo.criar_classificador_gatos_e_cachorros()
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
classificador.save_weights("classificador.h5")