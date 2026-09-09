# Classificador de Gatos e Cachorros com Rede Convolucional (CNN)
Este sistema utiliza uma arquitetura construída com Keras e TensorFlow para extrair características visuais e classificar imagens binárias. Abaixo está a explicação aprimorada do fluxo do código, dividida entre o tratamento dos dados e a arquitetura do modelo.

## 1. Pré-processamento e Aumento de Dados (Data Augmentation)
o link do dataset é https://drive.google.com/drive/folders/1PX3zQFAA1dpSPPoT77jWGQflOJ0NOdTW 
Antes de treinar o modelo, as imagens (2.000 de gatos e 2.000 de cachorros) passam por transformações automáticas usando o ImageDataGenerator.

Normalização: Os pixels, originalmente em uma escala de 0 a 255, são divididos por 255 (escala 0 a 1). Isso acelera o treinamento e estabiliza os cálculos matemáticos da rede.

Redimensionamento: Todas as imagens são padronizadas para a resolução de 64x64 pixels com 3 canais de cor (RGB).

Data Augmentation: Para evitar overfitting (quando a rede apenas memoriza as imagens exatas), o código cria variações dinâmicas das imagens originais a cada época. Isso inclui rotação de até 7 graus, espelhamento horizontal, cisalhamento (20%), mudanças na altura (7%) e zoom (20%).

## 2. Arquitetura da Rede (O caminho da imagem)
A rede é sequencial e dividida em duas fases principais: a Extração de Características (camadas convolucionais) e a Classificação (camadas densas).

Fase de Extração (Convolução e Pooling)
Este bloco é repetido duas vezes no código para capturar desde padrões simples (como linhas e bordas) até padrões mais complexos (como formatos de orelhas ou focinhos).

Conv2D (Convolução): Utiliza 32 filtros (3x3) que deslizam pela imagem multiplicando seus pixels por pesos. Isso gera mapas de características.

Ativação ReLU: Aplicada logo após a convolução, essa função matemática zera todos os valores negativos e mantém os positivos intactos, permitindo que a rede aprenda padrões complexos e não-lineares.

Batch Normalization: Normaliza os dados gerados pela camada anterior (ajustando a média para zero e a variância para um). Isso evita que os valores fiquem muito grandes e acelera significativamente o treinamento.

MaxPooling2D: Um filtro (2x2) percorre o resultado anterior selecionando apenas o maior valor de cada quadrante. Isso reduz o tamanho da imagem pela metade, diminuindo o processamento necessário e focando nas características mais marcantes da imagem.

Fase de Classificação
Após as características da imagem serem mapeadas e reduzidas, elas precisam ser interpretadas para a decisão final.

Flatten (Achatamento): Pega a matriz 3D resultante das convoluções e a "achata", transformando-a em uma única linha (um vetor unidimensional). Isso é necessário porque as camadas densas seguintes só aceitam dados em 1D.

Dense (Camadas Ocultas): Duas camadas totalmente conectadas com 128 neurônios cada. Aqui, a rede combina todas as características extraídas para entender o que a imagem representa.

Dropout (20%): Durante o treinamento, desliga aleatoriamente 20% dos neurônios dessa camada. Isso força a rede a não depender de neurônios específicos, tornando-a mais generalista e resistente a ruídos.

Camada de Saída: Uma última camada Dense com apenas 1 unidade e função de ativação Sigmoid. Ela devolve um valor entre 0 e 1 (uma probabilidade).

## 3. Inferência (O Loop Final)
Após o modelo ser treinado e os pesos ajustados pelo otimizador Adam, o loop while True permite testes manuais:

Recebe o caminho de uma imagem.

Aplica o mesmo pré-processamento do treinamento (corta para 64x64 e normaliza dividindo por 255).

O modelo faz a predição (predict). Se o valor retornado for maior que 0.5, classifica como Gato (1); caso contrário, classifica como Cachorro (0).
