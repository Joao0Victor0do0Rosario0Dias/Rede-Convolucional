# Rede-Convolucional
Um sistema que classifica imagens gatos e cachorros, por meio de uma rede neural convolucional(CNN).
## Pré processamento
A base de dados é composta de 2000 imagens de gatos e 2000 imagens de cachorros.
As imagens passam por um processo, a escala de 0 a 255 nos valores dos canais é transformada em valores de 0 a 1, as imagens são modificadas para terem a resolução de 64x64, também são criadas variações delas com mudanças como rotacionar em até 7 graus, espelhar horizontalmente, cisalhamento de até 20%, muda a altura em até 7% e aplica zoom de até 20%
## Explicação do funcionamento
A entrada é uma imagem de 64x64 pixels com os 3 canais rgb, essa entrada passa pela camada de convolução, depois normalização de lote e max pooling, aplica esse processo 2 vezes, depois passa pelo flatten, para passar depois por uma camada densa com 128 unidades com ativação relu e um dropout de 20% para repetir esse processo novamente e no final passa pela camada de saída com 1 unidade com ativação sigmoid.

A camada de convolução possui 32 filtros de 3x3 que passarão pela entrada pixel por pixel, multiplicando a entrada pelos pesos e aplicando a função de ativação "relu" que valor de entrada inalterado se ele for positivo, ou zero se for negativo, no final a saída para cada filtro será 62x62 e como são 32 filtros a saída da camada será (batch size, 62, 62, 32).
A normalização em lote calcula a média dos valores e variância, depois disso normaliza os valores subtraindo a média e dividindo pelo desvio padrão, no final aplica (valor * y) + b.
O max pooling possui um filtro (2,2), que passa pela entrada de forma que não é pixel por pixel, escolhendo o maior valor, como resultado reduz a imagem.
A camada Flatten recebe uma matriz e transforma em um vetor unidimensional.
A camada densa é uma camada totalmente conectada, ou seja conecta todas as unidades da camada anterior com todas as unidades da camada atual, os valores são multiplicados por pesos, depois somados e no final passam pela função de ativação.
O Dropout recebe valores e zera alguns deles, nesse caso zera 20% deles.
