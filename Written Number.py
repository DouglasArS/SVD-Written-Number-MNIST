#!/usr/bin/env python
# coding: utf-8




# Pegando os labels (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

import gzip
f = gzip.open('train-labels-idx1-ubyte.gz','r')

import numpy as np
f.read(8)
buf = f.read(60000)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
labels = data





# Index para grafico (Eixo x)
index = np.arange(0, 10)

# Quantidade de cada digito (Eixo Y)
qtd_number = []

for i in range(0, 10):
    qtd_number.append(np.count_nonzero(labels == i))

# Gerando grafico com numeros de imagens de cada digito

import matplotlib.pyplot as plt

plt.bar(index, qtd_number, color="blue")

plt.xticks(index)
plt.ylabel("Quantidades")
plt.xlabel("Numeros")
plt.title("Quantidade de Imagens por Digitos (x = Digitos de 1 a 9, y = Qtd_imagens)")
plt.show()






# Plotando Imagens Teste

import gzip
f = gzip.open('t10k-images-idx3-ubyte.gz','r')

image_size = 28

num_images = int(input("Escolha quanto imagens mostrar: "))

import numpy as np
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

import matplotlib.pyplot as plt

for i in range(0, num_images):
    print("Foto {}".format(i+1))
    print()
    image = np.asarray(data[i]).squeeze()
    plt.imshow(image)
    plt.show()

# Escolha a sua Imagem Teste
i = int(input("Escolha sua foto: "))
image = np.asarray(data[i-1]).squeeze()





# Descobrindo a foto a partir do Conjunto Treino

## Passo1: Trasformação da imagem teste escolhida e de todas as imagens do
## Conjunto Treino em vetores colunas .

## Passo2: Calcular a distancia euclidiana da imagem teste escolhida com as imagens do
## Conjunto Treino.

## Passo3: Pegar a imagem com menor distância e ler seu label.

f = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 60000

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

image = image.reshape((784,1))

distance = 0
index = 0

for i in range(0, 60000):
    alvo = np.asarray(data[i]).squeeze()
    alvo = alvo.reshape((784,1))

    aux = image - alvo

    if i == 0:
        distance = np.linalg.norm(aux)
    else:
        if np.linalg.norm(aux) < distance:
            distance = np.linalg.norm(aux)
            index = i

print()
print("Sua foto é o digito {}".format(labels[index]))


# Mostrando Imagem mais Próxima

print()
print("A foto mais semelhante é: ")
alvo = np.asarray(data[index]).squeeze()
plt.imshow(alvo)
plt.show()





