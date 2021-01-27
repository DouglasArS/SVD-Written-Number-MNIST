#!/usr/bin/env python
# coding: utf-8

# In[4]:


import gzip
import numpy as np
import matplotlib.pyplot as plt


# # Acessando os Labels do Conjunto Treino:

# In[18]:


file_train_labels = gzip.open('train-labels-idx1-ubyte.gz','r')

file_train_labels.read(8)

buf = file_train_labels.read(60000)

data = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)

train_labels = data


# In[6]:


# Pritando os labels do Conjunto Treino
train_labels


# In[ ]:





# In[ ]:





# # Acessando as Imagens do Conjunto Treino

# In[105]:


file_train_images = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 3000

file_train_images.read(16)

buf = file_train_images.read(image_size * image_size * num_images)

data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

train_images = data

for i in range(0, 100):
    plt.imshow(np.asarray(data[i]).squeeze())
    plt.show()
    print("Foto {}".format(i+1))
    print()


# In[ ]:





# In[ ]:





# # Tratando as imagens do Conjunto Treino:

# In[106]:


# Criando matriz de dados Não-Centralizada

x = np.asarray(train_images).squeeze().reshape(num_images, 784)
x.shape


# In[107]:


# Calculando media e centralizando dados
train_mean = np.mean(x, axis = 1)

X = (x.T - train_mean).T
X.shape


# In[ ]:





# In[108]:


# Calculando Matriz de Cov
covx = np.cov(X)


# In[109]:


covx.shape


# In[ ]:





# In[110]:


# Decomposição Espectral da CovX
V, Q = np.linalg.eig(covx)


# In[111]:


V.shape


# In[112]:


Q.shape


# In[113]:


# Calculando a matriz P
P = np.dot(X.T,Q)


# In[ ]:





# In[162]:


# Estabelecendo numero de autovalores
autovalor_num = 780


# In[ ]:





# In[163]:


# Criando a Matriz Final


# In[164]:


X = np.dot(Q[:,:autovalor_num], P.T[:autovalor_num,:])


# In[165]:


X.shape


# In[ ]:





# In[166]:


# Retonando a media
X = (X.T + train_mean).T


# In[167]:


X.shape


# In[ ]:





# In[168]:


# Voltando ao formato imagem


# In[169]:


image = X[1]


# In[170]:


image.shape


# In[171]:


image = image.reshape((28, 28)).astype(np.float32)


# In[172]:


image.shape


# In[173]:


image = np.asarray(image)
plt.imshow(image)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




