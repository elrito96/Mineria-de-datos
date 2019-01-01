#!/usr/bin/env python
# coding: utf-8

# ## Ejercicio Ultramétrico

# In[2]:


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0.2,0.2,0.35,0.35],[0.2,0,0.15,0.35,0.35],[0.2,0.15,0,0.35,0.35],[0.35,0.35,0.35,0,0.1],[0.35,0.35,0.35,0.1,0]])
Xo = X

print(Xo)
minValue = min(Xo[np.nonzero(Xo)])
print("\nLa distancia minima es : ",minValue)
print("Se unen 4 y 5\n")

Xo = np.delete(Xo,(4),axis=0)
Xo = np.delete(Xo,(4),axis=1)
print(Xo)

minValue = min(Xo[np.nonzero(Xo)])
print("\nLa distancia minima es : ",minValue)
print("Se unen 2 y 3\n")

Xo = np.delete(Xo,(2),axis=0)
Xo = np.delete(Xo,(2),axis=1)
print(Xo)

minValue = min(Xo[np.nonzero(Xo)])
print("\nLa distancia minima es : ",minValue)
print("Se unen 1 y 23\n")

Xo = np.delete(Xo,(0),axis=0)
Xo = np.delete(Xo,(0),axis=1)
print(Xo)

minValue = min(Xo[np.nonzero(Xo)])
print("\nLa distancia minima es : ",minValue)
print("Se unen 123 y 45\n")

Xo = np.delete(Xo,(0),axis=0)
Xo = np.delete(Xo,(0),axis=1)
print(Xo)

print("El dendograma que representa este proceso es el siguiente: ")

Xden = np.triu(X)

Z = hierarchy.linkage(Xden,'single')
plt.figure()
dn = hierarchy.dendrogram(Z)


# ## Ejercicio algoritmo AC minimo

# In[3]:


X = np.array([[0,1,3,4,7],[0,0,4,4,8],[0,0,0,2,8],[0,0,0,0,7],[0,0,0,0,0]])

Xden = np.triu(X)

Z = hierarchy.linkage(Xden,'single')
plt.figure()
dn = hierarchy.dendrogram(Z)
print("El proceso de aglormeración de clusters es: \n Se unen 1 y 2\n Se unen 3 y 4\n Se unen 12 y 34\n Se unen 5 y 1234")


# ## Ejercicio algoritmo AC maximo

# In[4]:


X = np.array([[0,1,3,4,7],[0,0,4,4,8],[0,0,0,2,8],[0,0,0,0,7],[0,0,0,0,0]])

Xden = np.triu(X)

Z = hierarchy.linkage(Xden,'complete')
plt.figure()
dn = hierarchy.dendrogram(Z)
print("El proceso de aglormeración de clusters es: \n Se unen 1 y 2\n Se unen 3 y 4\n Se unen 12 y 34\n Se unen 5 y 1234")
print("El resultado es el mismo que en el anterior, excepto que cambia la indexación alfa (eje y)")


# ## Ejercicio algoritmo Ward

# In[48]:


from scipy.spatial import distance

X = np.array([[1,2],[2,1],[2,2.7],[5,3],[6.5,2],[7,3]])
Ncluster = 3

# meter un cluster, devolver su heterogeneidad
def heterogeneidad( arg1 ):
    # en el caso 1, le llega como argumento [1,2]
    # calcular centroide del cluster, en este caso sabemos que cada muestra tiene 2 elementos.
    sumx = 0
    sumy = 0
    nelementos = arg1.shape[0]
    for i in range (0, nelementos):
        sumx = sumx + arg1[i][0]
        sumy = sumy + arg1[i][1]
    centroideX = sumx/nelementos
    centroideY = sumy/nelementos
    centroide = np.array([centroideX,centroideY])
    # calcular dsitancias de cada elemento al centroide
    heter = 0
    for i in range (0, nelementos):
        # ya que sabemos que cada instancia tiene 2 elementos
        for j in range (0, 2):
            heter = heter + (arg1[i][j]-centroide[j])**2
    return heter

# Principalmente hay 6 puntos, es decir 6 clusters, cada uno con un solo elemento por lo que la heterogeneidad de cada cluster debería ser 0


# Creamos la matriz de heterogeneidad, que contendrá la heterogeneidad de la partición dependiendo de la union de clusters que se haya hecho, por ejemplo la posicion 0 1 sería
# la heterogeneidad de la partición si se han unido los clusters 1 y 2. Será una matriz triangular superior.

s=(6,6)
mH = np.zeros(s)

#puntos iniciales
p1=np.array([1,2])
p2=np.array([2,1])
p3=np.array([2,2.7])
p4=np.array([5,3])
p5=np.array([6.5,2])
p6=np.array([7,3])
# Se unen 1 y 2 
c1=np.array([p1,p2])
c2=np.array([p3])
c3=np.array([p4])
c4=np.array([p5])
c5=np.array([p6])
mH[0][1]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 1 y 3
c1=np.array([p1,p3])
c2=np.array([p2])
c3=np.array([p4])
c4=np.array([p5])
c5=np.array([p6])
mH[0][2]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 1 y 4
c1=np.array([p1,p4])
c2=np.array([p3])
c3=np.array([p2])
c4=np.array([p5])
c5=np.array([p6])
mH[0][3]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 1 y 5
c1=np.array([p1,p5])
c2=np.array([p3])
c3=np.array([p4])
c4=np.array([p2])
c5=np.array([p6])
mH[0][4]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 1 y 6
c1=np.array([p1,p6])
c2=np.array([p3])
c3=np.array([p4])
c4=np.array([p5])
c5=np.array([p2])
mH[0][5]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 2 y 3
c1=np.array([p1])
c2=np.array([p2,p3])
c3=np.array([p4])
c4=np.array([p5])
c5=np.array([p6])
mH[1][2]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 2 y 4
c1=np.array([p1])
c2=np.array([p2,p4])
c3=np.array([p3])
c4=np.array([p5])
c5=np.array([p6])
mH[1][3]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 2 y 5
c1=np.array([p1])
c2=np.array([p2,p5])
c3=np.array([p4])
c4=np.array([p3])
c5=np.array([p6])
mH[1][4]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 2 y 6
c1=np.array([p1])
c2=np.array([p2,p6])
c3=np.array([p4])
c4=np.array([p5])
c5=np.array([p3])
mH[1][5]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 3 y 4
c1=np.array([p1])
c2=np.array([p2])
c3=np.array([p3,p4])
c4=np.array([p5])
c5=np.array([p6])
mH[2][3]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 3 y 5
c1=np.array([p1])
c2=np.array([p2])
c3=np.array([p3,p5])
c4=np.array([p4])
c5=np.array([p6])
mH[2][4]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 3 y 6
c1=np.array([p1])
c2=np.array([p2])
c3=np.array([p3,p6])
c4=np.array([p5])
c5=np.array([p3])
mH[2][5]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 4 y 5
c1=np.array([p1])
c2=np.array([p2])
c3=np.array([p3])
c4=np.array([p4,p5])
c5=np.array([p6])
mH[3][4]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 4 y 6
c1=np.array([p1])
c2=np.array([p2])
c3=np.array([p3])
c4=np.array([p4,p6])
c5=np.array([p5])
mH[3][5]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
# Se unen 5 y 6
c1=np.array([p1])
c2=np.array([p2])
c3=np.array([p3])
c4=np.array([p4])
c5=np.array([p5,p6])
mH[4][5]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)+heterogeneidad(c5)
print("Primera iteración\n")
print(mH)
minValue = min(mH[np.nonzero(mH)])

print("\nLa partición que genera menor heterogeneidad, que sera ",minValue, ", es la resultante de combinar los cluster 5 y 6")

# Segunda iteración despues de combinar 5 y 6

print("\nSegunda iteración")

s=(5,5)
mH = np.zeros(s)

# A partir de ahora llamaré cluster 4 a la combinación de clusters 5 y 6

# Se unen 1 y 2 
c1=np.array([p1,p2])
c2=np.array([p3])
c3=np.array([p4])
c4=np.array([p5,p6])
mH[0][1]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)
# Se unen 1 y 3
c1=np.array([p1,p3])
c2=np.array([p2])
c3=np.array([p4])
c4=np.array([p5,p6])
mH[0][2]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)
# Se unen 1 y 4
c1=np.array([p1,p4])
c2=np.array([p3])
c3=np.array([p2])
c4=np.array([p5,p6])
mH[0][3]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)
# Se unen 1 y 5
c1=np.array([p1,p5,p6])
c2=np.array([p3])
c3=np.array([p4])
c4=np.array([p2])
mH[0][4]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)
# Se unen 2 y 3
c1=np.array([p1])
c2=np.array([p2,p3])
c3=np.array([p4])
c4=np.array([p5,p6])
mH[1][2]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)
# Se unen 2 y 4
c1=np.array([p1])
c2=np.array([p2,p4])
c3=np.array([p3])
c4=np.array([p5,p6])
mH[1][3]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)
# Se unen 2 y 5
c1=np.array([p1])
c2=np.array([p3])
c3=np.array([p4])
c4=np.array([p2,p5,p6])
mH[1][4]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)
# Se unen 3 y 4
c1=np.array([p1])
c2=np.array([p2])
c3=np.array([p3,p4])
c4=np.array([p5,p6])
mH[2][3]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)
# Se unen 3 y 5
c1=np.array([p1])
c2=np.array([p2])
c3=np.array([p4])
c4=np.array([p3,p5,p6])
mH[2][4]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)
# Se unen 4 y 5
c1=np.array([p1])
c2=np.array([p3])
c3=np.array([p2])
c4=np.array([p4,p5,p6])
mH[3][4]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)+heterogeneidad(c4)

print(mH)
minValue = min(mH[np.nonzero(mH)])

print("\nLa partición que genera menor heterogeneidad, que sera ",minValue, ", es la resultante de combinar los cluster 1 y 3")

# Tercera iteración despues de combinar 1 y 3

print("\nTercera iteración")

s=(4,4)
mH = np.zeros(s)

# A partir de ahora llamaré cluster 1 a la combinación de clusters 1 y 3, y cluster 4 a la combinacción de 5 y 6

# Se unen 1 y 2
c1=np.array([p1,p3,p2])
c2=np.array([p4])
c3=np.array([p5,p6])
mH[0][1]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)
# Se unen 1 y 3
c1=np.array([p1,p3,p4])
c2=np.array([p2])
c3=np.array([p5,p6])
mH[0][2]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)
# Se unen 1 y 4
c1=np.array([p1,p3,p5,p6])
c2=np.array([p2])
c3=np.array([p4])
mH[0][3]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)
# Se unen 2 y 3
c1=np.array([p1,p3])
c2=np.array([p2,p4])
c3=np.array([p5,p6])
mH[1][2]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)
# Se unen 2 y 4
c1=np.array([p1,p3])
c2=np.array([p2,p5,p6])
c3=np.array([p4])
mH[1][3]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)
# Se unen 3 y 4
c1=np.array([p1,p3])
c2=np.array([p2])
c3=np.array([p4,p5,p6])
mH[2][3]=heterogeneidad(c1)+heterogeneidad(c2)+heterogeneidad(c3)

print(mH)
minValue = min(mH[np.nonzero(mH)])

print("\nLa partición que genera menor heterogeneidad, que sera ",minValue, ", es la resultante de combinar los cluster 1-3 y 2")

print("\nLa progresión del aglomeramiento de clusters ha sido:\n0:{1},{2},{3},{4},{5},{6}\n1:{1},{2},{3},{4},{5,6}\n2:{1,3},{2},{4},{5,6}\ny finalmente quedan los 3 clusters {1,2,3},{4},{5,6}")


# ## Ejercicio DB-Cluster

# In[55]:


D = np.array([[0,1,3,4,7],[0,0,4,4,8],[0,0,0,2,8],[0,0,0,0,7],[0,0,0,0,0]])
m=5
sumDist=0
for i in range(0,4):
    for j in range(0,4):
        sumDist=sumDist+D[i][j]**2
varInicial = (1/(2*m**2))*sumDist
print(varInicial)

# Para proseguir habría que hacer todas las posibles combinaciones de clusters con estas 5 muestras y nos quedarimos
# con la que la suma de las variabilidades geométricas de sus conjuntos sea la menor


# In[ ]:




