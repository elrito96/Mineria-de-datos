#!/usr/bin/env python
# coding: utf-8

# ## Experimento con 3 clústers.

# In[55]:


import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from statistics import mode
from scipy import stats

# El numero de clusters que vamos a usar se alamcena en esta variable NC
NC=3

# Parte del código para obtener el conjunto de datos, división en entrenamiento/prueba y normalización
iris = datasets.load_iris()
Y = iris.target
Y, Y.shape, iris.data.shape
normalizar = MinMaxScaler()
iris_norm = normalizar.fit(iris.data)
X = normalizar.transform(iris.data)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Creación del modelo
kmedias = KMeans(n_clusters=NC).fit(X_train)

# Calculo de las etiquetas de los clusters
clust0=y_train[np.argwhere(kmedias.labels_==0)]
clust1=y_train[np.argwhere(kmedias.labels_==1)]
clust2=y_train[np.argwhere(kmedias.labels_==2)]
m0 = stats.mode(clust0)
m1 = stats.mode(clust1)
m2 = stats.mode(clust2)
clustM=[m0[0][0][0],m1[0][0][0],m2[0][0][0]]
print("Con k = 3\n")
for j in range(0,NC):
    print("Moda del cluster",j," es :",clustM[j])

test_means=kmedias.predict(X_test)

# Inicializamos la matriz de confusión
M=[[0,0,0],[0,0,0],[0,0,0]]

# Rellenamos la matriz de confusión
for j in range(0, y_test.shape[0]):
    vPred = clustM[test_means[j]]
    vReal = y_test[j]
    M[vPred][vReal]+=1
print("\nClase predicha \ Clase real\n")

# Imprimimos la matriz de confusión
for j in range(0,NC):
    for k in range(0,NC):
        print(M[j][k], end =" ")
    print()

# Obtenemos la tasa de error
tacierto=(M[0][0]+M[1][1]+M[2][2])/50
print("\nLa tasa de acierto es de = ",tacierto)


# ## Código del ejercicio completo, probando entre 3 y 10 clústers y obtendremos la mayor tasa de acierto.

# In[54]:


import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from statistics import mode
from scipy import stats

#El numero de clusters que probaremos varia entre 3 y 10
NCini = 3
NCfin = 10

# Parte del código para obtener el conjunto de datos, división en entrenamiento/prueba y normalización
iris = datasets.load_iris()
Y = iris.target
Y, Y.shape, iris.data.shape
normalizar = MinMaxScaler()
iris_norm = normalizar.fit(iris.data)
X = normalizar.transform(iris.data)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#Vector que almacenará las tasas de acierto con cada k para calcular el mejor resultado al final
tasasAcierto=np.empty([NCfin-NCini+1])

for i in range(NCini,NCfin+1):
    print("\n--------------------------------------------------")
    print("\nExperimento con ",i," clusters")
    
    # Creación del modelo
    kmedias = KMeans(n_clusters=i).fit(X_train)

    # Almacenaremos las modas de cada cluster en m
    m = np.zeros(i, dtype=int)
    
    # Calculo de las modas de cada clúster
    for j in range(0,i):
        cluster=y_train[np.argwhere(kmedias.labels_==j)]
        m[j]=stats.mode(cluster)[0][0][0]
    
    # Probamos el conunto de prueba con el modelo
    test_means=kmedias.predict(X_test)

    # Inicializamos matriz de confusion
    M=[[0,0,0],[0,0,0],[0,0,0]]

    # Rellenamos la matriz de confusión
    for j in range(0, y_test.shape[0]):
        vPred = m[test_means[j]]
        vReal = y_test[j]
        M[vPred][vReal]+=1
    
    # Imprimimos la matriz de confusión
    print("\nClase predicha(filas) \ Clase real(columnas) [0|1|2]\n")
    for j in range(0,NC):
        for k in range(0,NC):
            print(M[j][k], end =" ")
        print()
    
    # Cálculo de las tasas de error para cada k y la minima global
    tacierto=(M[0][0]+M[1][1]+M[2][2])/50
    print("\nLa tasa de acierto es de = ",tacierto)
    tasasAcierto[i-NCini]=tacierto
print("\nLa mayor tasa de acierto se da con k-clusters =",np.argmax(tasasAcierto)+3,"y es ",np.amax(tasasAcierto))

# Como conclusión de este ejercicio, los resultados no son nada definitivos. Al repetirlo varias veces, en al menos alguna ocasión
# se ha obtenido una tasa de error de 0 con cualquier k. Para comparar los k de manera mas fiable sería recomendable repetir los experimentos
# un número elevado de veces (por ejemplo 100) y calcular la tasa de error promedio para poder hacer una comparación más fiable.


# In[ ]:




