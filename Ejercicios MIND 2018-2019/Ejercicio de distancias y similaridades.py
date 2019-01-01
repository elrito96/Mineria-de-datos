#!/usr/bin/env python
# coding: utf-8

# ## Ejercicio con atributos continuos reales

# In[96]:


from sklearn.cluster import AgglomerativeClustering
import numpy as np
from numpy import inf
from numpy.linalg import inv

X = np.array([[2.1],[3.1],[3.4],[1.9]])

Y = np.array([[1.2],[2.0],[1.7],[3.6]])

# Distancia euclidea
sum = 0
for i in range(0,X.shape[0]):
    loc = (X[i]-Y[i])**2
    sum = sum + loc
D = sum**(1/2)

print("La distancia euclídea es ",D[0])

# Distancia minkowsky con q = 1 y q = 3

q = 1

sum = 0
for i in range(0,X.shape[0]):
    loc = (X[i]-Y[i])**q
    sum = sum + loc
D = sum**(1/q)

print("La distancia Minkowsky con q = 1 es ",D[0])

q = 3

sum = 0
for i in range(0,X.shape[0]):
    loc = (X[i]-Y[i])**q
    sum = sum + loc
D = sum**(1/q)

print("La distancia Minkowsky con q = 3 es ",D[0])

# Distancia mahalanobis con matriz varianza - covarianza M

M = np.array([[2,0,0,0],[0,4,0,0],[0,0,4,0],[0,0,0,2]])
Minver = inv(M)
Mdifer = X-Y
Mul1 = np.matmul(Mdifer.T, Minver)
Mul2 = np.matmul(Mul1,Mdifer)
D = (Mul2)**(1/2)

print("La distancia Mahalanobis es ",D[0][0])


# ## Ejercicio de similaridades

# In[97]:


#Tabla de coincidencias

Xt = np.array([[1,0,0,0,1,1,0,1,0,0]])
X = Xt.T
Yt = np.array([[0,0,1,0,1,1,1,1,0,1]])
Y = Yt.T
a = 0
b = 0
c = 0
d = 0
for i in range(0,X.shape[0]):
    if(X[i][0]==0 and Y[i][0]==0):
        a+=1
    elif(X[i][0]==1 and Y[i][0]==0):
        c+=1
    elif(X[i][0]==0 and Y[i][0]==1):
        b+=1
    elif(X[i][0]==1 and Y[i][0]==1):
        d+=1
print("Tabla de coincidencias :")
print("x\y    0   1\n")
print("0     ",a," ",b)
print("1     ",c," ",d)

# similaridades de Sokal - Michel y Jaccard
sSM=(a+d)/X.shape[0]
print("\nLa similaridad de Sokal-Michel es : ",sSM)
sJ=d/(d+b+c)
print("\nLa similaridad de Jaccard : ",sJ)

# Distancia a través de la transformación de Gower

D1=(1-sSM)**(1/2)
print("\nLa distancia a través de Gower con similaridad Sokal-Michel es : ",D1)
D2=(1-sJ)**(1/2)
print("\nLa distancia a través de Gower con similaridad Jaccard es : ",D2)
print("\nLa distancia de Hamming es : ",c+b)


# ## Similaridad entre casos 2 y 7

# In[86]:


# siendo x1 y x2 continuos, x3 y x4 bits, y x5 y x6 cualitativas

C2 = np.array([[50.2,2.9,0,1,1,1]])
C7 = np.array([[52.3,3.7,1,1,1,2]])

n1=2
n2=2
n3=2

R1=54.1-49.8
R2=4.6-2.6
a=0
d=1
alfa=1

x1=C2[0][0]
x2=C2[0][1]
y1=C7[0][0]
y2=C7[0][1]

similaridad = ((1-(abs(x1-y1)/R1))+(1-(abs(x2-y2)/R2))+d+alfa)/(n1+(n2-a)+n3)

print("La similaridad entre los casos 2 y 7 es de : ", similaridad)

