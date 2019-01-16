import numpy as np
import matplotlib.pyplot as plt
import pywt

K0 = 256
x = np.arange(-K0, K0-1, 1)

plt.figure()
plt.plot(x)
plt.show()

xT = len(x) * [0]
for i in range(0, len(x)):  
    xT[i] = pywt.threshold(x[i], 100,'soft')
xT = np.array(xT)    

plt.figure()
plt.plot(x, label="x")
plt.plot(xT, color="red", label="xT")
plt.xlabel("k")
plt.legend()
plt.show()

#####################################################################
# On voudrait que pour tout k :                                     #
#                                                                   #
#                                                                   #
# xT(k) = 0                       si      |x(k)| <= T               #
# xT(k) = x(k)-T * sgn(x(k))      si      |x(k)| >  T               #
#####################################################################

print("On a pris un seuil = 100")

print("Ici |x(250)| = ", np.abs(x[250]), " <= 100 donc xT(100) devrait être = 0")
print("On a xT(100) = ", xT[250])
print("A priori c'est bon pour la première condition, de plus en regardant la courbe on voit que c'est bon")

print("Ici x(100) = ", np.abs(x[100]), " > 100 donc xT(100) devrait être = x(100) - 100 sgn(x(100)) = ", x[100] - (100 * np.sign(x[100])) )
print("On a xT(100) = ", xT[100])
print("La fonction thresholding avec le paramètre 'soft' est donc bien ce que l'on veut")

#####################################################################

y = 20 * [0]
k0 = 1
k1 = 8
a0 = 6
a1 = 5

y[k0] = k0 * a0
y[k1] = k1 * a1
y = np.array(y)

plt.figure()
plt.plot(y)
plt.show()

# y est donc notre signal contenant nos 2 Diracs

# On va bruit le signal en utilisant la fonction de l'exercice 2

def addwhitenoise (x, sigma):   
    bruit = np.random.normal(0, sigma, x.size) 
    return x + bruit

y_noised = addwhitenoise(y, max(np.abs(a0), np.abs(a1))/20)

plt.figure()
plt.plot(y_noised)
plt.show()

# J'applique la fonction de seuillage doux : 

yT = len(y) * [0]
for i in range(0, len(y)):  
    yT[i] = pywt.threshold(y[i], 0,'soft')
xT = np.array(yT) 

plt.figure()
plt.plot(yT)
plt.show()

print("Le signal contenant les 2 Diracs est parfaitement débruité")