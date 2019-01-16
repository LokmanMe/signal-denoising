# -*- coding: utf-8 -*-
"""
EXERCICE 1
@author: Lokman Mechouek
"""

import numpy as np
import matplotlib.pyplot as plt

### 1 - Chargement des données ###

signal = np.loadtxt("signal.txt") #On charge le fichier signal.txt
print(signal[:10]) #On affiche une petite partie des données. 

N = signal.size
print("La variable signal est alors un vecteur de", N, "nombres compris entre ", min(signal), " et ", max(signal))
print("Le signal est donc déjà normalisé") 

plt.figure()
plt.title("Un signal de taille 512")
plt.plot(signal)
plt.show() # Enfin, on affiche ici l'allure du signal.


### 2 - Décomposition en ondelettes ###

import pywt

ond = "db4" # On va utiliser la décomposition en ondelettes de Daubechies appelée "db4".
print(pywt.Wavelet('db4')) # On affiche ici les caractéristiques de l'ondelette "db4" et on remarque que c'est une ondelette orthogonale.
n0=3 # On va utiliser le niveau de décomposition 3

coeffs_l = pywt.wavedec(signal, ond, level=n0) #On utilise wavedec (litteralement decomposition du signal) 
#On a décomposé le signal en ondelettes de Daubechies avec les propriétés ci-dessus. On a alors une liste de 4 vecteurs (n0+1).
#le premier (indice 0) correspond au coefficient d'approximation et les autres (1, 2, 3) correspondent aux coefficients en ondelettes.
coefficients = np.concatenate(coeffs_l) #On met ces vecteurs à la suite pour pouvoir les afficher.

plt.figure()
plt.title("Le signal compressé")
plt.plot(coefficients, color="black")
plt.show()
#On a alors ici notre signal compressé de façon parcimonieuse (il y a beaucoup de coefficients nuls)


### 3 - Calcul des coefficients dans la base de Fourier (epsilon') ###

signal_f = np.fft.fft(signal) # On charge ici la transformée de Fourier (Fast Fourier Transform)
print(signal_f[0:3]) #On remarque que celui-ci ne s'exprime pas dans les réels mais dans les complexes (exemple avec les 3 premiers éléments)

#On affiche alors seulement la partie réelle de la transformée de Fourier
plt.figure()
plt.plot(np.abs(signal_f))
plt.title("Transformée de Fourier")
plt.show() 

coeffs_f = np.fft.fft(signal)/np.sqrt(N) #Pour avoir les coefficients de Fourier, on divise alors par racine de N)

#On va ensuite comparer les valeurs des coefficients de Fourier avec les valeurs des coefficients en ondelettes :

# Histogramme des valeurs réelles des coefficients de Fourier
plt.hist(np.abs(coeffs_f), bins = [0, 0.05, 0.1, 0.15, 0.2])
plt.xlim(0, 1)
plt.title("Valeurs absolues des coefficients de Fourier")
plt.show()

# Histogramme des valeurs réelles des coefficients des ondelettes de Daubechies
plt.hist(np.abs(coefficients), bins = [0, 0.05, 0.1, 0.15, 0.2])
plt.xlim(0, 1)
plt.title("Valeurs absolues des coefficients en ondelettes db4")
plt.show()

# On va chercher à savoir dans quelle base le signal est le plus parcimonieux. Pour cela, j'ai calculé le nombre de coefficients proches de 0 dans la base de Fourier puis dans la base de Daubechies :

# FOURIER #
k_fourier=0
i=0
for i in range(len(np.abs(coeffs_f))):
    if np.abs(coeffs_f)[i] <= 0.05 and np.abs(coeffs_f)[i] >= -0.05:
        k_fourier = k_fourier + 1
    else:
        pass
    i = i+1   
print(i, k_fourier)

# DAUBECHIES #
k_daubechies=0
i=0
for i in range(len(np.abs(coefficients))):
    if np.abs(coefficients)[i]<=0.05 and np.abs(coefficients)[i]>=-0.05:
        k_daubechies = k_daubechies+1
    else:
        pass
    i=i+1   
print(i, k_daubechies)

print("Le nombre de coefficients proches de 0 dans la base de Fourier est de : ", k_fourier)
print("Le nombre de coefficients proches de 0 dans la base de Daubechies est de : ", k_daubechies)
print("Le signal est donc plus parcimonieux dans la base de Daubechies !")

#On peut essayer de voir pour quel niveau la décomposition en ondelettes est la meilleure :

j=1
n = [] ; c0 = []
while j <= 6:
    ond = "db4"
    n0=j
    coeffs_l=pywt.wavedec(signal, ond, level=n0) 
    coefficients = np.concatenate(coeffs_l)
    k_daubechies=0
    i=0
    for i in range(len(np.abs(coefficients))):
        if np.abs(coefficients)[i]<=0.25 and np.abs(coefficients)[i]>=-0.25:
            k_daubechies = k_daubechies+1
        else:
            pass
        i=i+1   
    n.append(k_daubechies); c0.append(j)
    j=j+1
    print("Niveau de décomposition : ", n0)
    print("Nombre de coefficients nul : ", k_daubechies)

plt.plot(c0, n)
plt.ylabel('Nombre de coefficients nuls')
plt.xlabel('Niveau de décomposition')
plt.show()  # On remarque que plus le niveau de décomposition est élevé, plus le signal est parcimonieux dans la base d'ondelettes en question. 