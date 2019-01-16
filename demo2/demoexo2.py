# -*- coding: utf-8 -*-
"""
EXERCICE 2
@author: Lokman Mechouek
"""

import numpy as np
import matplotlib.pyplot as plt


def addwhitenoise (x, sigma):   #On définit notre fonction addwhitenoise qui prend les paramètres : x (un signal) et sigma (l'écart-type)
    bruit = np.random.normal(0, sigma, x.size)  #On définit le bruit comme un vecteur aléatoire de la même taille que le signal x, suivant une loi normale centrée sur 0 et pour variance sigma
    return x + bruit    #La fonction retourne un signal composé du signal x et du bruit

bruit_test = np.random.normal(0, 0.2, 500)
plt.figure()
plt.title("Un bruit blanc gaussien centré en 0, de variance 0.04")
plt.plot(bruit_test)
plt.show()

# On charge le fichier signal.txt du tp (j'ai changé de nom pour pas modifier par erreur celui de l'exercice 1)
signal_512 = np.loadtxt("signal 512.txt")

# On crée un signal constant par morceau
s = 512*[0]
for i in range(len(s)):
    if i <= 50:
        s[i] = 1
    elif i <= 250:
        s[i] = np.sin(0.1*i)
    elif i <= 400:
        s[i] = 0
    else:
        s[i] = np.sin(0.05*i)
signal_CPM = np.array(s) #On convertit la liste en vecteur sinon on a des soucis avec la fonction

        
#On plot nos signaux :

plt.figure()
plt.title("signal.txt")
plt.plot(signal_512)
plt.show()

plt.figure()
plt.title("Signal constant par morceaux")
plt.plot(signal_CPM)
plt.show()


#On ajoute du bruit à nos signaux :

s512_noised = addwhitenoise(signal_512, 0.4)
CPM_noised = addwhitenoise(signal_CPM, 0.2) 

#On plot nos signaux bruités :

plt.figure()
plt.title("signal.txt avec bruit gaussien ajouté")
plt.plot(s512_noised)
plt.plot(signal_512, color='red')
plt.show()

plt.figure()
plt.title("Signal constant par morceaux avec bruit gaussien ajouté")
plt.plot(CPM_noised)
plt.plot(signal_CPM, color='red')
plt.show()


# On va réaliser une décomposition en ondelette de Daubechies sur nos signaux clairs

import pywt

coeffs = pywt.wavedec(signal_512, "db4", level=3) 
coeff_512 = np.concatenate(coeffs)

coeffs = pywt.wavedec(signal_CPM, "db4", level=3) 
coeff_CPM = np.concatenate(coeffs)

# On plot les coefficients en ondelettes de Daubechies de nos signaux

plt.figure()
plt.title("Coefficients en ondelettes du signal.txt original")
plt.plot(coeff_512)
plt.show()

plt.figure()
plt.title("Coefficients en ondelettes du signal CPM original")
plt.plot(coeff_CPM)
plt.show()


# On va réaliser une décomposition en ondelette de Daubechies sur nos signaux bruités

coeffs = pywt.wavedec(s512_noised, "db4", level=3) 
coeff_s512_noised = np.concatenate(coeffs)

coeffs = pywt.wavedec(CPM_noised, "db4", level=3) 
coeff_CPM_noised = np.concatenate(coeffs)

# On plot les décompositions en ondelettes de nos signaux bruités

plt.figure()
plt.title("Coefficients en ondelettes de Daubechies du signal.txt bruité")
plt.plot(coeff_s512_noised)
plt.show()

plt.figure()
plt.title("Coefficients en ondelettes de Daubechies du signal CPM bruité")
plt.plot(coeff_CPM_noised)
plt.show()


#On va compter le nombre de coefficients nuls pour le signal 512 bruité :

k_daubechies=0
i=0
for i in range(len(np.abs(coeff_s512_noised))):
    if np.abs(coeff_s512_noised)[i]<=0.05 and np.abs(coeff_s512_noised)[i]>=-0.05:
        k_daubechies = k_daubechies+1
    else:
        pass
    i=i+1   
print("Le nombre de coefficients proches de 0 dans la base de Daubechies du signal.txt bruité est de : ", k_daubechies)
#On remarque que les signaux bruités ne sont pas parcimonieux dans la base d'ondelettes de Daubechies.

#J'aimerais tester une autre base d'ondelettes, celle de Haar :

coeffs_haar = pywt.wavedec(s512_noised, "haar", level=3) 
coeff_s512_noised_haar = np.concatenate(coeffs)
plt.figure()
plt.title("Coefficients en ondelettes de Haar du signal.txt bruité")
plt.plot(coeff_s512_noised_haar)
plt.show()

k_haar=0
i=0
for i in range(len(np.abs(coeff_s512_noised_haar))):
    if np.abs(coeff_s512_noised_haar)[i]<=0.05 and np.abs(coeff_s512_noised_haar)[i]>=-0.05:
        k_haar = k_haar+1
    else:
        pass
    i=i+1   
print("Le nombre de coefficients proches de 0 dans la base de Haar est de : ", k_haar)
#Les signaux bruités sont plus parcimonieux dans la base d'ondelettes de Haar, mais il y a tout de même beaucoup trop de coefficients différents de zéro.


# On plot les approximations de nos signaux

plt.figure()
plt.title("Coefficients en ondelettes du signal.txt")
plt.plot(coeff_512[:70], color="blue", label = "Approximation du signal.txt original")
plt.plot(coeff_s512_noised[:70], color="red", label = "Approximation du signal.txt bruité")
plt.legend()
plt.show()

plt.figure()
plt.title("Coefficients en ondelettes du signal CPM")
plt.plot(coeff_CPM[:70], color="blue", label = "Approximation du signal CPM original")
plt.plot(coeff_CPM_noised[:70], color="red", label = "Approximation du signal CPM bruité")
plt.legend()
plt.show()

