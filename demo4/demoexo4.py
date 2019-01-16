# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 03:41:30 2018

@author: lok
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt


def addwhitenoise (x, sigma):
    bruit = np.random.normal(0, sigma, x.size) 
    return x + bruit   


#On va reprendre le signal que l'on va bruiter
s512 = np.loadtxt("signal.txt")
s512_noised = addwhitenoise(s512, 0.1)

plt.figure()
plt.title("signal.txt avec bruit gaussien de variance 0.01 ajouté")
plt.plot(s512_noised)
plt.show()

#DECOMPOSITION EN ONDELETTES DE DAUBECHIES
ond = "db4" 
n0=3 

coeffs_s512 = pywt.wavedec(s512_noised, ond, level=n0) # On recupere les coefficients d'ondelettes de Daubechies

plt.figure()
plt.title("coeffs")
plt.plot(np.concatenate(coeffs_s512))
plt.show()

for i in range(1, len(coeffs_s512)): # Pour chaque coefficients d'ondelettes (et uniquement ceux de l'ondelettes) :       
    coeffs_s512[i] = pywt.threshold(coeffs_s512[i], 0.2,'soft') # On seuille les coefficients en utilisant la fonction pywavelets.threshold et en lui fournissant les paramètres : coeffs du signal bruité, seuil, méthode de seuillage
    s512_debruite = pywt.waverec(coeffs_s512, ond) # On recompose le signal, il correspond alors au signal débruité 

plt.figure()
plt.title("Recomposition à partir de la base d'ondelettes")
plt.plot(s512_debruite, label = "signal recomposé", color = "blue")
plt.plot(s512, label="signal original", color = "orange")
plt.legend()
plt.show()

#Maintenant que l'on sait comment marche la fonction pywt.thresholding, on va créer la fonction :
def soft_thresholding(x, T):    
    for j in range(1,len(x)):        
        x[j] = pywt.threshold(x[j], T,'soft') 
        signal_seuilled = pywt.waverec(x, ond) 
    return signal_seuilled


#Le signal.txt a bien été recomposé grâce au seuillage doux. 
#Essayons maintenant la fonction sur notre signal constant par morceaux bruité:

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
CPM_noised = addwhitenoise(signal_CPM, 0.2) 

plt.figure()
plt.title("Signal constant par morceaux avec bruit gaussien ajouté")
plt.plot(CPM_noised)
plt.plot(signal_CPM, color='red')
plt.show()

#On décompose en ondelettes (on teste les ondelettes de Haar cette fois-ci)
ond = "haar" 
n0=3 
coeffs_CPM = pywt.wavedec(CPM_noised, ond, level=n0) # On recupere les coefficients d'ondelettes de Harr

#On applique maintenant notre seuillage doux :
CPM_seuil = soft_thresholding(coeffs_CPM , 0.8)

plt.figure()
plt.title("coeffs seuillés")
plt.plot(CPM_seuil)
plt.plot(signal_CPM, color="r")
plt.show()

#Maintenant on va plotter et calculer l'erreur entre le signal.txt débruité et le signal.txt original

erreur = len(s512_debruite)*[0]
for i in range(len(s512_debruite)):
    erreur[i] = (s512[i] - s512_debruite[i])**2
    
plt.figure()
plt.title("Courbe d'erreur de débruitage par ondelettes de Daubechies")
plt.plot(erreur, color="r")
plt.show()

plt.figure()
plt.title("erreur")
plt.plot(s512)
plt.plot(erreur, color="r")
plt.show()
#On s'aperçoit que les pics d'erreurs correspondent aux variations brutales dans le signal (là où la dérivée n'est pas calculable)

erreur_quadratique = np.sum(erreur)
print(erreur_quadratique)

#Calculons l'erreur pour le signal constant par morceaux, on avait utilisé la décomposition de Haar
erreur = len(signal_CPM)*[0]
for i in range(len(signal_CPM)):
    erreur[i] = (signal_CPM[i] - CPM_seuil[i])**2
    
plt.figure()
plt.title("Courbe d'erreur de débruitage par ondelettes de Daubechies")
plt.plot(erreur, color="r")
plt.show()

plt.figure()
plt.title("erreur")
plt.plot(signal_CPM)
plt.plot(erreur, color="r")
plt.show()
#En utilisant la décomposition de Haar pour débruiter notre signal, l'erreur est beaucoup plus élevée

erreur_quadratique = np.sum(erreur)
print(erreur_quadratique)

# !!!
# !!! J'avais utilisé l'erreur quadratique avant d'avoir vu qu'il existait le SNR. Je calcule alors le SNR ci-dessous 
# !!! 

s = 10 * np.log10 (np.mean(s512**2) / np.mean ( ( s512 * s512_debruite )**2 ) )
print("Le SNR est égal à : ", s)


# Maintenant je vais créer une fonction qui va : prendre un signal bruité, le décomposer, le seuiller, le recomposer puis calculer le SNR du signal recomposé


def rec_signal(x, sigma, ondelette, niveau):   #on donne à la fonction les paramètres : signal original à bruiter/débruiter, ondelettes à utiliser, niveau de décomposition
    
    x_noised = addwhitenoise(x, sigma)    #bruitage

    coeffs_x = pywt.wavedec(x_noised, ondelette, level=niveau)    # décompose
    
    for i in range(1, len(coeffs_x)):        # seuillage doux     
        coeffs_x[i] = pywt.threshold(coeffs_x[i], 0.2,'soft') 
        x_denoised = pywt.waverec(coeffs_x, ondelette)
        
    s = 10 * np.log10 (np.mean(x**2) / np.mean ( ( x * x_denoised )**2 ) )      #SNR
    print(s)
    
    return (x_denoised, s)



i=0
SNRs = 15 * [0]
sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]


for i in range(len(SNRs)):
    test, SNR = rec_signal(s512, sigmas[i], "db4", 3)
    SNRs[i] = SNR

SNRs = np.array(SNRs)
sigmas = np.array(sigmas)


plt.figure()
plt.title("SNR en fonction du niveau de bruit")
plt.plot(sigmas, SNRs, color = "purple")
plt.xlabel("Ecart-type du bruit blanc gaussien ajouté")
plt.ylabel("SNR")
plt.legend()
plt.show()


#Pour créer la Heatmap (x,y) = (niveau de bruit, niveau de décomposition)

j=0; i=0
SNR_decomp = np.ndarray((15,7)) 
sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]

for j in range(1, 7):
    for i in range(len(sigmas)):
        test, SNR = rec_signal(s512, sigmas[i], "db4", j)
        SNR_decomp[i][j] = SNR

for sig in range(len(sigmas)):
    SNR_decomp[sig][0] = sigmas[sig]



