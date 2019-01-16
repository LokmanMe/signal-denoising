import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import pywt


a = read("son.wav")
np.array(a[1],dtype=float)

son = a[1]

plt.figure()
plt.title("Un son furtif de maracas trouvé sur le net")
plt.plot(son)
plt.show()

ond = "db4" 
n0=4
coeffs_l = pywt.wavedec(son, ond, level=n0) 
coefficients = np.concatenate(coeffs_l) 

plt.figure()
plt.title("Décomposition en ondelettes")
plt.plot(coefficients)
plt.show()

son_rec = pywt.waverec(coeffs_l, ond)   #juste pour tester qu'on peut bien le recomposer
plt.figure()
plt.title("Son recomposé")
plt.plot(son_rec)
plt.show()



def addwhitenoise (x, sigma):  
    bruit = np.random.normal(0, sigma, x.size) 
    return x + bruit

son_bruite = addwhitenoise(son, 500)
son_rec = pywt.waverec(coeffs_l, ond)  
plt.figure()
plt.title("Son bruité")
plt.plot(son_bruite)
plt.show()

son_bruite = son_bruite.astype(np.int16)
write("son_bruite.wav", 44100, son_bruite) #si on compare les sons à l'oreille, on entend bien qu'il est différent maintenant.

#Essayons de débruiter cela avec une décomposition par ondelettes puis seuillage

ond = "db22" 
n0=7
coeffs_son_bruite = pywt.wavedec(son_bruite, ond, level=n0) 
coefficients_son_bruite = np.concatenate(coeffs_son_bruite) 

plt.figure()
plt.title("Décomposition en ondelettes du son bruité")
plt.plot(coefficients_son_bruite)
plt.show()

#Maintenant on seuille 

for i in range(1, len(coeffs_son_bruite)):     
    coeffs_son_bruite[i] = pywt.threshold(coeffs_son_bruite[i], 0.05,'soft') 
    son_debruite = pywt.waverec(coeffs_son_bruite, ond) 

plt.figure()
plt.title("Recomposition à partir de la base d'ondelettes")
plt.plot(son_debruite, label = "son recomposé", color = "blue")
plt.plot(son, label="son original", color = "orange")
plt.legend()
plt.show()

son_debruite = son_debruite.astype(np.int16)
write("son_debruite.wav", 44100, son_debruite) 

#Graphiquement on a l'impression que la décomposition a bien réussi à débruiter le son, mais à l'oreille le son bruité = son débruité... :(

s = 10 * np.log10 (np.mean(son**2) / np.mean ( ( son * son_rec[0:8779, ] )**2 ) )
print("Le SNR est égal à : ", s)















