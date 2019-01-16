import numpy as np
import matplotlib.pyplot as plt
import pywt

def addwhitenoise (x, sigma):
    bruit = np.random.normal(0, sigma, x.size) 
    return x + bruit 

s512 = np.loadtxt("signal.txt")
s512_noised = addwhitenoise(s512, 0.05)

def rec_signal(x_noised, T):   #on donne à la fonction les paramètres : signal bruité, seuil du soft_threshold
    
    coeffs_x = pywt.wavedec(x_noised, "db4", level=3)    # décompose
    
    for i in range(1, len(coeffs_x)):        # seuillage doux     
        coeffs_x[i] = pywt.threshold(coeffs_x[i], T,'soft') 
        x_denoised = pywt.waverec(coeffs_x, "db4")
        
    s = 10 * np.log10 (np.mean(s512**2) / np.mean ( ( s512 * x_denoised )**2 ) )      #SNR
    print(s)
    
    return (x_denoised, s)


seuil=np.ndarray([100,2])*0

s = 0
for i in range(100):
    seuil[i][0] = s
    s = s + 0.01
    
for i in range(100):
    test, SNR = rec_signal(s512_noised, seuil[i][0])
    seuil[i][1] = SNR



plt.figure()
plt.title("SNR en fonction du seuil")
plt.plot(seuil[:,0], seuil[:,1], color = "purple")
plt.xlabel("seuil")
plt.ylabel("SNR")
plt.legend()
plt.show()

# Je ne comprends pas pourquoi, plus le seuil augmente, meilleur est le signal recomposé.
# C'est probablement faux...