import numpy as np
import matplotlib.pyplot as plt  # temporaire

import scipy.io.wavfile
import math

def tfct(sig, fe, nfft, winSize, hopSize):
		
	hopSize= int(math.floor(winSize * hopRasio))
	sigSize= sig.shape[0]

	#calculer le nombre de trame
	nbBin= ((sigSize - winSize) // hopSize) 
	
	#reserver de la memoire pour le resultat
	resTfct= np.empty((nbBin, nfft), np.complex128)
	
	#calculer la fenetre de hanning
	win= np.hanning(winSize)

	#pour toute les trames calculer la fft
	for b in range(0, nbBin-1):
		indDeb= b * hopSize
		indFin= (b * hopSize) + winSize
		resTfct[b,:]= np.fft.fft( sig[indDeb:indFin] * win , nfft)

	return resTfct


#def main():
fe, sig= scipy.io.wavfile.read("audio_gammepno.wav")	 # le wav doit avoir un seul canal

nfft= 2**10
winSize= 2**10
hopRasio= 1./4

SIG= tfct(sig, fe, nfft, winSize, hopSize)

#plotTmp= np.abs(resTfct)			#test d'affichage
#plotTmp= np.transpose(plotTmp)
#plt.imshow(plotTmp, aspect='auto')
#plt.show()





#if __name__ == "__main__":
#	main()