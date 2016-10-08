import numpy as np
import matplotlib.pyplot as plt  # temporaire

import scipy as sp
import scipy.io.wavfile
import math
import sounddevice as sd


def tfct(sig, nfft, winSize, hopRatio):
		
	hopSize= int(math.floor(winSize * hopRatio))
	sigSize= sig.shape[0]

	#calculer le nombre de trame
	nbBin= ((sigSize - winSize) // hopSize) 
	
	#reserver de la memoire pour le resultat
	resTfct= np.empty((nbBin, nfft), np.complex128)
	
	#calculer la fenetre de hanning
	win= sp.signal.blackmanharris(winSize)

	#pour toute les trames calculer la fft
	for b in range(0, nbBin):
		indDeb= b * hopSize
		indFin= (b * hopSize) + winSize
		resTfct[b,:]= np.fft.fft( sig[indDeb:indFin] * win , nfft)

	return resTfct

def ola(win, winSize, hopSize, nbBin): #ola va calculer le gain que l'adition de fenetre va apporter
	winOla= np.zeros(nbBin * hopSize + winSize)
	for b in range(0,nbBin):
		indDeb= b * hopSize
		indFin= (b * hopSize) + winSize
		winOla[indDeb: indFin]= winOla[indDeb: indFin] + win

	return winOla

def itfct(SIG, nfft, winSize, hopRatio):	
	hopSize= int(math.floor(winSize * hopRatio))
	SIGSize= SIG.shape[0]

	#calculer le nombre de trame
	nbBin= SIGSize
	sigSize= SIGSize * hopSize + winSize

	#reserver de la memoire pour le resultat
	sigSyn= np.zeros(sigSize)

	#calculer la fenetre de blackman-harris4
	win= sp.signal.blackmanharris(winSize)

	for b in range(0, nbBin):
		indDeb= b * hopSize
		indFin= (b * hopSize) + winSize
		
		ys = np.real(np.fft.ifft(SIG[b,:]))						##probleme dans ma ifft
		ys= ys * win
		sigSyn[indDeb:indFin]= sigSyn[indDeb:indFin] + ys 	

	#normaliser le signal
	winOla= ola(win*win, winSize, hopSize, nbBin)
	sigSyn= sigSyn / np.max(winOla)

	return sigSyn



def main():
	fe, sig= scipy.io.wavfile.read("audio_gammepno.wav")	 # le wav doit avoir un seul canal

	nfft= 2**14
	winSize= 2**10
	hopRatio= 1./8.

	SIG= tfct(sig, nfft, winSize, hopRatio)

	plotTmp= np.abs(SIG)			#test d'affichage
	plotTmp= np.transpose(plotTmp)
	plt.imshow(plotTmp, aspect='auto')
	plt.show()

	sigSyn= itfct(SIG, nfft, winSize, hopRatio)

	plt.plot(sig)
	plt.plot(sigSyn)
	plt.show()

	sig= np.divide(sig, np.max(sig))
	sigSyn= np.divide(sigSyn, np.max(sigSyn))

	#sd.play(sig)
	#sd.play(sigSyn)


if __name__ == "__main__":
	main()