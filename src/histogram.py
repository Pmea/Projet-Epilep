# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 22:29:04 2016

@author: Pmea
"""


import numpy as np

from moviepy.editor import *
import matplotlib.pyplot as plt

import tfct
import os
import sys
import math

import scipy.signal

def create_histogram_for_image(img, clr_res):
    p= np.zeros(( 2**clr_res, 2**clr_res, 2**clr_res ))   #color RGB

    img.astype(float)
    img= np.multiply(img, 2**clr_res )
    img= np.divide(img, 2**8)
    img.astype(int)

    for line in img:
        for pixel in line:
            p[pixel[0], pixel[1], pixel[2]]+=1
    return p

def compute_CHD(video):
    nb_frames= video.shape[0]
    nb_pixel_per_frame=  video[0].shape[0] * video[0].shape[1]

    p_array= np.empty((nb_frames), dtype= object)
    for frame in range(nb_frames):
        p_array[frame]= create_histogram_for_image(video[frame], 4)

    chd= np.zeros((nb_frames-1))
    for frame in range(nb_frames-1):
        chd[frame]= np.sum( np.abs(p_array[frame+1] - p_array[frame]) )

    chd= np.divide(chd, nb_pixel_per_frame)
    chd= np.divide(chd, np.sum(chd))
    
    return chd

def compute_FAD(video):
    nb_frames= video.shape[0]
    video_width= video.shape[1]
    video_height= video.shape[2]
    nb_pixel_per_frame= video_height * video_width

    deriv_p= np.zeros((nb_frames-1))

    for f in range(nb_frames-1):
        diff_pixels= np.abs( video[f+1,:,:,:] - video[f,:,:,:] )
        diff_frame= np.sum(diff_pixels, axis=2)
        deriv_p[f]= np.sum(diff_frame)

    deriv_p= np.divide(deriv_p, nb_pixel_per_frame * 200)  #le 200 est arbitraire 
    deriv_p= np.divide(deriv_p, np.sum(deriv_p))
    
    return deriv_p

def compute_tfct_one_pixel():
    pass


def compute_tfct_all_pixel(video):
    nb_frames= video.shape[0]
    video_width= video.shape[1]
    video_height= video.shape[2]
    nb_pixel_per_frame= video_height * video_width

   # deriv_p= compute_FAD(video)

    deriv_p= np.zeros((nb_frames-1))

    for f in range(nb_frames-1):
        diff_pixels= video[f+1,:,:,:] - video[f,:,:,:] #on reviens au probleme des choses qui se compence #tf pour tous les points puis somme # reflechir un autre algo
        diff_frame= np.sum(diff_pixels, axis=2)
        deriv_p[f]= np.sum(diff_frame)

    ##test 
    ## resample 
    deriv_p_resample= scipy.signal.resample( deriv_p, (nb_frames-1)*2)    
    ## multiplication des samples
    deriv_p_mult2= np.zeros(3*(nb_frames-1))
    for i in range(0, 3*(nb_frames-1)): #double le nombre d'echantillon
        deriv_p_mult2[i]= deriv_p[i/3]
   
    deriv_p_mult2_filtre= np.zeros(3*(nb_frames-1))

    for i in range(1, 3*(nb_frames-1)-1): 
        deriv_p_mult2_filtre[i]= (1/2.*deriv_p_mult2[i+1] + deriv_p_mult2[i] + 1/2. *deriv_p_mult2[i-1]) / 2.

    deriv_p= deriv_p_mult2_filtre
    #calcule de la tf global
    nfft= 2**10                                     
    winSize= 2**8
    hopRatio= 1./4

    SIG= tfct.tfct(deriv_p, nfft, winSize, hopRatio)

    SIG= np.abs(SIG)
    SIG= np.transpose(SIG)

    max_sig= np.argmax(SIG[:nfft/2], axis=0)

    plt.imshow(SIG, aspect="auto")
    plt.plot(max_sig, c='r')
    plt.show()

    #nb_deriv= deriv_p.shape()[0]
    #for f in range(0, nb_frames):
     #   for i in range(video_height):
      #      for j in range(video_width):
                
    #on fait la tfct de ce signal


def load_video(filename):
    filename = os.path.expanduser(filename)
    main_clip= VideoFileClip(filename)
    
    W,H = main_clip.size
    fps= main_clip.fps
    
    return main_clip

def videoclip_to_matrix(videoclip):
    nb_frames= math.ceil(videoclip.fps * videoclip.duration) # approximation, we can lose 1-2 frames at last
    #nb_frames= sum(1 for x in videoclip.iter_frames()) # the exact one

    video_width, video_height= videoclip.size
    video_mtx= np.zeros((nb_frames ,video_height, video_width, 3))
    print (video_mtx.shape)

    count= 0
    for frame in videoclip.iter_frames():
        print(str(count+1)+'/'+ str(nb_frames))
        video_mtx[count,:,:,:]=frame
        count+=1

    return video_mtx, nb_frames, video_width, video_height

def save_video(video, filename):
    filename = os.path.expanduser(filename)
    print (filename)
    video.write_videofile(filename, codec='libx264', audio=False)

def scale_video(video, new_width, new_height, filename_dest=None):
    print (new_height)
    scale_video= video.resize((new_width, new_height))
    print (scale_video.size)
    if filename_dest != None:
        print("SAVE")
        save_video(scale_video, filename_dest)

    return scale_video


print ("Debut du test")
filename= "~/Projet/Projet-Epilep/video/video-test-synthesis-4.mov"
main_clip= load_video(filename)
scale_clip=scale_video(main_clip, 40, 25)#, filename+"-scale.mp4")

video_mtx, nb_frames, W, H= videoclip_to_matrix(scale_clip)

CHD= compute_CHD(video_mtx)
FAD= compute_FAD(video_mtx)

plt.plot(CHD, label="CHD")
plt.plot(FAD, label="FAD")
plt.legend()
plt.show()

print ("Fin du test")



