# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 22:29:04 2016

@author: Pmea
"""


import numpy as np

from moviepy.editor import *

import os
import sys



def create_histogram_for_image(img, clr_res):
    p= np.zeros(( 2**clr_res, 2**clr_res, 2**clr_res ))   #color RGB
    for line in img:
        for pixel in line:
            val_r= pixel[0] // clr_res  
            val_g= pixel[1] // clr_res
            val_b= pixel[2] // clr_res
            p[val_r, val_g, val_b]+=1 
    return p

def compute_CHD(video):
    pass


def load_video(filename):
    filename = os.path.expanduser(filename)
    main_clip= VideoFileClip(filename)
    
    W,H = main_clip.size
    fps= main_clip.fps
    
    print (fps)
    return main_clip

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


def main():
    print ("Debut du test")
    filename= "~/Projet/epilspie/video-real-2.mp4"
    main_clip= load_video(filename)
    scale_clip=scale_video(main_clip, 200,320, filename+"-scale.mp4")
    print (scale_clip.size)

    print ("Fin du test")
    



if __name__== '__main__':
    main()
