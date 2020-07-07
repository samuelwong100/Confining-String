# -*- coding: utf-8 -*-
"""
File Name: replot_all_BPS.py
Purpose: 
Author: Samuel Wong
"""
import os
import pickle
from Source import plot_BPS

def get_all_folders():
    folders = []
    for x in os.walk("BPS Solitons"):
        folders.append(x[0]+"/")
    folders.remove("BPS Solitons/")
    return folders

folders = get_all_folders()
for folder in folders:
    pickle_in = open(folder+"BPS_dict","rb")
    BPS_dict = pickle.load(pickle_in)

        
    

#plot_BPS(N,z,f,num,h,vac0,vacf,save_plot,folder)