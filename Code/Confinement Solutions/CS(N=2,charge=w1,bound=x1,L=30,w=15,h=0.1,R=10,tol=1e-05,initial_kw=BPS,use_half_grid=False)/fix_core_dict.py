# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 02:26:27 2020

@author: samue
"""
import pickle

pickle_in = open("core_dict","rb")
core_dict = pickle.load(pickle_in)
new_key = "initial field"
old_key = "initital field"
core_dict[new_key] = core_dict.pop(old_key)
with open("core_dict2","wb") as file:
    pickle.dump(core_dict, file)