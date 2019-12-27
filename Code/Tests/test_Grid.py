# -*- coding: utf-8 -*-
"""
File Name: test_Grid.py
Purpose:  test Grid class.
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
from Grid import Grid, Grid_Dipole, Standard_Dipole_Grid

def test_Grid():
    #should see red lines on axis
    g1 = Grid(-3,3,-3,3,0.1)
    g1.plot_empty_grid()
    
    #should only see one red line
    g2 = Grid(0.5,3,-3,3,0.1)
    g2.plot_empty_grid()
    
def test_Grid_Dipole():
    #should see red lines on axis
    g = Grid_Dipole(-3,3,-3,3,0.1,0.9)
    g.plot_empty_grid()
    
def test_Standard_Dipole_Grid():
    g = Standard_Dipole_Grid(L=30,w=10,h=0.5,R=5)
    g.plot_empty_grid()
    
def test_Standard_Dipole_Grid_charge_location():
    G = Standard_Dipole_Grid(10,10,0.1,2)
    print("right = ", G.right_charge) #should be 1
    print("left = ", G.left_charge) #should be -1

#test_Standard_Dipole_Grid() #pass
#test_Standard_Dipole_Grid_charge_location() #pass