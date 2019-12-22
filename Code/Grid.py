# -*- coding: utf-8 -*-
"""
File Name: Grid.py
Purpose: General class for storing grid parameters.
Author: Samuel Wong
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import sqrt

class Grid():
    """
    Store the defining grid parameters.
    Compute and store the number of rows, columns, axis location.
    
    Constants
    ----------------------------------------
    z0 (float) = left-most point of grid
    zf (float) = right-most point of grid
    y0 (float) = bottom point of grid
    yf (float) = top point of grid
    h (float) = grid pixel
    num_z (int) = number of points in z direction
    num_y (int)  = number of points in y direction
    z_axis (int) = row number of z-axis; None if not applicable
    y_axis (int) = column number of y-axis; None if not applicable
    z (array) = an array from z0 to zf spaced by h
    y (array) = an array from y0 to yf spaced by h
    zv (array) = z-component of meshgrid
    yv (array) = y-component of meshgrid
    """
    
    def __init__(self,z0,zf,y0,yf,h):
        # first check the validity of the input
        self._validate(z0,zf,y0,yf,h)
        # then store the relevant grid parameters
        self.z0 = z0
        self.zf = zf
        self.y0 = y0
        self.yf = yf
        self.h = h
        self.num_z = int((zf-z0)/h)
        self.num_y = int((yf-y0)/h)
        if z0<=0 and zf>=0:
            self.y_axis = int((abs(z0)/(zf-z0))*self.num_z)
        else:
            self.y_axis = None
        if y0<=0 and yf>=0:
            self.z_axis = int((abs(y0)/(yf-y0))*self.num_y)
        else:
            self.z_axis = None
        self.z = np.linspace(z0,zf,self.num_z)
        self.y = np.linspace(y0,yf,self.num_y)
        self.zv, self.yv = np.meshgrid(self.z, self.y)
        
    def _validate(self,z0,zf,y0,yf,h):
        if zf <= z0:
            raise Exception("zf cannot be less than z0.")
        if yf <= y0:
            raise Exception("yf cannot be less than y0.")
        if h >= zf-z0 or h>= yf-y0:
            raise Exception("h is too large compared to the grid.")
            
    def plot_empty_grid(self):
        """
        Plot an empty grid to show what the grid looks like.
        """
        #plot an empty grid
        f = np.ones(shape=(self.num_y,self.num_z))*np.nan
        # make a figure + axes
        fig, ax = plt.subplots(1, 1,figsize = (10,10))
        # make color map
        cmap = matplotlib.colors.ListedColormap(['r', 'g', 'b'])
        # set the 'bad' values (nan) to be white and transparent
        cmap.set_bad(color='w', alpha=0)
        # draw the grid
        for z in self.z:
            ax.axvline(z, lw=2, color='k', zorder=5)
        for y in self.y:
            ax.axhline(y, lw=2, color='k', zorder=5)
        if self.y_axis is not None:
            ax.axvline(self.z[self.y_axis],color='r',lw=2,zorder=5)
        if self.z_axis is not None:
            ax.axhline(self.y[self.z_axis],color='r',lw=2,zorder=5)
        # draw the boxes
        ax.imshow(f, interpolation='none', cmap=cmap, 
                  extent=[self.z0, self.zf,self.y0, self.yf],
                  zorder=0)
        fig.suptitle("Empty Grid",fontsize=20)
        
class Grid_Dipole(Grid):
    #assumes zero is at the center
    def __init__(self,z0,zf,y0,yf,h,R_fraction):
        self._validate_R_fraction(R_fraction) #check validity of R_fraction
        Grid.__init__(self,z0,zf,y0,yf,h) #call parent class constructor
        self.R_fraction = R_fraction #fraction of distance of dipole over horizontal length
        self.R = R_fraction*(zf-z0) #actual distance between dipoles
        self.right_charge = R_fraction*(zf-z0)/2 #location of right charge
        self.left_charge = -self.right_charge #location of left charge
        #axis number of right charge
        self.right_axis = self.y_axis + int(self.R_fraction*self.num_z/2)
        #axis number of left charge
        self.left_axis = self.y_axis - int(self.R_fraction*self.num_z/2)
        
    def _validate_R_fraction(self,R_fraction):
        if R_fraction >= 1:
            raise Exception("R_fraction cannot be greater than or equal to 1.")
            
    def plot_empty_grid(self):
        """
        Plot an empty grid to show what the grid looks like.
        """
        #plot an empty grid
        f = np.ones(shape=(self.num_y,self.num_z))*np.nan
        # make a figure + axes
        fig, ax = plt.subplots(1, 1,figsize = (10,10))
        # make color map
        cmap = matplotlib.colors.ListedColormap(['r', 'g', 'b'])
        # set the 'bad' values (nan) to be white and transparent
        cmap.set_bad(color='w', alpha=0)
        # draw the grid
        for z in self.z:
            ax.axvline(z, lw=2, color='grey', zorder=5)
        for y in self.y:
            ax.axhline(y, lw=2, color='grey', zorder=5)
        if self.y_axis is not None:
            ax.axvline(self.z[self.y_axis],color='r',lw=2,zorder=5)
        if self.z_axis is not None:
            ax.axhline(self.y[self.z_axis],color='r',lw=2,zorder=5)
        ax.axvline(self.z[self.left_axis],color='r',lw=2,zorder=5)
        ax.axvline(self.z[self.right_axis],color='r',lw=2,zorder=5)
        # draw the boxes
        ax.imshow(f, interpolation='none', cmap=cmap, 
                  extent=[self.z0, self.zf,self.y0, self.yf],
                  zorder=0)
        fig.suptitle("Empty Grid",fontsize=20)
        
class Standard_Dipole_Grid(Grid_Dipole):
    """
    Store the defining grid parameters for a standard grid with dipole in the
    middle.
    Compute and store the number of rows, columns, axis location.
    
    Constants
    ----------------------------------------
    L (float) = horizontal length of grid
    w (float) = verticle width of grid
    R_fraction (float) = fraction (decimal) of distance of dipole over horizontal length
    R (float) = distance between two charges
    right_charge (float) = location of right charge
    left_charge (float) = location of left charge
    right_axis (int) = axis number of right charge
    left_axis (int) = axis number of left charge
    z0 (float) = left-most point of grid
    zf (float) = right-most point of grid
    y0 (float) = bottom point of grid
    yf (float) = top point of grid
    h (float) = grid pixel
    num_z (int) = number of points in z direction
    num_y (int)  = number of points in y direction
    z_axis (int) = row number of z-axis; None if not applicable
    y_axis (int) = column number of y-axis; None if not applicable
    z (array) = an array from z0 to zf spaced by h
    y (array) = an array from y0 to yf spaced by h
    zv (array) = z-component of meshgrid
    yv (array) = y-component of meshgrid
    """
    def __init__(self,L,w,h,R):
        self.L = L
        self.w = w
        Grid_Dipole.__init__(self,-L/2,L/2,-w/2,w/2,h,R/L)
        