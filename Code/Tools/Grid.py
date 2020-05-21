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
    #WARNING: assumes zero is at the horizontal center of the grid
    def __init__(self,z0,zf,y0,yf,h,R_fraction):
        self._validate_R_fraction(R_fraction) #check validity of R_fraction
        #check that the grid has the y axis at its horizontal center
        self._validate_center(z0,zf)
        Grid.__init__(self,z0,zf,y0,yf,h) #call parent class constructor
        #fraction of distance of dipole over horizontal length
        self.R_fraction = R_fraction
        self.right_charge = R_fraction*(zf-z0)/2 #location of right charge
        self.left_charge = -self.right_charge #location of left charge
        #axis number of right charge
        self.right_axis = self.y_axis + int(self.R_fraction*self.num_z/2)
        #axis number of left charge
        self.left_axis = self.y_axis - int(self.R_fraction*self.num_z/2)
        
    def _validate_R_fraction(self,R_fraction):
        if R_fraction >= 1:
            raise Exception("R_fraction cannot be greater than or equal to 1.")
            
    def _validate_center(self,z0,zf):
        if not np.abs(zf+z0) < 0.00001: #make sure 0 is the center (approx)
            raise Exception("z_0 must be equal to -z_f")
            
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
        self.R = R
        Grid_Dipole.__init__(self,-L/2,L/2,-w/2,w/2,h,R/L)
        
class Half_Grid(Grid):
    def __init__(self,sdg):
        #sdg is a Standard Dipole Grid
        self.L = sdg.L
        self.w = sdg.w
        self.R = sdg.R
        self.R_fraction = sdg.R_fraction
        #call the parent class, which is a general grid, with y-width unchanged
        #but the z-length reduced to half, starting at zero
        super().__init__(z0=0, zf=self.L/2, y0=-self.w/2, yf=self.w/2,
                      h=sdg.h)
        #Note: for the following 2 lines, no longer need to divide by 2
        #since the half grid is already reflected in the zf and num_z values
        #being half of original
        self.right_charge = self.R_fraction*self.zf
        #axis number of right charge
        self.right_axis = int(self.R_fraction*self.num_z)
        
    def generate_full_grid(self):
        return Modified_Dipole_Grid(self)
        
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
        ax.axvline(self.z[self.right_axis],color='r',lw=2,zorder=5)
        # draw the boxes
        ax.imshow(f, interpolation='none', cmap=cmap, 
                  extent=[self.z0, self.zf,self.y0, self.yf],
                  zorder=0)
        fig.suptitle("Empty Grid",fontsize=20)
        
class Modified_Dipole_Grid(Grid_Dipole):
    def __init__(self,hg):
        #hg is the original half grid
        #all major parameters are the same except for z0
        self.L = hg.L
        self.w = hg.w
        self.R = hg.R
        self.R_fraction = hg.R_fraction
        print(self.R_fraction)
        print(hg.R_fraction)
        self.h = hg.h
        self.z0 = -self.L/2
        self.zf = self.L/2
        self.y0 = -self.w/2
        self.yf = self.w/2
        #the vertical y list doesn't change
        self.y = hg.y
        #multiply z by -1, exclude the first element, which is 0, flip
        z_left = np.flip((-1*hg.z)[1:])
        self.z = np.concatenate((z_left,hg.z))
        #To find the left and right axis, we use the fact that 
        #hg.num_z-1 is the last index of the half grid, and by a reflection
        #symmetry, it is also the z-index of the origin in the new grid
        self.middle_z = hg.num_z-1
        #the new left and right axis are equidistance to the new middle z 
        #with a distance equal to the half grid right axis distance
        self.left_axis = self.middle_z - hg.right_axis
        self.right_axis = self.middle_z + hg.right_axis
        #for a visual description proof of the above calculation, see image
        #in Feb 23 diary entry
        self.num_y = hg.num_y
        self.num_z = self.z.size
        self.zv, self.yv = np.meshgrid(self.z, self.y)
        self.y_axis = self.middle_z
        self.z_axis = hg.z_axis
        