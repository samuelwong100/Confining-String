# -*- coding: utf-8 -*-
"""
File Name: Relaxation_Full_Grid.py
Purpose: Relaxation for the full-grid method
Author: Samuel Wong
"""
import numpy as np
from numpy import pi
from copy import deepcopy
import matplotlib.pyplot as plt
from Math import SU

class Relaxation_Full_Grid():
    """
    Solve the boundary value problem of a complex Poisson equation with m
    components using relaxation method and full-grid. Uses Dirichlet boundary
    condition with the boundary values being constant.
    
    The PDE has to be of the form
    (d^2/dz^2 + d^2/dy^2) f = g(f,z,y)
    where f is a vector.
    
    Variables
    ----------------------------------------
    grid (Grid) = a Grid object that stores the grid parameters
    m (int) = number of fields (i.e. components)
    bound (array) = an array of shape (m,) that defines the boundary (vector) 
                    value of the m-component field
    laplacian (function) = the Laplacian function, also called source function.
                    Mathematically, it has the form, g(f,z,y)
                    Here, g takes a m-vector grid and returns a new grid of
                    the same shape. We assume that g returns an 
                    (m,num_y,num_z) array
    tol (float) = tolerance of error
    max_loop (int) = maximum number of loops allowed; the relaxation loop will
                    stop once this is exceeded
    x0 (str) = key word for the initial grid:
               if x0 == None, then the initial grid is uniformly equal to bound
               if x0 == "one-one", then the initial grid is uniformly 1+i 
               if x0 == "zero", then the initial grid is uniformly 0
    diagnose (bool) = whether to display progress as the code is running
    x (array) = the final solution
                if the equation has not been solved, x = None
                if the equation is solved, x is the grid variable that
                represents the field. It has an m-vector for each (z,y). Here,
                it is an array of shape (m,num_y,num_z).
    error (array/list) = the list of error; empty list if equation has not been
                    solved
    loop (int) = number of loops actually ran;
                 if equation has not been solved yet, loop = 0
    """
    def __init__(self,grid,m,bound,laplacian,tol,max_loop,x0,diagnose):
        self.grid = grid
        self.m = m
        self.bound = bound
        self.laplacian = laplacian
        self.tol = tol
        self.max_loop = max_loop
        self.x0 = x0
        self.diagnose = diagnose
        #initialize solution before equation is solved
        self.x = None
        self.error = []
        self.loop = 0
    
    def solve(self):
        """
        Solve the euqation and save result within the object.
        """
        x = self._set_x0() #initialize the vector grid
        x = self._relaxation(x) #solve using relaxation method
        self.x = x #save x into the class
    
    def _set_x0(self):
        """
        Return the initial grid, x0.
        """
        #The first index of the grid tells which field this grid is for
        #The next two index refers to the coordinate on the rectangle grid
        x0 = np.ones(shape=(self.m,self.grid.num_y,self.grid.num_z),
                         dtype=complex)
        #the default initial grid is equal to boundary everywhere
        if self.x0 is None:
            for i in range(self.m):
                x0[i,:,:] *= np.bound[i]
        else:
            #preset the initial grid options that do not depend on bound values
            if self.x0 == "one-one":
                x0 *= complex(1,1)
            elif self.x0 == "zero":
                x0 *= complex(0,0)
            #enforce the boundary condition
            x0 = self._apply_bound(x0)
        return x0
    
    def _apply_bound(self,x_old):
        x = deepcopy(x_old)
        #set all the left sides to bound
        #take all component (first index); for each component, take all rows
        #(second index); take the first/left column (third index); set it to
        #bound
        bound_2d = np.array([self.bound])
        x[:,:,0] = np.repeat(bound_2d,self.grid.num_y,axis=0).T
        #repeat for bounds in other directions
        x[:,:,-1] = np.repeat(bound_2d,self.grid.num_y,axis=0).T
        x[:,0,:] = np.repeat(bound_2d.T,self.grid.num_z,axis=1)
        x[:,-1,:] = np.repeat(bound_2d.T,self.grid.num_z,axis=1)
        return x
    
    def _relaxation(self,x):
        while self._continue_loop():
            x_new = self._update(x) # update a new grid for x
            self.error.append(self._get_error(x_new,x)) # get new error
            x = x_new #set x to the new grid
            self.loop += 1
            self._diagnostic_plot(x)
        self.error = np.array(self.error) #change error into an array
        return x
    
    def _continue_loop(self):
        #return whether to continue the relaxation loop
        contin = True
        if self.error == []:
            #continue if no trials had occured yet
            contin = True
        elif self.error[-1] < self.tol:
            #discontinue if the latest error is already less than tolerance
            contin = False
        elif self.loop > self.max_loop:
            #discontinue if maximum loop is reached
            contin = False
        return contin
    
    def _update(self,x_old):
        # replace each element of x_old with average of 4 neighboring points,
        # plus a source term;
        x = deepcopy(x_old)
        source_term = self.laplacian(x)
        # we loop over each element in the grid, skipping over the edge.
        # so we start loop at 1, avoiding 0. We ignore the last one by -1
        for row in range(1,self.grid.num_y-1):
            for col in range(1,self.grid.num_z-1):
                x[:,row,col] = (x[:,row-1,col] + x[:,row+1,col] +
                              x[:,row,col-1] + x[:,row,col+1]
                              - source_term[:,row,col]*(self.grid.h)**2)/4
        # since we skipped over the edge, the Dirichlet boundary is automatically
        #enforced. If we want Nuemann boundary, then we alter edge values
        if self.bound is None:
            #set the boundary such that normal derivative is zero
            x[:,0,:]=x[:,1,:]
            x[:,-1,:]=x[:,-2,:]
            x[:,:,0]=x[:,:,1]
            x[:,:,-1]=x[:,:,-2]
        return x
    
    def _get_error(self,x_new,x):
        # we define error to be the average absolute value of the difference
        # in each component of the matrix.
        normalization= self.m * self.grid.num_z * self.grid.num_y
        return np.sum(np.abs(x_new - x))/normalization

    def _diagnostic_plot(self,x):
        if self.diagnose and self.loop % 50 == 0:
            print("loop =",self.loop,"error =",self.error[-1])
            for i in range(self.m):
                plt.figure()
                plt.pcolormesh(self.grid.zv,self.grid.yv,np.real(x[i,:,:]))
                plt.colorbar()
                plt.show()

                plt.figure()
                plt.pcolormesh(self.grid.zv,self.grid.yv,np.imag(x[i,:,:]))
                plt.colorbar()
                plt.show()
