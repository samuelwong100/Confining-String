# -*- coding: utf-8 -*-
"""
File Name: Relaxation.py
Purpose: Relaxation algorithm for solving Poisson BVP
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
sys.path.append("../BPS_Package")
import numpy as np
from numpy import pi
from copy import deepcopy
import matplotlib.pyplot as plt
from Math import Superpotential
from solve_BPS import solve_BPS

class Relaxation():
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
    N (int) = number of color
    m (int) = number of fields (i.e. components)
    bound (array) = an array of shape (m,) that defines the boundary (vector) 
                    value of the m-component field
    charge (array) = an array of shape (m,) that describes the charge (usually
                     a linear combo of fundamental weights)
    laplacian (function) = the Laplacian function, also called source function.
                    Default to be the full-grid EOM.
                    Mathematically, it has the form, g(f,z,y)
                    Here, g takes a m-vector grid and returns a new grid of
                    the same shape. We assume that g returns an 
                    (m,num_y,num_z) array.
    tol (float) = tolerance of error; the relaxation loop will stop once the
                  error becomes smaller than the tolerance
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
    def __init__(self,grid,N,bound,charge,tol,max_loop,x0,diagnose):
        self.grid = grid
        self.N = N
        self.m = N-1
        self.bound = bound
        self.charge = charge
        self.bound_vec = bound.imaginary_vector
        self.charge_vec = charge.real_vector
        self.tol = tol
        self.max_loop = max_loop
        self.x0 = x0
        self.diagnose = diagnose
        self.W = Superpotential(N)
        #default laplacian function is the full-grid EOM (equation of motion)
        #this can be altered by creating a child class
        self.laplacian = self._full_grid_EOM
        #initialize solution before equation is solved
        self.x = None
        self.error = []
        self.loop = 0
        #create a constant source term, which is independent of field
        self.source_term = self._define_source_term()
    
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
                x0[i,:,:] *= self.bound_vec[i]
        else:
            #preset the initial grid options that do not depend on bound values
            if self.x0 == "one-one":
                x0 *= complex(1,1)
            elif self.x0 == "zero":
                x0 *= complex(0,0)
            elif self.x0 == 'BPS':
                x0 = self._BPS_x0(x0)
            #enforce the boundary condition
            x0 = self._apply_bound(x0)
            print("Plot initial grid")
            for i in range(self.m):
                plt.figure()
                plt.pcolormesh(self.grid.zv,self.grid.yv,np.real(x0[i,:,:]))
                plt.colorbar()
                plt.title("$\phi$"+str(i+1))
                plt.show()

                plt.figure()
                plt.pcolormesh(self.grid.zv,self.grid.yv,np.imag(x0[i,:,:]))
                plt.colorbar()
                plt.title("$\sigma$"+str(i+1))
                plt.show()
        return x0

    def _BPS_x0(self,x0):
        #the following BPS only works in SU(2) and boundary is 0
        #then the inside vacua is +/- x_1 across origin. So the initial grid
        #should be 2 BPS, one being the negative anti-kink of the other
        #In general, this is too complicated
        if self.N == 2 and str(self.bound)== "x0":
            y_half,x_half,B = solve_BPS(N=self.N,vac0_arg=str(self.bound),
                                      vacf_arg="x1",z0=self.grid.y0,
                                      zf=0,h=self.grid.h,folder="",
                                      tol=1e-5,save_plot=False)
            #need some numpy shape gymnastic here
            #BPS solution is stored in the form of (y,m), where the rows y
            #are the points, and the columns, m, are the fields
            #Confining string solutions are stored in the form of (m,y,z),
            #where the layers m are the field, the rows y are the vertical,
            #the columns z are the horizontal
            #x_half_transpose has shape (m,y)
            x_half_transpose = x_half.T
            half_num = x_half_transpose.shape[1]
            #a verticle slice of x should be two reversed BPS walls merged
            #together, going from boundary to inner vacua and then negative vacua
            #comes back back (due to the relaxation, it the discontinuity in the
            #middle in this case is +/- x_1)
            x_slice = np.zeros(shape=(self.m,self.grid.num_y),dtype=complex)
            x_slice[:,0:half_num] = x_half_transpose
            x_slice[:,-1-half_num:-1] = -np.flip(x_half_transpose,1)
            #first set x0 entirely equal to boundary values
            for i in range(self.m):
                x0[i,:,:] *= self.bound_vec[i]
            #for the 2 columns between the two charges, set to x_slice
            for k in range(self.grid.num_z):
                if self.grid.left_axis <= k <= self.grid.right_axis:
                    x0[:,:,k] = x_slice
        elif self.N ==3 and str(self.bound) == "x1":
            y_half,x_top_half,B_top = solve_BPS(N=self.N,vac0_arg=str(self.bound),
                          vacf_arg="x0",z0=self.grid.y0,
                          zf=0,h=self.grid.h,folder="",
                          tol=1e-5,save_plot=False)
            y_half,x_bottom_half,B_bottom = solve_BPS(N=self.N,vac0_arg=str(self.charge),
                          vacf_arg=str(self.bound),z0=self.grid.y0,
                          zf=0,h=self.grid.h,folder="",
                          tol=1e-5,save_plot=False)
            x_top_half_trans = x_top_half.T
            x_bottom_half_trans = x_bottom_half.T
            half_num = x_top_half_trans.shape[1]
            x_slice = np.zeros(shape=(self.m,self.grid.num_y),dtype=complex)
            x_slice[:,-1-half_num:-1] = np.flip(x_top_half_trans,1)
            x_slice[:,0:half_num] = np.flip(x_bottom_half_trans,1)
            #first set x0 entirely equal to boundary values
            for i in range(self.m):
                x0[i,:,:] *= self.bound_vec[i]
            #for the 2 columns between the two charges, set to x_slice
            for k in range(self.grid.num_z):
                if self.grid.left_axis <= k <= self.grid.right_axis:
                    x0[:,:,k] = x_slice
        #save BPS and initial grid
        self.B_top = B_top #store BPS object
        self.B_bottom = B_bottom
        self.top_BPS = x_top_half
        self.bottom_BPS = x_bottom_half
        self.y_half = y_half
        self.BPS_slice = x_slice
        self.initial_grid = x0
        return x0
    
    def _apply_bound(self,x_old):
        x = deepcopy(x_old)
        #set all the left sides to bound
        #take all component (first index); for each component, take all rows
        #(second index); take the first/left column (third index); set it to
        #bound
        bound_2d = np.array([self.bound_vec])
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
        #enforced.
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
                plt.title("$\phi$"+str(i+1))
                plt.show()

                plt.figure()
                plt.pcolormesh(self.grid.zv,self.grid.yv,np.imag(x[i,:,:]))
                plt.colorbar()
                plt.title("$\sigma$"+str(i+1))
                plt.show()

    def _full_grid_EOM(self,x):
        return self._source_term + self._potential_term(x)
    
    def _potential_term(self,x):
        return self.W.potential_term_on_grid_fast_optimized(x)
    
    def _define_source_term(self):
        # return i 2pi C_a d(delta(y))/dy int_{-R/2}^{R/2} delta(z-z')dz'
        result = np.zeros(shape=(self.m,self.grid.num_y,self.grid.num_z),
                          dtype=complex)
        #the derivative of delta function in y gives something close to infinity
        #for y just below 0 and somthing close to -infinity for y just above 0
        #here, we have x(y = 0^{-}) = 1/h^2 and x(y=0^{+})= -1/h^2
        #note that the lower row correspond to higher y
        inf = 1/(self.grid.h**2)
        result[:,self.grid.z_axis-1,:] = -inf
        result[:,self.grid.z_axis,:] = inf
        #set grid to 0 unless it is on z_axis and between -R/2 and R/2
        result[:,:,0:self.grid.left_axis]=0 #left_axis included in source
        result[:,:,self.grid.right_axis+1:]=0 #right_axis included in source
        #multiply everything by charge (outside relevant rows, everything is
        #zero anyway)
        for i in range(self.N-1):
            result[i,:,:] *= self.charge_vec[i]
        #overall coefficient
        coeff = 1j*2*pi
        result = coeff*result
        return result
    
class Continue_Relaxation(Relaxation):
    def __init__(self,old_sol,new_max_loop,charge,bound,diagnose):
        super().__init__(old_sol.grid, old_sol.N, bound,charge, old_sol.tol,
             new_max_loop, old_sol.x0, diagnose)
        #the loop and error do not start at 0 and empty list!
        self.loop = old_sol.loop
        self.error = list(old_sol.error) #change error from array back to list
        #initialize solution to old field
        self.x = old_sol.x
        #store BPS related object if original field solution started with BPS
        if self.x0 == 'BPS':
            self.B_top = old_sol.B_top
            self.B_bottom = old_sol.B_bottom
            self.top_BPS = old_sol.top_BPS
            self.bottom_BPS = old_sol.bottom_BPS
            self.y_half = old_sol.y_half
            self.BPS_slice = old_sol.BPS_slice
            #the initial grid here denotes the initial grid of the old soluiton
            #of course, the continue relaxation initial grid is the final 
            #grid of the old solution, but that is store is self.x
            #Here, it is as if we started with some BPS and just ran longer
            self.initial_grid = old_sol.initial_grid
        
    def _set_x0(self):
        #overide initial field function to be self.x, which was set to old
        #field already in the init
        return self.x