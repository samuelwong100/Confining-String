# -*- coding: utf-8 -*-
"""
File Name: Solution_Viewer.py
Purpose: Class for describing the field solution
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
import os
import numpy as np
import pickle
from scipy.integrate import simps
import matplotlib.pyplot as plt
from Math import Superpotential
from Grid import Standard_Dipole_Grid

class Solution_Viewer():
    """
    Analyzing and displaying the field solution.
    
    Variables
    ----------------------------------------
    grid (Standard_Dipole_Grid)
    x (array) = the solution array with complex data type and with shape
                (m,grid.num_y,grid.num_z)
    m (int) = the number of dimensions of the solution field
    error (array) = the list of error
    loop (int) = number of loops actually ran
    title (str) = title of file path
    """
    def __init__(self,title):
        # check for the existence of the file path
        if os.path.exists(title+"core_dict"):
            pickle_in = open(title+"core_dict","rb")
            core_dict = pickle.load(pickle_in)
            self.N = core_dict["N"]
            self.x = core_dict["field"]
            self.m = self.x.shape[0]
            self.error = core_dict["error"]
            self.loop = core_dict["loop"]
            self.L = core_dict["L"]    
            self.w = core_dict["w"]
            self.h = core_dict["h"]
            self.R = core_dict["R"]
            self.grid = Standard_Dipole_Grid(self.L,self.w,self.h,self.R)
            self.title = title
        else:
            raise Exception("Solution file does not exist.")
            
    def get_phi_n(self,n):
        """
        Return the real part of the nth component of the vector field.
        
        Input
        -------------------------------------------
        n (int) = the component of the vector field
    
        Output
        --------------------------------------------
        result (array) = an array of shape (grid.num_y,grid.num_z);
                  the real part of the nth layer of the vector field.
        """
        if n >= self.m:
            raise Exception("n must be less than or equal to m-1.")
        return np.real(self.x)[n,:,:]
    
    def get_sigma_n(self,n):
        """
        Return the imaginary part of the nth component of the vector field.
        
        Input
        -------------------------------------------
        n (int) = the component of the vector field
    
        Output
        --------------------------------------------
        result (array) = an array of shape (grid.num_y,grid.num_z);
                  the imaginary part of the nth layer of the vector field.
        """
        if n >= self.m:
            raise Exception("n must be less than or equal to m-1.")
        return np.imag(self.x)[n,:,:]
    
    def plot_phi_n(self,n):
        """
        Plot the real part of the nth component of the vector field.
        
        Input
        -------------------------------------------
        n (int) = the component of the vector field
        """
        self._quick_plot(self.get_phi_n(n),
                         "$\phi_{}$".format(str(n)))
        
    def plot_sigma_n(self,n):
        """
        Plot the imaginary part of the nth component of the vector field.
        
        Input
        -------------------------------------------
        n (int) = the component of the vector field
        """
        self._quick_plot(self.get_sigma_n(n),
                         "$\sigma_{}$".format(str(n)))
    
    def _quick_plot(self,field,title):
        plt.figure()
        plt.pcolormesh(self.grid.zv,self.grid.yv,field)
        plt.colorbar()
        plt.title(title)
        plt.show()
        
    def plot_error(self):
        """
        Plot the error.
        """
        plt.figure()
        plt.plot(np.arange(0,self.loop,1),self.error)
        plt.title("Error")
        plt.show()
    
#
#class Solution_Viewer0():
#    """
#    Analyzing and displaying the field solution.
#    
#    Variables
#    ----------------------------------------
#    grid (Standard_Dipole_Grid)
#    x (array) = the solution array with complex data type and with shape
#                (m,grid.num_y,grid.num_z)
#    m (int) = the number of dimensions of the solution field
#    error (array) = the list of error
#    loop (int) = number of loops actually ran
#    title (str) = title of file to be saved
#    """
#    def __init__(self,grid,x,error,loop,title):
#        self.grid = grid
#        self.x = x
#        self.m = x.shape[0]
#        self.error = error
#        self.loop = loop
#        self._energy = "Not calculated yet"
#        self._energy_density = "Not calculated yet"
#        self.title = title
#        self.save()
#        
#    def print_attributes(self):
#        print()
#        print("Attributes:")
#        print("L = " + str(self.grid.L))
#        print("w = " + str(self.grid.w))
#        print("h = " + str(self.grid.h))
#        print("R = " + str(self.grid.R))
#        print("loop = " + str(self.loop))
#        print("error = " + str(self.error[-1]))
#        print("energy = " + str(self._energy))
#        print()
#        

#    

#        

#        
#    def get_gradient_energy_density(self):
#        """
#        Return the energy density from gradient of the field
#        
#        Output
#        --------------------------------------------
#        energy_density (array) = the energy density from gradient of the field;
#                                 an array of shape (grid.num_y,grid.num_z).
#        """
#        if self._energy_density == "Not calculated yet":
#            #compute the derivative in each direction
#            dxdz,dxdy = self._get_derivative()
#            #compute the square of the gradient
#            dx_squared = np.abs(dxdz)**2 + np.abs(dxdy)**2
#            #add up energy over each component of field
#            energy_density = dx_squared.sum(axis=0)
#            #save result
#            self._energy_density = energy_density
#            self.save()
#        else:
#            energy_density = self._energy_density
#        return energy_density
#
#    def get_gradient_energy(self):
#        """
#        Return the value of the gradient energy
#        
#        Output
#        --------------------------------------------
#        energy (float) = the total gradient energy
#        """
#        if self._energy == "Not calculated yet":
#            #integrate to get energy
#            energy = simps(simps(self.get_gradient_energy_density(), 
#                                 self.grid.z),self.grid.y)
#            self._energy = energy #save into object
#            self.save()
#        else:
#            energy = self._energy
#        return energy
#    
#    def plot_gradient_energy_density(self,save=False):
#        """
#        Plot the gradient energy density.
#        """
#        plt.figure()
#        plt.pcolormesh(self.grid.zv,self.grid.yv,
#                       self.get_gradient_energy_density(),cmap='jet')
#        plt.colorbar()
#        #plt.title("Gradient Energy Density")
#        if save:
#            plt.savefig(self.title+" Gradient Energy Density.png")
#        plt.show()
#    
#    def save(self):
#        """
#        Save this Field Solution object
#        """
#        with open(self.title, 'wb') as file:
#            pickle.dump(self, file)
#            
#    def get_laplacian(self):
#        #initialize second derivative in each direction
#        d2xdz = np.zeros(shape=self.x.shape,dtype=complex)
#        d2xdy = np.zeros(shape=self.x.shape,dtype=complex)
#        for i in range(self.m): #loop over each layer
#            for j in range(self.grid.num_y): #loop over each row
#                for k in range(self.grid.num_z): #loop over each column
#                    d2xdz[i][j][k] = self._get_d2xdz_ijk(i,j,k)
#                    d2xdy[i][j][k] = self._get_d2xdy_ijk(i,j,k)
#        return d2xdz + d2xdy
#        
#    def _get_d2xdz_ijk(self,i,j,k):
#        if k == 0: #one sided second derivative on the edge
#            result = (self.x[i][j][k+2] - 2*self.x[i][j][k+1] +
#                      self.x[i][j][k])/(self.grid.h**2)
#        elif k==self.grid.num_z-1: #one sided second derivative on the edge
#            result = (self.x[i][j][k] - 2*self.x[i][j][k-1] +
#                      self.x[i][j][k-2])/(self.grid.h**2)
#        else: #two sided second derivative elsewhere
#            result = (self.x[i][j][k+1] - 2*self.x[i][j][k] +
#                      self.x[i][j][k-1])/(self.grid.h**2)
#        return result
#    
#    def _get_d2xdy_ijk(self,i,j,k):
#        if j == 0: #one sided derivative on the edge
#            result = (self.x[i][j+2][k] - 2*self.x[i][j+1][k] +
#                      self.x[i][j][k])/(self.grid.h**2)
#        elif j==self.grid.num_y-1: #one sided derivative on the edge
#            result = (self.x[i][j][k] - 2*self.x[i][j-1][k] +
#                      self.x[i][j-2][k])/(self.grid.h**2)
#        else: #two sided derivative elsewhere
#            result = (self.x[i][j+1][k] - 2*self.x[i][j][k] +
#                      self.x[i][j-1][k])/(self.grid.h**2)
#        return result
#        
#    def _get_derivative(self):
#        #initialize derivative in each direction
#        dxdz = np.zeros(shape=self.x.shape,dtype=complex)
#        dxdy = np.zeros(shape=self.x.shape,dtype=complex)
#        for i in range(self.m): #loop over each layer
#            for j in range(self.grid.num_y): #loop over each row
#                for k in range(self.grid.num_z): #loop over each column
#                    dxdz[i][j][k] = self._get_dxdz_ijk(i,j,k)
#                    dxdy[i][j][k] = self._get_dxdy_ijk(i,j,k)
#        return dxdz, dxdy
#    
#    def _get_dxdz_ijk(self,i,j,k):
#        if k == 0: #one sided derivative on the edge
#            result = (self.x[i][j][k+1] - self.x[i][j][k])/self.grid.h
#        elif k==self.grid.num_z-1: #one sided derivative on the edge
#            result = (self.x[i][j][k] - self.x[i][j][k-1])/self.grid.h
#        else: #two sided derivative elsewhere
#            result = (self.x[i][j][k+1] - self.x[i][j][k-1])/(2*self.grid.h)
#        return result
#    
#    def _get_dxdy_ijk(self,i,j,k):
#        if j == 0: #one sided derivative on the edge
#            result = (self.x[i][j+1][k] - self.x[i][j][k])/self.grid.h
#        elif j==self.grid.num_y-1: #one sided derivative on the edge
#            result = (self.x[i][j][k] - self.x[i][j-1][k])/self.grid.h
#        else: #two sided derivative elsewhere
#            result = (self.x[i][j+1][k] - self.x[i][j-1][k])/(2*self.grid.h)
#        return result
#    
#class Deconfinement_Solution(Field_Solution):
#    """
#    Deconfinement picture.
#    
#    Attributes
#    ----------------------------------------
#    grid (Grid) = a Grid object
#    x (array) = the solution array with complex data type and with shape
#                (m,grid.num_y,grid.num_z)
#    m (int) = the number of dimensions of the solution field
#    error (array) = the list of error
#    loop (int) = number of loops actually ran
#    """
#    def __init__(self,N,bound_side,bound_middle,bound_bottom,grid,x,error,loop,
#                 title):
#        self.N = N
#        self.bound_side = bound_side
#        self.bound_middle = bound_middle
#        self.bound_bottom = bound_bottom
#        Field_Solution.__init__(self,grid,x,error,loop,title)
#        
#    def print_attributes(self):
#        print()
#        print("Attributes:")
#        print("N = " + str(self.N))
#        print("bound_side =" + str(self.bound_side))
#        print("bound_middle =" + str(self.bound_middle))
#        print("bound_bottom =" + str(self.bound_bottom))
#        print("L = " + str(self.grid.L))
#        print("w = " + str(self.grid.w))
#        print("R = " + str(self.grid.R))
#        print("R/L = " + str(self.grid.R_fraction))
#        print("h = " + str(self.grid.h))
#        print("loop = " + str(self.loop))
#        print("error = " + str(self.error[-1]))
#        print("energy = " + str(self._energy))
#        print()
#        
#    def plot_imag_nth_component(self,n):
#        """
#        Plot the imaginary part of the nth component of the vector field.
#        
#        Input
#        -------------------------------------------
#        n (int) = the component of the vector field
#        """
#        plt.figure()
#        plt.pcolormesh(self.grid.zv,self.grid.yv,self.get_imag_nth_component(n))
#        plt.colorbar()
#        #the actual field is 1 larger due to counting
#        plt.title("$\sigma_{}$".format(n+1)) 
#        plt.show()
#        
#    def plot_real_nth_component(self,n):
#        """
#        Plot the real part of the nth component of the vector field.
#        
#        Input
#        -------------------------------------------
#        n (int) = the component of the vector field
#        """
#        plt.figure()
#        plt.pcolormesh(self.grid.zv,self.grid.yv,self.get_real_nth_component(n))
#        plt.colorbar()
#        plt.title("$\phi_{}$".format(n+1))
#        plt.show()
#
#    """
#    When computing the gradient energy density for the deconfinement solution,
#    the problem of the monodromy of sigma arises. We get around it by
#    ignoring the energy in the jump across verticle axes across monodromy.
#    This is achieved by modifying the derivative function that is
#    called while calculating gradient energy density.
#    """
#    def _get_dxdz_ijk(self,i,j,k):
#        if k == 0: #one sided derivative on the edge
#            result = (self.x[i][j][k+1] - self.x[i][j][k])/self.grid.h
#        elif k==self.grid.num_z-1: #one sided derivative on the edge
#            result = (self.x[i][j][k] - self.x[i][j][k-1])/self.grid.h
#        #monodromy
#        elif (k==self.grid.left_axis-1) or (k==self.grid.right_axis-1):
#            result = (self.x[i][j][k] - self.x[i][j][k-1])/self.grid.h
#        elif (k==self.grid.left_axis) or (k==self.grid.right_axis):
#            result = (self.x[i][j][k+1] - self.x[i][j][k])/self.grid.h
#        else: #two sided derivative elsewhere
#            result = (self.x[i][j][k+1] - self.x[i][j][k-1])/(2*self.grid.h)
#        return result
#    
#    def get_potential_energy_density(self):
#        W = Superpotential(self.N)
#        ped = (1/4)*W.dWdx_absolute_square_on_grid(self.x)
#        ped = np.real(ped) #it is real anyway
#        return ped
#               
#    def get_potential_energy(self):
#        return simps(simps(self.get_potential_energy_density(),self.grid.z),self.grid.y)
#    
#    def plot_potential_energy_density(self,save=False):
#        """
#        Plot the potential energy density.
#        """
#        plt.figure()
#        plt.pcolormesh(self.grid.zv,self.grid.yv,
#                       self.get_potential_energy_density(),cmap='jet')
#        plt.colorbar()
#        #plt.title("Potential Energy Density")
#        if save:
#            plt.savefig(self.title+" Potential Energy Density.png")
#        plt.show()
#
#    def get_energy_density(self):
#        return self.get_potential_energy_density() \
#               + self.get_gradient_energy_density()
#               
#    def get_energy(self):
#        return simps(simps(self.get_energy_density(),self.grid.z),self.grid.y)
#    
#    def plot_energy_density(self,save=False):
#        """
#        Plot the energy density.
#        """
#        plt.figure()
#        plt.pcolormesh(self.grid.zv,self.grid.yv,
#                       self.get_energy_density(), cmap='jet')
#        plt.colorbar()
#        #plt.title("Energy Density")
#        if save:
#            plt.savefig(self.title+" Energy Density.png")
#        plt.show()
#        
