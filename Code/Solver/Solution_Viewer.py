# -*- coding: utf-8 -*-
"""
File Name: Solution_Viewer.py
Purpose: Class for describing the field solution
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
sys.path.append("../BPS_Package")
import os
import numpy as np
import pickle
from scipy.integrate import simps
import matplotlib.pyplot as plt
from Math import Superpotential
from Grid import Standard_Dipole_Grid
from Relaxation import Relaxation
from Sigma_Critical import Sigma_Critical
from plot_BPS import plot_BPS as external_plot_BPS

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
            self.core_dict = core_dict
            self.N = core_dict["N"]
            self.x = core_dict["field"]
            self.m = self.x.shape[0]
            self.error = core_dict["error"]
            self.loop = core_dict["loop"]
            self.max_loop = core_dict["max_loop"]
            self.L = core_dict["L"]    
            self.w = core_dict["w"]
            self.h = core_dict["h"]
            self.R = core_dict["R"]
            self.grid = core_dict["grid"]
            self.bound_arg = core_dict["bound_arg"]
            self.charge_arg = core_dict["charge_arg"]
            self.folder_title = title
            self.x0 = core_dict["x0"]
            self.relax = core_dict["relax"]
            if self.x0 == "BPS":
                self.top_BPS = core_dict["BPS_top"]
                self.bottom_BPS = core_dict["BPS_bottom"]
                self.y_half = core_dict["BPS_y"]
                self.BPS_slice = core_dict["BPS_slice"]
                self.initial_grid = core_dict["initial_grid"]
                self.B_top = core_dict["B_top"]
                self.B_bottom = core_dict["B_bottom"]
        else:
            raise Exception("Solution file does not exist.")
            
    def display_all(self):
        self.print_attributes()
        self.plot_error()
        self.plot_x_all()
        self.plot_laplacian_all()
        self.plot_gradient_energy_density()
        self.plot_potential_energy_density()
        self.plot_energy_density()
        self.store_energy()
        if self.x0 == "BPS":
            self.plot_initial_grid()
            self.plot_BPS()
            self.compare_slice_with_BPS()
            
    def print_attributes(self):
        print()
        print("Attributes:")
        print("N = " + str(self.N))
        print("charge_arg = " + self.charge_arg)
        print("bound_arg = " + self.bound_arg)
        print("max loop = " + str(self.max_loop))
        print("L = " + str(self.grid.L))
        print("w = " + str(self.grid.w))
        print("h = " + str(self.grid.h))
        print("R = " + str(self.grid.R))
        print("loop = " + str(self.loop))
        print("error = " + str(self.error[-1]))
        print("energy = " + str(self.get_energy()))
        print()
            
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
                         "$\phi_{}$".format(str(n+1)), "phi_"+str(n+1))
        
    def plot_sigma_n(self,n):
        """
        Plot the imaginary part of the nth component of the vector field.
        
        Input
        -------------------------------------------
        n (int) = the component of the vector field
        """
        self._quick_plot(self.get_sigma_n(n),
                         "$\sigma_{}$".format(str(n+1)), "sigma_"+str(n+1))
        
    def plot_x_all(self):
        for n in range(self.m):
            self.plot_phi_n(n)
            self.plot_sigma_n(n)

    def plot_error(self):
        """
        Plot the error.
        """
        plt.figure()
        plt.plot(np.arange(0,self.loop,1),np.log(self.error))
        plt.ylabel("log(error)")
        plt.title("Error")
        plt.savefig(self.folder_title+"Error.png")
        plt.show()
        
    def get_laplacian(self):
        #initialize second derivative in each direction
        d2xdz = np.zeros(shape=self.x.shape,dtype=complex)
        d2xdy = np.zeros(shape=self.x.shape,dtype=complex)
        for i in range(self.m): #loop over each layer
            for j in range(self.grid.num_y): #loop over each row
                for k in range(self.grid.num_z): #loop over each column
                    d2xdz[i][j][k] = self._get_d2xdz_ijk(i,j,k)
                    d2xdy[i][j][k] = self._get_d2xdy_ijk(i,j,k)
        return d2xdz + d2xdy

    def plot_laplacian_all(self):
        """
        Plot and compare the numerical and theoretical Laplacian to verify
        that the solution actually solves the PDE
        """
        lap_num = self.get_laplacian()
        lap_theo = self._get_lap_theo()
        for n in range(self.m):
            self._plot_laplacian_n(n,lap_num,lap_theo)
    
    def get_gradient_energy_density(self):
        """
        Return the energy density from gradient of the field
        
        Output
        --------------------------------------------
        energy_density (array) = the energy density from gradient of the field;
                                 an array of shape (grid.num_y,grid.num_z).
        """
        dxdz,dxdy = self._get_derivative() #derivative in each direction
        dx_squared = np.abs(dxdz)**2 + np.abs(dxdy)**2 #square of the gradient
        gradient_energy_density = dx_squared.sum(axis=0) #sum over components
        return gradient_energy_density

    def get_gradient_energy(self):
        """
        Return the value of the gradient energy
        
        Output
        --------------------------------------------
        gradient_energy (float) = the total gradient energy
        """
        #integrate to get energy
        gradient_energy = simps(simps(self.get_gradient_energy_density(), 
                             self.grid.z),self.grid.y)
        return gradient_energy
    
    def plot_gradient_energy_density(self):
        self._quick_plot(self.get_gradient_energy_density(),
                         "Gradient Energy Density",
                         "Gradient_Energy_Density",
                         cmap='jet')

    def get_potential_energy_density(self):
        W = Superpotential(self.N)
        ped = (1/4)*W.dWdx_absolute_square_on_grid(self.x)
        ped = np.real(ped) #it is real anyway
        return ped
               
    def get_potential_energy(self):
        return simps(simps(self.get_potential_energy_density(),self.grid.z),
                     self.grid.y)
    
    def plot_potential_energy_density(self):
        self._quick_plot(self.get_potential_energy_density(),
                         "Potential Energy Density",
                         "Potential_Energy_Density",
                         cmap='jet')
        
    def get_energy_density(self):
        return self.get_potential_energy_density() \
               + self.get_gradient_energy_density()
               
    def get_energy(self):
        return simps(simps(self.get_energy_density(),self.grid.z),self.grid.y)
    
    def store_energy(self):
        with open(self.folder_title+"core_dict","wb") as file:
            self.core_dict["energy"] = self.get_energy()
            pickle.dump(self.core_dict, file)
    
    def plot_energy_density(self):
        self._quick_plot(self.get_energy_density(),
                         "Energy Density",
                         "Energy_Density",
                         cmap='jet')

    def plot_BPS(self):
        #the following only works if the boundary is x1
        external_plot_BPS(N=self.N,z=self.y_half,f=self.top_BPS,B=self.B_top,
                          h=self.h,folder=self.folder_title,
                          vac0=self.bound_arg,vacf="x0",save_plot=True)
        external_plot_BPS(N=self.N,z=self.y_half,f=self.bottom_BPS,
                          B=self.B_bottom,h=self.h,folder=self.folder_title,
                          vac0=self.charge_arg,vacf=self.bound_arg,
                          save_plot=True)
        
    def plot_initial_grid(self):
        for n in range(self.m):
            phi_n = np.real(self.initial_grid[n,:,:])
            sigma_n = np.imag(self.initial_grid[n,:,:])
            self._quick_plot(phi_n,"initial phi_{}".format(str(n+1)),
                             "initial_phi_{}".format(str(n+1)))
            self._quick_plot(sigma_n,"initial sigma_{}".format(str(n+1)),
                             "initial_sigma_{}".format(str(n+1)))
            
    def compare_slice_with_BPS(self):
        for n in range(self.m):
            plt.figure()
            #take a vertical slice through middle
            middle_col = int(self.grid.num_z/2)
            plt.plot(self.grid.y, self.get_phi_n(n)[:,middle_col],
                     label="final $\phi_{}$".format(str(n+1)))
            plt.plot(self.grid.y, np.real(self.BPS_slice[n,:]),
                     label="BPS $\phi_{}$".format(str(n+1)))
            plt.legend()
            plt.title("compare slice with BPS phi_{}".format(str(n+1)))
            plt.savefig(self.folder_title +
                        "compare_slice_with_BPS_phi_{}.png".format(str(n+1)))
            plt.show()
            
            plt.figure()
            plt.plot(self.grid.y, self.get_sigma_n(n)[:,middle_col],
                     label="final $\sigma_{}$".format(str(n+1)))
            plt.plot(self.grid.y, np.imag(self.BPS_slice[n,:]),
                     label="BPS $\sigma_{}$".format(str(n+1)))
            plt.legend()
            plt.title("compare slice with BPS sigma_{}".format(str(n+1)))
            plt.savefig(self.folder_title +
                    "compare_slice_with_BPS_sigma_{}.png".format(str(n+1)))
            plt.show()
            

    def _quick_plot(self,field,title,file_title,cmap=None):
        plt.figure()
        plt.pcolormesh(self.grid.zv,self.grid.yv,field,cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.savefig(self.folder_title+file_title+".png")
        plt.show()
        
    def _quick_plot_laplacian(self,field,ax,title,fig):
        im = ax.pcolormesh(self.grid.zv,self.grid.yv,field)
        ax.set_title(title)
        fig.colorbar(im,ax=ax)
        
    def _plot_laplacian_n(self,n,lap_num,lap_theo):
        #row= real & imag of fields; col= numeric vs theoretic
        fig, axs = plt.subplots(2, 2) 
        fig.subplots_adjust(hspace=0.7)
        fig.subplots_adjust(wspace=0.7)
        self._quick_plot_laplacian(np.real(lap_num[n,:,:]),axs[0, 0],
                        "$\\nabla^2 \phi_{}$ numeric".format(str(n+1)),
                                   fig)
        self._quick_plot_laplacian(np.real(lap_theo[n,:,:]),axs[0,1],
                        "$\\nabla^2 \phi_{}$ theoretic".format(str(n+1)),
                                   fig)
        self._quick_plot_laplacian(np.imag(lap_num[n,:,:]),axs[1, 0],
                        "$\\nabla^2 \sigma_{}$ numeric".format(str(n+1)),
                                   fig)
        self._quick_plot_laplacian(np.imag(lap_theo[n,:,:]),axs[1,1],
                        "$\\nabla^2 \sigma_{}$ theoretic".format(str(n+1)),
                            fig)
        #add axis label such that repeated are avoided
        #for ax in axs.flat:
            #ax.set(xlabel='z', ylabel='y')
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        #for ax in axs.flat:
            #ax.label_outer()
        fig.savefig(self.folder_title+"Laplacian_{}.png".format(str(n+1)))
            
    def _get_lap_theo(self):
        #return theoretical laplacian
        charge = Sigma_Critical(self.N,self.charge_arg)
        bound = Sigma_Critical(self.N,self.bound_arg)
        relax = Relaxation(self.grid,self.N,bound,charge,
                           self.max_loop,x0=None,diagnose=False)
        return relax._full_grid_EOM(self.x)
    
    def _get_derivative(self):
        #initialize derivative in each direction
        dxdz = np.zeros(shape=self.x.shape,dtype=complex)
        dxdy = np.zeros(shape=self.x.shape,dtype=complex)
        for i in range(self.m): #loop over each layer
            for j in range(self.grid.num_y): #loop over each row
                for k in range(self.grid.num_z): #loop over each column
                    dxdz[i][j][k] = self._get_dxdz_ijk(i,j,k)
                    dxdy[i][j][k] = self._get_dxdy_ijk(i,j,k)
        return dxdz, dxdy

    def _get_dxdz_ijk(self,i,j,k):
        if k == 0: #one sided derivative on the edge
            result = (self.x[i][j][k+1] - self.x[i][j][k])/self.grid.h
        elif k==self.grid.num_z-1: #one sided derivative on the edge
            result = (self.x[i][j][k] - self.x[i][j][k-1])/self.grid.h
        else: #two sided derivative elsewhere
            result = (self.x[i][j][k+1] - self.x[i][j][k-1])/(2*self.grid.h)
        return result
    
    def _get_dxdy_ijk(self,i,j,k):
        if j == 0: #one sided derivative on the edge
            result = (self.x[i][j+1][k] - self.x[i][j][k])/self.grid.h
        elif j==self.grid.num_y-1: #one sided derivative on the edge
            result = (self.x[i][j][k] - self.x[i][j-1][k])/self.grid.h
        #monodromy
        elif j==self.grid.z_axis-1 and self.grid.left_axis<= k <=self.grid.right_axis:
            result = (self.x[i][j][k] - self.x[i][j-1][k])/self.grid.h
        elif j==self.grid.z_axis and self.grid.left_axis<= k <=self.grid.right_axis:
            result = (self.x[i][j+1][k] - self.x[i][j][k])/self.grid.h
        else: #two sided derivative elsewhere
            result = (self.x[i][j+1][k] - self.x[i][j-1][k])/(2*self.grid.h)
        return result
    
    def _get_d2xdz_ijk(self,i,j,k):
        if k == 0: #one sided second derivative on the edge (forward difference)
            result = (self.x[i][j][k+2] - 2*self.x[i][j][k+1] +
                      self.x[i][j][k])/(self.grid.h**2)
        elif k==self.grid.num_z-1: #one sided second derivative on the edge
            result = (self.x[i][j][k] - 2*self.x[i][j][k-1] +
                      self.x[i][j][k-2])/(self.grid.h**2)
        else: #two sided second derivative elsewhere
            result = (self.x[i][j][k+1] - 2*self.x[i][j][k] +
                      self.x[i][j][k-1])/(self.grid.h**2)
        return result
    
    def _get_d2xdy_ijk(self,i,j,k):
        if j == 0: #one sided derivative on the edge
            result = (self.x[i][j+2][k] - 2*self.x[i][j+1][k] +
                      self.x[i][j][k])/(self.grid.h**2)
        elif j==self.grid.num_y-1: #one sided derivative on the edge
            result = (self.x[i][j][k] - 2*self.x[i][j-1][k] +
                      self.x[i][j-2][k])/(self.grid.h**2)
        else: #two sided derivative elsewhere
            result = (self.x[i][j+1][k] - 2*self.x[i][j][k] +
                      self.x[i][j-1][k])/(self.grid.h**2)
        return result
    