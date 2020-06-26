# -*- coding: utf-8 -*-
"""
File Name: .py
Purpose: 
Author: Samuel Wong
"""
import os
import pickle
import sys
sys.path.append("../BPS_Package")
from solve_BPS import solve_BPS
import numpy as np
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy

"""
===============================================================================
                                    Math
===============================================================================
Conventions
                        
1. The general complex field, x, is stored as a (m x n) array. Each of the m
rows is a new point. So if x is only storing the field at one point, then its
shape is [1,n], ie., x = array([[x_1, x_2, ..., x_n]]), where x_i are complex. 
Each of the n columns are the component of the fields. In SU(N), n = N-1.

2. One exception is when we are storing a few constant vectors in SU(N) class.
In this case, even though there can be n components to these vectors, and there
might be m of these at a time, if m = 1, we store it as array([c1,c2,...,cn]).
For example, SU.rho = array([rho_1, rho_2, ...., rho_n]), since there is only
one rho ever.

Subsections:
SU(N)
Superpotential
Sigma Space Critical Points
Miscellaneous Math
===============================================================================
"""

""" ============== subsection: SU(N) ======================================="""
def delta(i,j):
    #kronecker delta
    if i==j:
        return 1
    else:
        return 0


class SU():
    """
    A class representing an SU(N) gauge group. It is used to compute and store
    constants such as fundamental weights and weights of the fundamental 
    representation.
    
    Convention: sets of vectors are stored as rows of a matrix. To acess the
    first nu vector, for example, call SU.nu[0,:].
    """
    def __init__(self,N):
        self.N = N #rank of gauge group
        self.nu = self._define_nu() # weight of fundamental representation
        self.alpha = self._define_alpha() # simple roots and affine root
        self.w = self._define_w() # fundamental weights
        self.rho = self._define_rho() # Weyl's vector

    def _theta(self,a,A):
        # a=1,...,N-1
        # A=1,...,N
        if a<1 or a==self.N:
            raise Exception
        if a>=A:
            return 1
        elif a<A:
            return 0

    def _lambda(self,a,A):
        # the mathematical definition of lambda
        return (1/sqrt(a*(a+1)))*(self._theta(a,A) - a*delta(a+1,A))

    def _lambda_modified(self,A,a):
        # to use labmda to create matrix, first switch rows & columns, then
        # add 1 since python starts counting at 0 but lambda starts at 1
        return self._lambda(a+1,A+1)

    def _define_nu(self):
        # create a N by N-1 matrix that is defined by lambda
        # there are N nu vectors, each N-1 dimensional
        # the rows are the nu vectors
        nu = np.empty(shape=(self.N,self.N-1))
        for A in range(self.N):
            for a in range(self.N-1):
                nu[A][a] = self._lambda_modified(A,a)
        return nu

    def _define_alpha(self):
        #initialize an empty list
        ls=[]
        summation= 0
        # define the first N-1 alpha vectors
        for a in range(0,self.N-1):
            ls.append(self.nu[a,:] - self.nu[a+1,:])
            summation += self.nu[a,:] - self.nu[a+1,:]
        # the last alpha is the negative of the sum of the previous alpha
        ls.append(-1*summation)
        alpha = np.array(ls)
        return alpha

    def _define_w(self):
        # the i-th w is the sum of the first i nu
        # it is only defined up to the N-1th W
        # w is N-1 by N-1 matrix.
        w = np.zeros(shape=(self.N-1,self.N-1))
        summation = np.zeros(self.N-1)
        for A in range(0,self.N-1):
            # for every row of w, add up increasingly more nu vectors
            summation += self.nu[A,:]
            w[A,:] = deepcopy(summation)        
        return w

    def _define_rho(self):
        summation= np.zeros(self.N-1)
        # rho is a N-1 dimensional row vector
        for A in range(0,self.N-1):
            summation += self.w[A,:]
        return summation

""" ============== subsection: Superpotential ============================="""
def dot_vec_with_vec_field(vec,vec_field):
    #assume vec is an array of shape (n,)
    #vector field is an array of shape (n,x,y), where the field has 
    #n components, and lives on a x-y grid.
    #return the dot product the the vector and the field at every point
    #on the grid. The return is a (x,y) grid object
    return np.sum((vec*(vec_field.T)).T,axis=0)

class Superpotential():
    """
    Superpotential in SU(N).
    """
    def __init__(self,N):
        self.N = N
        self.s = SU(N)
        self.x_min = self._define_x_min() # minimum of superpotential

    def __call__(self,x):
        # x is an num by N-1 array, where num is the number of points
        # and N-1 is the number of complex fields,
        num = x.shape[0]
        result = np.zeros(shape=(num,1),dtype=complex)
        for row in range(num):
            summation = 0
            for i in range(0,self.N):
                summation += exp(np.dot(self.s.alpha[i,:],x[row,:]))
            result[row] = summation
        return result

    def _define_x_min(self):
        ls = []
        #there are N minimum vectors, each is 1 by N-1 dimensional
        for k in range(0,self.N+1):
            ls.append(1j*(2*pi/self.N)*k*self.s.rho)
        x_min = np.array(ls)
        return x_min
    
    def potential_term_on_grid_fast_optimized(self,x):
        #to optimize, absorb all for loop over grid into numpy vectorization
        #ok to loop over N, since they are small
        x_conj = np.conjugate(x)
        summation = np.zeros(shape=x.shape,dtype=complex)
        #loop over a first instead of b, so that terms like exp_B only
        #computes once
        #the following is the equation we get from applying the idenity of
        #alpha_a dot alpha_b in terms of 3 delta function
        for a in range(self.N):
            A = np.exp(dot_vec_with_vec_field(self.s.alpha[a],x))
            exp_B = np.exp(dot_vec_with_vec_field(self.s.alpha[a],x_conj))
            exp_C = np.exp(dot_vec_with_vec_field(self.s.alpha[(a-1)%self.N],x_conj))
            exp_D = np.exp(dot_vec_with_vec_field(self.s.alpha[(a+1)%self.N],x_conj))
            #the a-th term in the vector field summation
            vec_a = np.zeros(shape=x.shape,dtype=complex)
            for b in range(self.N-1):
                #the b-th component of the resulting vector field
                B = exp_B*self.s.alpha[a][b]
                C = exp_C*self.s.alpha[(a-1)%self.N][b]
                D = exp_D*self.s.alpha[(a+1)%self.N][b]
                vec_a[b,:,:] += 2*B-C-D
            summation += A*vec_a
        return summation/4
    
    def create_laplacian_function(self,DFG,charge_vec,use_half_grid):
        if use_half_grid:
            source_term = self._define_half_grid_source_term(DFG,charge_vec)
        else:
            source_term = self._define_full_grid_source_term(DFG,charge_vec)
            
        def laplacian_function(x):
            return source_term + self.potential_term_on_grid_fast_optimized(x)
        
        return laplacian_function
    
    def _define_full_grid_source_term(self,DFG,charge_vec):
        """ assumes charge vec contains 2pi factor, and is real """
        # return i 2pi C_a d(delta(y))/dy int_{-R/2}^{R/2} delta(z-z')dz'
        result = DFG.create_zeros_vector_field(self.N-1)
        #the derivative of delta function in y gives something close to infinity
        #for y just below 0 and somthing close to -infinity for y just above 0
        #here, we have x(y = 0^{-}) = 1/h^2 and x(y=0^{+})= -1/h^2
        #note that the lower row correspond to higher y
        inf = 1/(DFG.h_squared)
        result[:,DFG.z_axis_number-1,:] = -inf
        result[:,DFG.z_axis_number,:] = inf
        #set grid to 0 unless it is on z_axis and between -R/2 and R/2
        result[:,:,0:DFG.left_charge_axis_number]=0 #left_axis included in source
        result[:,:,DFG.right_charge_axis_number+1:]=0 #right_axis included in source
        #multiply everything by charge (outside relevant rows, everything is
        #zero anyway)
        for i in range(self.N-1):
            result[i,:,:] *= charge_vec[i]
        result = 1j*result # overall coefficient
        return result
    
    def _define_half_grid_source_term(self,DFG,charge_vec):
        full_grid_source_term = self._define_full_grid_source_term(DFG,charge_vec)
        half_grid = self.DFG.half_grid
        half_grid_source_term = half_grid.full_vector_field_to_half(
                full_grid_source_term)
        return half_grid_source_term

""" ============== subsection: Sigma_Critical ============================="""

class Sigma_Critical():
    """
    Full name: Sigma Space Crictical Points
    A class that handles the vector and string name of general ciritical points
    of superpotential. This includes the charge of quarks in the form of linear
    combinations of fundamental weights and field vaccua, which can be sum of 
    x_min vector with fundamental weights.
    
    Warning: There is a problem of convention. Technically, the sigma space is
    the imaginary component of the complex field, so sigma itself is real. When
    dealing with charge, we often talk about fundamental weights, which are 
    taken to be a real vector. But when we talk about x_min, it is usually in
    a context of it being the vacua of a full complex field, so it is often
    taken to be a purely imaginary vector. Since we want to have a class that
    deals with both generically, I will create two vectors, real and imaginary.
    They only differ by a factor of i.
    
    Variables
    ----------------------------------------
    N (int) = Number of colors
    arg (str) = a string that describes a linear combo of fundamental weights
                and/or x_min vectors;
                Must be in the form of:
               "wi +/-xj +/-wk ..."
               where the first letter of each term is either 'w' or 'x',
               followed by a integer (i,j,k), and it is either "+" or "-" for 
               each sign.
    vector (array) = an array of shape (N-1,)
    """
    def __init__(self,N,arg):
        self.N = N
        self.S = SU(N)
        self.W = Superpotential(N)
        self.name = arg
        self.real_vector = self._define_real_vector(arg)
        self.imaginary_vector = 1j*self.real_vector
            
    def __str__(self):
        return self.name
    
    def _define_real_vector(self,arg):
        str_list = arg.split() #split the string into list; separator is space
        return self._sum_up_terms(str_list)
            
    def _sum_up_terms(self, str_list):
        summation = np.zeros(shape=(self.N-1,),dtype=complex)
        for term in str_list:
            sign, term = self._get_term_sign(term)
            if len(term)==2:
                k = int(term[1])
                if term[0] == 'w':
                    #the usual w is already real
                    #the zeroth w corresponds to w1, need a shift
                    #the sigma space critical point is 2pi times w
                    summation += sign*2*pi*np.real(self.S.w[k-1,:])
                elif term[0] == 'x':
                    #the usual x_min is stored as purely imaginary
                    #want to convert to a real vector
                    #the zeroth x_min corresponds to x0, so no shift
                    #the x_min includes 2pi already
                    summation += sign*np.imag(self.W.x_min[k,:])
                else:
                    raise Exception("Unacceptable sigma crticial points \
                                    specification due to non x or w character")
            else:
                raise Exception("Unacceptable sigma crticial points \
                                    specification due to term too long.")
        return summation
        
    def _get_term_sign(self,term):
        sign = 1
        if term[0] == "+":
            term = term.replace("+","")
        elif term[0] == "-":
            sign = -1
            term = term.replace("-","")
        return sign, term
    
""" ============== subsection: Miscellaneous Math Functions ================"""
def within_epsilon(x,target):
    epsilon = 1e-5
    if isinstance(target,tuple):
        result = True
        for i in range(len(target)):
            result = result and target[i] - epsilon < x[i] < target[i] + epsilon
        return result
    elif isinstance(target,np.ndarray):
        result = True
        size= x.size
        x_compressed = x.reshape((size,))
        target_compressed = target.reshape((size,))
        for i in range(size):
            result = result and (target_compressed[i] - epsilon < 
                                 x_compressed[i] < target_compressed[i] + epsilon)
        return result
    else:
        return target - epsilon < x < target + epsilon

"""
===============================================================================
                                    Grid
===============================================================================
"""
class Grid():
    def __init__(self,num_z,num_y,h=0.1,origin="center"):
        self._verify_num(num_z,num_y)
        self.num_z = num_z
        self.num_y = num_y
        self.h = h
        self.h_squared = self.h**2
        #number of points in each half of grid
        self.num_z_half = int((self.num_z-1)/2)
        self.num_y_half = int((self.num_y-1)/2)
        self.z0,self.zf,self.y0,self.yf = self._get_boundary(origin)
        self.horizontal_length = self.zf - self.z0
        self.vertical_length = self.yf - self.y0
        self.z_linspace = np.linspace(self.z0,self.zf,self.num_z)
        self.y_linspace = np.linspace(self.y0,self.yf,self.num_y)
        self.zv, self.yv = np.meshgrid(self.z_linspace, self.y_linspace)
        self.z_axis_number, self.y_axis_number = self._get_axis_number()
    
    def _verify_num(self,num_z,num_y):
        if type(num_z) is not int:
            raise Exception("num_z must be integer.")
        if type(num_y) is not int:
            raise Exception("num_y must be integer.")
        if not isinstance(self,Dipole_Half_Grid): #half grid can have even numz
            if num_z % 2 == 0:
                raise Exception("num_z must be odd.")
        if num_y % 2 == 0:
            raise Exception("num_y must be odd.")
    
    def _get_boundary(self,origin):
        self.y0 = -self.num_y_half*self.h
        self.yf = self.num_y_half*self.h
        if origin == "center":
            self.z0 = -self.num_z_half*self.h
            self.zf = self.num_z_half*self.h
        elif origin == "left":
            self.z0 = -(self.num_z-1)*self.h #num_z-1 is number of interval
            self.zf = 0
        return self.z0,self.zf,self.y0,self.yf
    
    def _get_axis_number(self):
        #notice axis number is the 0 of opposite direction
        self.z_axis_number = self.y_position_to_y_number(0)
        self.y_axis_number = self.z_position_to_z_number(0)
        return self.z_axis_number, self.y_axis_number

    def z_number_to_position(self,nz):
        #convert the axis number along z-direction to position
        #n starts at 0
        return self.z_linspace[nz]
    
    def y_number_to_position(self,ny):
        #convert the axis number along y-direction to position
        #n starts at 0
        return self.y_linspace[ny]
    
    def zy_number_to_position(self,nz,ny):
        return (self.z_number_to_position(nz), self.y_number_to_position(ny))
    
    def _position_to_number(self,x,direction):
        #x can be either z or y
        #return the nearest x_number given a x_position; round down if necessary
        if direction == "z":
            linspace = self.z_linspace
            num = self.num_z
            x0 = self.z0
            xf = self.zf
        elif direction == "y":
            linspace = self.y_linspace
            num = self.num_y
            x0 = self.y0
            xf = self.yf    
        if x < x0 or x > xf:
            raise Exception("{}-position out of bound.".format(direction))
        elif x == x0:
            return 0
        elif x == xf:
            return num - 1
        else:
            #get the next index where x is between next index and previous index
            next_index = np.searchsorted(linspace,x)
            mid_value = (linspace[next_index] + linspace[next_index-1])/2
            if x == mid_value:
                x_number = next_index - 1 #need to round down to get axis right
            elif x < mid_value:
                x_number = next_index - 1
            else:
                x_number = next_index
            return x_number
    
    def z_position_to_z_number(self,z):
        return self._position_to_number(z,"z")
        
    def y_position_to_y_number(self,y):
        return self._position_to_number(y,"y")
        
    def zy_position_to_zy_number(self,z,y):
        nz = self.z_position_to_z_number(z)
        ny = self.y_position_to_y_number(y)
        return (nz,ny)
    
    def get_nearest_position_on_grid(self,z,y):
        return self.zy_number_to_position(*self.zy_position_to_zy_number(z,y))
    
    def create_zeros_vector_field(self,m,dtype=complex):
        #m is number of component of vector field
        return np.zeros(shape=(m,self.num_y,self.num_z),dtype=dtype)
    
    def create_ones_vector_field(self,m,dtype=complex):
        #m is number of component of vector field
        return np.ones(shape=(m,self.num_y,self.num_z),dtype=dtype)
    
    def create_constant_vector_field(self,vec):
        x = self.create_ones_vector_field(vec.size)
        for i in range(x.shape[0]):
            x[i,:,:] *= vec[i]
        return x
    
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
        for z in self.z_linspace:
            ax.axvline(z, lw=2, color='k', zorder=5)
        for y in self.y_linspace:
            ax.axhline(y, lw=2, color='k', zorder=5)
        #plot z and y axes in red
        ax.axvline(self.z_linspace[self.y_axis_number],color='r',lw=2,zorder=5)
        ax.axhline(self.y_linspace[self.z_axis_number],color='r',lw=2,zorder=5)
        if isinstance(self,Dipole_Full_Grid):
            #plot left and right axis if this is a dipole full grid
            ax.axvline(self.z_linspace[self.left_charge_axis_number],color='r',lw=2,zorder=5)
            ax.axvline(self.z_linspace[self.right_charge_axis_number],color='r',lw=2,zorder=5)
        if isinstance(self,Dipole_Half_Grid):
            ax.axvline(self.z_linspace[self.left_charge_axis_number],color='r',lw=2,zorder=5)
        # draw the boxes
        ax.imshow(f, interpolation='none', cmap=cmap, 
                  extent=[self.z0, self.zf,self.y0, self.yf],
                  zorder=0)
        if isinstance(self,Dipole_Half_Grid):
            fig.suptitle("Empty Half Grid",fontsize=20)
        else:
            fig.suptitle("Empty Grid",fontsize=20)
        

class Dipole_Full_Grid(Grid):
    def __init__(self,num_z,num_y,num_R,h=0.1):
        Grid.__init__(self,num_z,num_y,h,origin="center")
        #num_R is the number of vertical grid lines that are at or between
        #the charges, inclusive. Since 0 is always in the middle, R also
        #has to be odd
        self._verify_num_R(num_R)
        self.num_R = num_R
        self.num_R_interval = self.num_R-1
        self.R = self.num_R_interval*self.h #R is separation distance
        self.left_charge_axis_number = \
                        self.y_axis_number - int(self.num_R_interval/2)
        self.right_charge_axis_number = \
                        self.y_axis_number + int(self.num_R_interval/2)
        self.left_charge_z_position = self.z_number_to_position(
                self.left_charge_axis_number)
        self.right_charge_z_position = self.z_number_to_position(
                self.right_charge_axis_number)
        self.half_grid = Dipole_Half_Grid(self,
                num_z=self.num_z_half+1,
                num_y=self.num_y,
                num_R_half=int(self.num_R_interval/2)+1,
                h=self.h)
        
    def _verify_num_R(self,num_R):
        if num_R % 2 == 0:
            raise Exception("num_R must be odd, as it is the number of vertical\
                lines between and including the charges, and 0 is between.")
        if num_R >= self.num_z:
            raise Exception("num_R must be smaller than num_z.")
            
        
class Dipole_Half_Grid(Grid):
    def __init__(self,parent,num_z,num_y,num_R_half,h=0.1):
        Grid.__init__(self,num_z,num_y,h=0.1,origin="left")
        self._verify_num_R_half(num_R_half)
        self.num_R_half = num_R_half
        self.num_R_half_interval = self.num_R_half-1
        self.R_half = self.num_R_half_interval*self.h
        self.left_charge_axis_number = \
                        self.y_axis_number - self.num_R_half_interval
        self.left_charge_z_position = self.z_number_to_position(
                self.left_charge_axis_number)
        self.parent_grid = parent
        
    def _verify_num_R_half(self,num_R_half):
        if num_R_half >= self.num_z:
            raise Exception("num_R_half must be smaller than num_z.")
            
    def reflect_vector_field(self,x):
        #currently, x contains the central column plus everthing to the left
        #slice out everything to the left of central column
        x_left = x[:,:,:-1]
        #flip horionztally
        x_right = np.flip(x_left,axis=2)
        #join right with the original (including center column)
        return np.concatenate((x,x_right),axis=2)
    
    def full_vector_field_to_half(self,x):
        #slize out a vector field on full grid to just left half and central
        #column
        return x[:,:,:self.y_axis_number+1]

"""
===============================================================================
                                    Relaxation
===============================================================================
"""    
def relaxation_algorithm(x_initial,laplacian_function,full_grid,tol,
                         use_half_grid):
    """
    Without regard to whether we use half grid method, create an initial
    vector field on full grid, pass in along with the full grid and laplacian
    function. This relaxation algorithm converts the initial full field
    to half field, run it on half grid method if desired, and return the
    the full grid result.
    Note: x_initial needs to have correct boundary condition
    """
    if use_half_grid:
        half_grid = full_grid.half_grid
        x_initial_half = half_grid.full_vector_field_to_half(x_initial)
        x_half, error, loop = _relaxation_while_loop(x_initial_half,
                                                    laplacian_function,
                                                    half_grid,tol,
                                                    _relaxation_update_half_grid)
        x_full = half_grid.reflect_vector_field(x_half)
        return x_full, error, loop
    else:
        return _relaxation_while_loop(x_initial,laplacian_function,full_grid,tol,
                                   _relaxation_update_full_grid)

def _relaxation_while_loop(x,laplacian_function,grid,tol,update_function):
    error = [tol+1] #initialize a "fake" error so that while loop can run
    loop = 0
    while error[-1]>tol:
        x_new = update_function(x,laplacian_function,grid)
        error.append(_get_error(x_new,x))
        x = x_new
        loop += 1
        _diagnostic_plot(loop,error,grid,x)
    del error[0] #delete the first, fake error
    error = np.array(error) #change error into an array
    return x, error, loop

def _relaxation_update_full_grid(x_old,laplacian_function,grid):
    # replace each element of x_old with average of 4 neighboring points,
    # minus laplacian
    x = deepcopy(x_old) #keep the old field to compare for error later
    laplacian = laplacian_function(x)
    # we loop over each element in the field grid, skipping over the edge.
    # so we start loop at 1, avoiding 0. We ignore the last one by -1
    for row in range(1,grid.num_y-1):
        for col in range(1,grid.num_z-1):
            x[:,row,col] = (x[:,row-1,col] + x[:,row+1,col] +
                          x[:,row,col-1] + x[:,row,col+1]
                          - laplacian[:,row,col]*grid.h_squared)/4
    # since we skipped over the edge, the Dirichlet boundary is automatically
    #enforced. (ie full grid method)
    return x

def _relaxation_update_half_grid(x_old,laplacian_function,grid):
    x=_relaxation_update_full_grid(x_old,laplacian_function,grid)
    #set the last column equal to its neighboring column to maintain a
    #Neumann boundary condition. Note that the half grid has y axis on 
    #the right, ie this is a left half grid.
    x[:,:,-1] = x[:,:,-2]
    return x

def _get_error(x_new,x):
    return np.max(np.abs(x_new-x))/np.max(np.abs(x_new))

def _diagnostic_plot(loop,error,grid,x):
    if loop % 100 == 0:
        print("loop =",loop,"error =",error[-1])
        for i in range(x.shape[0]):
            plt.figure()
            plt.pcolormesh(grid.zv,grid.yv,np.real(x[i,:,:]))
            plt.colorbar()
            plt.title("$\phi$"+str(i+1))
            plt.show()

#            plt.figure()
#            plt.pcolormesh(self.grid.zv,self.grid.yv,np.imag(x[i,:,:]))
#            plt.colorbar()
#            plt.title("$\sigma$"+str(i+1))
#            plt.show()

"""
===============================================================================
                          Confining String Solver
===============================================================================
"""
def confining_string_solver(N,charge_arg,bound_arg,L,w,h,R,tol,initial_kw="BPS",
           use_half_grid=True,diagnose=False):
    title = get_title(N,charge,bound,L,w,h,R,tol,initial_kw,use_half_grid)
    path = get_path(title)
    num_z,num_y,num_R = canonical_length_num_conversion(L,w,R,h)
    DFG = Dipole_Full_Grid(num_z,num_y,num_R,h)
    charge = Sigma_Critical(N,charge_arg)
    bound = Sigma_Critical(N,bound_arg)
    W = Superpotential(N)
    laplacian_function = W.create_laplacian_function(DFG,charge.real_vector,
                                                     use_half_grid)
    x_initial = initialize_field(N,DFG,charge,bound,initial_kw)
    x, error, loop = relaxation_algorithm(x_initial,laplacian_function,
                                          DFG,tol,use_half_grid)

def get_title(N,charge,bound,L,w,h,R,tol,initial_kw,use_half_grid):
    title =\
    ('CS(N={},charge={},bound={},L={},w={},h={},R={},'+ \
    'tol={},initial_kw={},use_half_grid={})').format(str(N),charge,
    bound,str(L),str(w),str(h),str(R),str(tol),initial_kw,str(use_half_grid))
    return title

def get_path(title):
    path = "../Results/Solutions/"+title+"/"
    return path

def store_solution(relax,title,N,charge_arg,bound_arg,L,w,h,R,max_loop,x0):
    path = "../Results/Solutions/"+title+"/"
    #store the core result in a dictionary
    core_dict = {"N":N,"charge_arg":charge_arg,"bound_arg":bound_arg,"L":L,
                  "w":w,"h":h,"R":R,"max_loop":max_loop,"x0":x0,
                  "loop":relax.loop,"field":relax.x,
                  "error":relax.error,"grid":relax.grid,"relax":relax}
    if x0 == "BPS":
        core_dict["BPS_top"] = relax.top_BPS
        core_dict["BPS_bottom"] = relax.bottom_BPS
        core_dict["BPS_y"] = relax.y_half
        core_dict["BPS_slice"] = relax.BPS_slice
        core_dict["initial_grid"] = relax.initial_grid
        core_dict["B_top"] = relax.B_top #store BPS objects
        core_dict["B_bottom"] = relax.B_bottom
    #create directory for new folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+"core_dict","wb") as file:
        pickle.dump(core_dict, file)
    
def canonical_length_num_conversion(L,w,R,h):
    if isinstance(L,int) and isinstance(w,int) and isinstance(R,int) and h==0.1:
        num_z = int(L/h)+1 #this is guranteed to be odd if L is integer
        num_y = int(w/h)+1
        num_R = int(R/h)+1
    return num_z,num_y,num_R

def initialize_field(N,DFG,charge,bound,initial_kw):
    if initial_kw == "constant":
        x0 = DFG.create_constant_vector_field(bound.imaginary_vector)
    elif initial_kw == "zero":
        x0 = DFG.create_zeros_vector_field(N-1)
    elif initial_kw == "BPS":
        x0 = _BPS_initial_field(N,DFG,charge,bound)
    x0 = enforce_boundary(x0,DFG,bound.imaginary_vector)
    return x0

def enforce_boundary(x_old,grid,bound_vec):
    x = deepcopy(x_old)
    #set all the left sides to bound
    #take all component (first index); for each component, take all rows
    #(second index); take the first/left column (third index); set it to
    #bound
    bound_2d = np.array([bound_vec])
    x[:,:,0] = np.repeat(bound_2d,grid.num_y,axis=0).T
    #repeat for bounds in other directions
    x[:,:,-1] = np.repeat(bound_2d,grid.num_y,axis=0).T
    x[:,0,:] = np.repeat(bound_2d.T,grid.num_z,axis=1)
    x[:,-1,:] = np.repeat(bound_2d.T,grid.num_z,axis=1)
    return x
    
def _BPS_initial_field(N,DFG,charge,bound):
    if str(bound) == "x1":
        y_half,x_top_half,B_top, x_bottom_half,B_bottom = \
        _get_BPS_from_ordered_vacua(str(bound),"x0",str(charge),
                                    vacf_arg=str(bound))
    elif str(bound) == "x2" and str(charge) == "w1 -w2 +w3" and self.N==4:
        y_half,x_top_half,B_top, x_bottom_half,B_bottom = \
        _get_BPS_from_ordered_vacua(str(bound),"w2","w1+w3",str(bound))

    x_top_half_trans = x_top_half.T
    x_bottom_half_trans = x_bottom_half.T
    half_num = x_top_half_trans.shape[1]
    x_slice = np.zeros(shape=(N-1,DFG.num_y),dtype=complex)
    x_slice[:,-1-half_num:-1] = np.flip(x_top_half_trans,1)
    x_slice[:,0:half_num] = np.flip(x_bottom_half_trans,1)
    #first set x0 entirely equal to boundary values
    for i in range(N-1):
        x0[i,:,:] *= bound.imaginary_vector[i]
    #for the 2 columns between the two charges, set to x_slice
    for k in range(DFG.num_z):
        if DFG.left_charge_axis_number <= k <= DFG.right_charge_axis_number:
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

def _get_BPS_from_ordered_vacua(v1,v2,v3,v4):
    y_half,x_top_half,B_top = _call_BPS(
            top=True,vac0_arg=v1,vacf_arg=v2,N,DFG)
    y_half,x_bottom_half,B_bottom = _call_BPS(
            top=False,vac0_arg=v3,vacf_arg=v4,N,DFG)
    return y_half, x_top_half, B_top, x_bottom_half, B_bottom

def _call_BPS(top,vac0_arg,vacf_arg,N,DFG):
    return solve_BPS(N=N, separation_R=DFG.R, top=top, vac0_arg=vac0_arg,
                    vacf_arg=vacf_arg, z0=DFG.y0, zf=0, h=DFG.h, folder="",
                    tol=1e-5, save_plot=False)

if __name__ == "__main__":
    pass
    
