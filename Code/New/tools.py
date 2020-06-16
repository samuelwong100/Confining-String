# -*- coding: utf-8 -*-
"""
File Name: .py
Purpose: 
Author: Samuel Wong
"""
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
Miscellaneous Math Functions
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

""" ============== subsection: Miscellaneous Math Functions =============="""

def grad(f, points, dx = 1e-5):
    """
    NAME:
        grad
    PURPOSE:
        Calculate the numerical value of gradient for an array of points, using
        a function that is able to take an array of points
    INPUT:
        f = a differentiable function that takes an array of m points, each
        with n dimensions, and returns an (m,1) array
        points = (m,n) array, representing m points, each with n dimensions
    OUTPUT:
        (m,n) array, each row being a gradient
    """
    n = np.shape(points)[1]
    increment = dx*np.identity(n)
    df = []
    for row in increment:
        df.append((f(points + row) - f(points-row))/(2*dx))
    return np.array(df).T[0]    

def derivative_sample(x,h):
    """
    return the derivative of a sample of a function, x, which can have multiple
    components (column), and the points are stored as rows.
    """
    #get the derivaitve and fix the boundary issues
    first = (x[1] - x[0])/h
    last = (x[-1] - x[-2])/h
    dxdz = (np.roll(x,-1,axis=0) - np.roll(x,1,axis=0))/(2*h)
    dxdz[0] = first
    dxdz[-1] = last
    return dxdz

"""
===============================================================================
                                    Field
===============================================================================

Subsections:
Grid
Vector Field
===============================================================================
"""
class Grid():
    def __init__(self,num_z,num_y,h=0.1,origin="center"):
        self._verify_num(num_z,num_y)
        self.num_z = num_z
        self.num_y = num_y
        self.h = h
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
    
    def z_position_to_z_number(self,z):
        #return the nearest z_number given a z_position; round down if necessary
        if z < self.z0 or z > self.zf:
            raise Exception("z-position out of bound.")
        elif z == self.z0:
            return 0
        elif z == self.zf:
            return self.num_z - 1
        else:
            #get the next index where z is between next index and previous index
            next_index = np.searchsorted(self.z_linspace,z)
            mid_value = (self.z_linspace[next_index] + self.z_linspace[next_index-1])/2
            if z == mid_value:
                z_number = next_index - 1 #need to round down to get axis right
            elif z < mid_value:
                z_number = next_index - 1
            else:
                z_number = next_index
            return z_number
        
    def y_position_to_y_number(self,y):
        #return the nearest y_number given a y_position; round down if necessary
        if y < self.y0 or y > self.yf:
            raise Exception("y-position out of bound.")
        elif y == self.y0:
            return 0
        elif y == self.yf:
            return self.num_y - 1
        else:
            #get the next index where z is between next index and previous index
            next_index = np.searchsorted(self.y_linspace,y)
            mid_value = (self.y_linspace[next_index] + self.y_linspace[next_index-1])/2
            if y == mid_value:
                y_number = next_index - 1
            elif y < mid_value:
                y_number = next_index - 1
            else:
                y_number = next_index
            return y_number
        
    def zy_position_to_zy_number(self,z,y):
        nz = self.z_position_to_z_number(z)
        ny = self.y_position_to_y_number(y)
        return (nz,ny)
    
    def get_nearest_position_on_grid(self,z,y):
        return self.zy_number_to_position(*self.zy_position_to_zy_number(z,y))
    
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
        self.half_grid = Dipole_Half_Grid(
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
    def __init__(self,num_z,num_y,num_R_half,h=0.1):
        Grid.__init__(self,num_z,num_y,h=0.1,origin="left")
        self._verify_num_R_half(num_R_half)
        self.num_R_half = num_R_half
        self.num_R_half_interval = self.num_R_half-1
        self.R_half = self.num_R_half_interval*self.h
        self.left_charge_axis_number = \
                        self.y_axis_number - self.num_R_half_interval
        self.left_charge_z_position = self.z_number_to_position(
                self.left_charge_axis_number)
        
    def _verify_num_R_half(self,num_R_half):
        if num_R_half >= self.num_z:
            raise Exception("num_R_half must be smaller than num_z.")


if __name__ == "__main__":
    DFG = Dipole_Full_Grid(101,101,51)
    DFG.plot_empty_grid()
    hg = DFG.half_grid
    hg.plot_empty_grid()
    

    
