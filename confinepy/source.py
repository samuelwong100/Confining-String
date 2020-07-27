# -*- coding: utf-8 -*-
"""
File Name: Source.py
Purpose: 
Author: Samuel Wong
"""
import os
import pickle
from copy import deepcopy
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import simps, trapz
from scipy.optimize import curve_fit
from numba import jit
import sympy
from sympy import init_printing
from sympy.matrices import Matrix
import warnings
warnings.filterwarnings('ignore') #ignore numba warnings

#before script starts, check if running as a package from Confining-String folder
#if so, change path to confinepy folder
if os.path.basename(os.getcwd()) == "Confining-String":
    os.chdir(os.path.join(os.getcwd(),"confinepy"))

"""
===============================================================================
                                 General Helper Functions
===============================================================================
"""

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

def create_path(path):
    #create directory for new folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

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
numpy vectorize helpers
SU(N)
Superpotential
Sigma Space Critical Points
Miscellaneous Math
===============================================================================
"""
""" ============== subsection: numpy vectorize helpers 2D =============="""
@jit(nopython=False)
def constant_times_scalar_field_numba(constant,scalar_field):
    m,n = scalar_field.shape
    result = np.zeros(shape=(m,n),dtype=complex)
    for row in range(m):
        for col in range(n):
            result[row][col] = constant * scalar_field[row][col]
    return result

@jit(nopython=False)
def exponentiate_scalar_field_numba(scalar_field):
    m,n = scalar_field.shape
    result = np.zeros(shape=(m,n),dtype=complex)
    for row in range(m):
        for col in range(n):
            result[row][col] = np.exp(scalar_field[row][col])
    return result

def dot_vec_with_vec_field(vec,vec_field):
    #assume vec is an array of shape (n,)
    #vector field is an array of shape (n,x,y), where the field has 
    #n components, and lives on a x-y grid.
    #return the dot product the the vector and the field at every point
    #on the grid. The return is a (x,y) grid object
    return np.sum((vec*(vec_field.T)).T,axis=0)

@jit(nopython=False)
def dot_vec_with_vec_field_numba(vec,vec_field):
    #assume vec is an array of shape (n,)
    #vector field is an array of shape (n,x,y), where the field has 
    #n components, and lives on a x-y grid.
    #return the dot product the the vector and the field at every point
    #on the grid. The return is a (x,y) grid object
    comp,m,n = vec_field.shape
    summation = np.zeros(shape=(m,n),dtype=complex)
    for c in range(comp):
        for row in range(m):
            for col in range(n):
                summation[row][col] += vec[c]*vec_field[c][row][col]
    return summation

def scalar_field_times_vector(scalar_field,vector):
    #vector is of shape (components,)
    #scalar field is of shape (m,n)
    #return a vector field which is each component of vector multiplied by 
    #scalar field
    components = vector.shape[0]
    m,n = scalar_field.shape
    outer = np.outer(vector,scalar_field)
    return outer.reshape(components,m,n)

@jit(nopython=False)
def scalar_field_times_vector_field_numba(scalar_field,vec_field):
    #warning: it turns out this is slower than numpy multiply.
    comp,m,n = vec_field.shape
    result = np.zeros(shape=(comp,m,n),dtype=complex)
    for c in range(comp):
        for row in range(m):
            for col in range(n):
                result[c][row][col] = \
                scalar_field[row][col] * vec_field[c][row][col]
    return result

""" ============== subsection: numpy vectorize helpers 1D =============="""

def list_of_costants_times_vector(constants_ls,vector):
    #constants_ls is of shape (size,)
    #vector is of shape(n,)
    size=constants_ls.size
    return constants_ls.reshape(size,1)*vector

@jit(nopython=False)
def exponentiate_scalar_field_1D_numba(scalar_field):
    points = scalar_field.size
    result = np.zeros(shape=(points,),dtype=complex)
    for point in range(points):
            result[point] = np.exp(scalar_field[point])
    return result

@jit(nopython=False)
def dot_vec_with_vec_field_1D_numba(vec,vec_field):
    points, comps = vec_field.shape
    summation = np.zeros(shape=(points,),dtype=complex)
    for point in range(points):
        for comp in range(comps):
                summation[point] += vec[comp]*vec_field[point][comp]
    return summation

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
    def __init__(self,N,analytic=False):
        self.analytic = analytic # whether in analytic mode
        if self.analytic:
            init_printing()
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
        if not self.analytic:
            # the mathematical definition of lambda
            return (1/np.sqrt(a*(a+1)))*(self._theta(a,A) - a*delta(a+1,A))
        else:
            return (1/sympy.sqrt(a*(a+1)))*(self._theta(a,A) - a*delta(a+1,A))

    def _lambda_modified(self,A,a):
        # to use labmda to create matrix, first switch rows & columns, then
        # add 1 since python starts counting at 0 but lambda starts at 1
        return self._lambda(a+1,A+1)

    def _define_nu(self):
        if not self.analytic:
            # create a N by N-1 matrix that is defined by lambda
            # there are N nu vectors, each N-1 dimensional
            # the rows are the nu vectors
            nu = np.empty(shape=(self.N,self.N-1))
            for A in range(self.N):
                for a in range(self.N-1):
                    nu[A][a] = self._lambda_modified(A,a)
            return nu
        else:
            return Matrix(self.N,self.N-1,self._lambda_modified)

    def _define_alpha(self):
        if not self.analytic:
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
        else:
            #initialize an empty list
            ls=[]
            summation= Matrix.zeros(1,self.N-1)
            # define the first N-1 alpha vectors
            for A in range(0,self.N-1):
                ls.append(self.nu.row(A) - self.nu.row(A+1))
                summation += self.nu.row(A) - self.nu.row(A+1)
            # the last alpha is the negative of the sum of the previous alpha
            ls.append(-1*summation)
            alpha = Matrix(ls)
            return alpha

    def _define_w(self):
        if not self.analytic:
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
        else:
            # the i-th w is the sum of the first i nu
            # it is only defined up to the N-1th W
            # w is N-1 by N-1 matrix.
            ls=[]
            summation= Matrix.zeros(1,self.N-1)
            for A in range(0,self.N-1):
                # for every row of w, add up increasingly more nu vectors
                summation += self.nu.row(A)
                ls.append(summation)
            w = Matrix(ls)
            return w

    def _define_rho(self):
        if not self.analytic:
            summation= np.zeros(self.N-1)
            # rho is a N-1 dimensional row vector
            for A in range(0,self.N-1):
                summation += self.w[A,:]
            return summation
        else:
            summation= Matrix.zeros(1,self.N-1)
            # rho is a N-1 dimensional row vector
            for A in range(0,self.N-1):
                summation += self.w.row(A)
            return summation

""" ============== subsection: Superpotential ============================="""
class Superpotential():
    """
    Superpotential in SU(N).
    """
    def __init__(self,N):
        self.N = N
        self.N_minus_1 = self.N-1
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
                summation += np.exp(np.dot(self.s.alpha[i,:],x[row,:]))
            result[row] = summation
        return result

    def _define_x_min(self):
        ls = []
        #there are N minimum vectors, each is 1 by N-1 dimensional
        for k in range(0,self.N+1):
            ls.append(1j*(2*np.pi/self.N)*k*self.s.rho)
        x_min = np.array(ls)
        return x_min
    
    @jit(nopython=False)
    def potential_term_on_grid_numba(self,x):
        #to optimize, absorb all for loop over grid into numpy vectorization
        #ok to loop over N, since they are small
        x_conj = np.conjugate(x)
        summation = np.zeros(shape=x.shape,dtype=complex)
        #loop over a first instead of b, so that terms like exp_B only
        #computes once
        #the following is the equation we get from applying the idenity of
        #alpha_a dot alpha_b in terms of 3 delta function
        for a in range(self.N):
            a_minus_1 = (a-1) % self.N
            a_plus_1 = (a+1) % self.N
            A = exponentiate_scalar_field_numba(
                dot_vec_with_vec_field_numba(self.s.alpha[a],x))
            exp_B = exponentiate_scalar_field_numba(
                dot_vec_with_vec_field_numba(self.s.alpha[a],x_conj))
            exp_C = exponentiate_scalar_field_numba(
                dot_vec_with_vec_field_numba(self.s.alpha[a_minus_1],x_conj))
            exp_D = exponentiate_scalar_field_numba(
                dot_vec_with_vec_field_numba(self.s.alpha[a_plus_1],x_conj))
            #the a-th term in the vector field summation
            vec_a = np.zeros(shape=x.shape,dtype=complex)
            for b in range(self.N_minus_1):
                #the b-th component of the resulting vector field
                B = constant_times_scalar_field_numba(
                    self.s.alpha[a][b],exp_B)
                C = constant_times_scalar_field_numba(
                    self.s.alpha[a_minus_1][b],exp_C)
                D = constant_times_scalar_field_numba(
                    self.s.alpha[a_plus_1][b],exp_D)
                vec_a[b,:,:] += 2*B-C-D
            #summation += A*vec_a
            summation += scalar_field_times_vector_field_numba(A,vec_a)
        return summation/4
    
    # def potential_term_on_grid_vectorized(self,x):
    #     #to optimize, absorb all for loop over grid into numpy vectorization
    #     #ok to loop over N, since they are small
    #     x_conj = np.conjugate(x)
    #     summation = np.zeros(shape=x.shape,dtype=complex)
    #     #loop over a first instead of b, so that terms like exp_B only
    #     #computes once
    #     #the following is the equation we get from applying the idenity of
    #     #alpha_a dot alpha_b in terms of 3 delta function
    #     for a in range(self.N):
    #         a_minus_1 = (a-1) % self.N
    #         a_plus_1 = (a+1) % self.N
    #         A = np.exp(dot_vec_with_vec_field(self.s.alpha[a],x))
    #         exp_B = np.exp(dot_vec_with_vec_field(self.s.alpha[a],x_conj))
    #         exp_C = np.exp(dot_vec_with_vec_field(self.s.alpha[a_minus_1],x_conj))
    #         exp_D = np.exp(dot_vec_with_vec_field(self.s.alpha[a_plus_1],x_conj))
    #         #the a-th term in the vector field summation
    #         vec_a = np.zeros(shape=x.shape,dtype=complex)
    #         B = scalar_field_times_vector(exp_B,self.s.alpha[a])
    #         C = scalar_field_times_vector(exp_C,self.s.alpha[a_minus_1])
    #         D = scalar_field_times_vector(exp_D,self.s.alpha[a_plus_1])
    #         vec_a += 2*B-C-D
    #         summation += A*vec_a
    #     return summation/4
    
    def create_laplacian_function(self,DFG,charge_vec,use_half_grid):
        if use_half_grid:
            source_term = self._define_half_grid_source_term(DFG,charge_vec)
        else:
            source_term = self._define_full_grid_source_term(DFG,charge_vec)
            
        def laplacian_function(x):
            return source_term + self.potential_term_on_grid_numba(x)
        
        return laplacian_function
    
    def _define_full_grid_source_term(self,DFG,charge_vec):
        """ assumes charge vec contains 2pi factor, and is real """
        # return i 2pi C_a d(delta(y))/dy int_{-R/2}^{R/2} delta(z-z')dz'
        result = DFG.create_zeros_vector_field(self.N_minus_1)
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
        for i in range(self.N_minus_1):
            result[i,:,:] *= charge_vec[i]
        result = 1j*result # overall coefficient
        return result
    
    def _define_half_grid_source_term(self,DFG,charge_vec):
        full_grid_source_term = self._define_full_grid_source_term(DFG,charge_vec)
        half_grid = DFG.half_grid
        half_grid_source_term = half_grid.full_vector_field_to_half(
                full_grid_source_term)
        return half_grid_source_term
    
    def dWdx_absolute_square_on_grid(self,x):
        #use the formula for dot product of alpha in terms of 3 delta function
        x_conj = np.conjugate(x)
        m,n = (x.shape[1],x.shape[2])
        summation = np.zeros(shape=(m,n),dtype=complex)
        for a in range(self.N):
            exp_alpha_a_x = exponentiate_scalar_field_numba(
                dot_vec_with_vec_field_numba(self.s.alpha[a],x))
            exp_alpha_a_x_conj = exponentiate_scalar_field_numba(
                dot_vec_with_vec_field_numba(self.s.alpha[a],x_conj))
            exp_alpha_a_plus1_x_conj = exponentiate_scalar_field_numba(
                dot_vec_with_vec_field_numba(self.s.alpha[(a+1) % self.N],x_conj))
            exp_alpha_a_minus1_x_conj = exponentiate_scalar_field_numba(
                dot_vec_with_vec_field_numba(self.s.alpha[(a-1) % self.N],x_conj))
            summation += exp_alpha_a_x * (
                2*exp_alpha_a_x_conj
                - exp_alpha_a_plus1_x_conj
                - exp_alpha_a_minus1_x_conj)
        return summation
    
    def dWdx_absolute_square(self,x):
        #the same function as above but for BPS 1D field
        #use the formula for dot product of alpha in terms of 3 delta function
        x_conj = np.conjugate(x)
        points,comps = x.shape #shape of 1D field convention
        summation = np.zeros(shape=(points,),dtype=complex)
        for a in range(self.N):
            exp_alpha_a_x = exponentiate_scalar_field_1D_numba(
                dot_vec_with_vec_field_1D_numba(self.s.alpha[a],x))
            exp_alpha_a_x_conj = exponentiate_scalar_field_1D_numba(
                dot_vec_with_vec_field_1D_numba(self.s.alpha[a],x_conj))
            exp_alpha_a_plus1_x_conj = exponentiate_scalar_field_1D_numba(
                dot_vec_with_vec_field_1D_numba(self.s.alpha[(a+1) % self.N],x_conj))
            exp_alpha_a_minus1_x_conj = exponentiate_scalar_field_1D_numba(
                dot_vec_with_vec_field_1D_numba(self.s.alpha[(a-1) % self.N],x_conj))
            summation += exp_alpha_a_x * (
                2*exp_alpha_a_x_conj
                - exp_alpha_a_plus1_x_conj
                - exp_alpha_a_minus1_x_conj)
        return summation
    
    """" Note: do not delete below code of old implementation of absolute square
    They are used for testing against the new one."""
    # def dWdx_absolute_square_on_grid_old(self,x):
    #     """
    #     Absolute squared of gradient, evaluted on a grid.
        
    #     Input
    #     -------------------------------------------
    #     x (array) = a grid of shape (layers,n,m)
    
    #     Output
    #     --------------------------------------------
    #     dWdx (array) = a grid of shape (n,m)
    #     """
    #     layers,row_num,col_num = x.shape
    #     result = np.zeros(shape=(row_num,col_num),dtype=complex)
    #     for row in range(row_num):
    #         #take the row from each layer, transpose it such that the different fields
    #         #of the same point are in a row vector
    #         result[row,:] = self.dWdx_absolute_square(x[:,row,:].T)
    #     return result
    
    # def dWdx_absolute_square(self,x):
    #     """
    #     Absolute squared of gradient.
        
    #     Input
    #     -------------------------------------------
    #     x (array) = the value of the field at m points;an array of shape
    #                 (m,N-1); where m is number of points,
    #                 N-1 is number of field components
    
    #     Output
    #     --------------------------------------------
    #     dWdx (array) = an array of shape (m,)
    #     """
    #     dWdx = self.dWdx(x)
    #     dWdx_conj = np.conjugate(dWdx)
    #     return np.sum(dWdx*dWdx_conj,axis=1)
    
    # def dWdx(self,x):
    #     """
    #     First derivative vector, ie. gradient
        
    #     Input
    #     -------------------------------------------
    #     x (array) = the value of the field at m points;an array of shape
    #                 (m,N-1); where m is number of points,
    #                 N-1 is number of field components
    
    #     Output
    #     --------------------------------------------
    #     dWdx (array) = an array of shape (m,N-1), where each row is the
    #                     gradient vector of the corresponding point
    #     """
    #     m = x.shape[0]
    #     dWdx = np.zeros(shape=(m,self.N-1),dtype=complex)
    #     for i in range(m):
    #         #fix the ith point
    #         for j in range(self.N-1):
    #         #fix the jth partial derivative
    #             dWdx[i][j] = self._jth_parital_for_dWdx(x[i,:],j)
    #     return dWdx
    
    # def _jth_parital_for_dWdx(self,x_row,j):
    #     #intialize sum of exponential
    #     summation = 0
    #     for k in range(self.N):
    #         summation += exp(np.dot(self.s.alpha[k,:],x_row))*self.s.alpha[k][j]
    #     return summation

""" ============== subsection: Sigma_Critical ============================="""

class Sigma_Critical(): #TODO: make this compatible with N>10
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
                    summation += sign*2*np.pi*np.real(self.S.w[k-1,:])
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
def grad(f, points, dx = 1e-6):
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
        self.num_y_minus_1 = self.num_y-1
        self.num_z_minus_1 = self.num_z-1
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
def relaxation_algorithm(x_initial,laplacian_function,full_grid,sor,tol,
                         use_half_grid,diagnose):
    """
    Without regard to whether we use half grid method, create an initial
    vector field on full grid, pass in along with the full grid and laplacian
    function. This relaxation algorithm converts the initial full field
    to half field, run it on half grid method if desired, and return the
    the full grid result.
    Note: x_initial needs to have correct boundary condition
    """
    if not use_half_grid:
        # if using full grid, just call the normal relaxation while loop
        #with the full grid update method
        return _relaxation_while_loop(
            x_initial,laplacian_function,full_grid,sor,tol,
            _relaxation_update_full_grid_with_sor_numba,diagnose)
    else:
        #if using half grid, first create half grid object
        half_grid = full_grid.half_grid
        #fold the x_initial field in half
        x_initial_half = half_grid.full_vector_field_to_half(x_initial)
        #run the relaxation while loop, passing in the half grid update method
        x_half, error = _relaxation_while_loop(
            x_initial_half,laplacian_function,half_grid,sor,tol,
            _relaxation_update_half_grid_with_sor_numba,diagnose)
        #finally, unfold the half grid field by a reflection and return full
        #result
        x_full = half_grid.reflect_vector_field(x_half)
        return x_full, error
        

def _relaxation_while_loop(x,laplacian_function,grid,sor,tol,update_function,
                           diagnose):
    #wrote two versions, with/without diagnose, to avoid wasting time
    #checking if diagnose is true every loop
    if diagnose:
        return _relaxation_while_loop_with_diagnose(x,laplacian_function,grid,
                                                    sor,tol,update_function)
    else:
        return _relaxation_while_loop_without_diagnose(x,laplacian_function,grid,
                                                    sor,tol,update_function)

def _relaxation_while_loop_with_diagnose(x,laplacian_function,grid,sor,tol,
                                         update_function):
    error = [tol+1] #initialize a "fake" error so that while loop can run
    comp = x.shape[0] #number of components of fields
    one_minus_sor = 1-sor
    while error[-1]>tol:
        x_new = update_function(x,laplacian_function,sor,one_minus_sor,grid,comp)
        error.append(_get_error(x_new,x))
        x = x_new
        _diagnostic_plot(error,grid,x)
    del error[0] #delete the first, fake error
    error = np.array(error) #change error into an array
    return x, error

def _relaxation_while_loop_without_diagnose(x,laplacian_function,grid,sor,tol,
                                         update_function):
    error = [tol+1] #initialize a "fake" error so that while loop can run
    comp = x.shape[0] #number of components of fields
    one_minus_sor = 1-sor
    while error[-1]>tol:
        x_new = update_function(x,laplacian_function,sor,one_minus_sor,grid,comp)
        error.append(_get_error(x_new,x))
        x = x_new
    del error[0] #delete the first, fake error
    error = np.array(error) #change error into an array
    return x, error
    

@jit(nopython=False)
def _relaxation_update_full_grid_with_sor_numba(x_old,laplacian_function,sor,
                                                one_minus_sor,grid,comp):
    #despite despite contrary result in one loop test, numba version of 
    #update is WAY faster!
    # replace each element of x_old with average of 4 neighboring points,
    # minus laplacian
    x = deepcopy(x_old) #keep the old field to compare for error later
    laplacian = laplacian_function(x)
    # we loop over each element in the field grid, skipping over the edge.
    # so we start loop at 1, avoiding 0. We ignore the last one by -1
    for row in range(1,grid.num_y_minus_1):
        row_minus_1 = row-1
        row_plus_1 = row+1
        for col in range(1,grid.num_z_minus_1):
            col_minus_1 = col-1
            col_plus_1 = col+1
            for c in range(comp):
                x[c,row,col] = (x[c,row_minus_1,col] + x[c,row_plus_1,col] +
                              x[c,row,col_minus_1] + x[c,row,col_plus_1]
                              - laplacian[c,row,col]*grid.h_squared)/4
                x[c,row,col] = sor*x[c,row,col] + one_minus_sor*x_old[c,row,col]
    # since we skipped over the edge, the Dirichlet boundary is automatically
    #enforced. (ie full grid method)
    return x

def _relaxation_update_half_grid_with_sor_numba(x_old,laplacian_function,sor,
                                                one_minus_sor,grid,comp):
    x=_relaxation_update_full_grid_with_sor_numba(x_old,laplacian_function,sor,
                                                one_minus_sor,grid,comp)
    #set the last column equal to its neighboring column to maintain a
    #Neumann boundary condition. Note that the half grid has y axis on 
    #the right, ie this is a left half grid.
    x[:,:,-1] = x[:,:,-2]
    return x

# def _relaxation_update_full_grid(x_old,laplacian_function,grid,comp):
#     # replace each element of x_old with average of 4 neighboring points,
#     # minus laplacian
#     x = deepcopy(x_old) #keep the old field to compare for error later
#     laplacian = laplacian_function(x)
#     # we loop over each element in the field grid, skipping over the edge.
#     # so we start loop at 1, avoiding 0. We ignore the last one by -1
#     for row in range(1,grid.num_y_minus_1):
#         row_minus_1 = row-1
#         row_plus_1 = row+1
#         for col in range(1,grid.num_z_minus_1):
#             x[:,row,col] = (x[:,row_minus_1,col] + x[:,row_plus_1,col] +
#                               x[:,row,col-1] + x[:,row,col+1]
#                               - laplacian[:,row,col]*grid.h_squared)/4
#     # since we skipped over the edge, the Dirichlet boundary is automatically
#     #enforced. (ie full grid method)
#     return x

def _get_error(x_new,x):
    return np.max(np.abs(x_new-x))/np.max(np.abs(x_new))

def _diagnostic_plot(error,grid,x):
    loop = len(error)
    if loop % 100 == 0:
        print("relaxation 2D: loop =",loop,"error =",error[-1])
        for i in range(x.shape[0]):
            #plot phi_i
            plt.figure()
            plt.pcolormesh(grid.zv,grid.yv,np.real(x[i,:,:]))
            plt.colorbar()
            plt.title("$\phi$"+str(i+1))
            plt.show()
            #plot sigma_i
            plt.figure()
            plt.pcolormesh(grid.zv,grid.yv,np.imag(x[i,:,:]))
            plt.colorbar()
            plt.title("$\sigma$"+str(i+1))
            plt.show()

"""
===============================================================================
                          Confining String Solver
===============================================================================
"""
@timeit
def confining_string_solver(N,charge_arg,bound_arg,L,w,R,sor=0,h=0.1,tol=1e-9,
                            initial_kw="BPS",use_half_grid=True,
                            diagnose=False):
    #if no sor given, use default sor
    if sor==0:
        sor = best_sor_for_N(N)
    else:
        _validate_sor(sor)
    #create the title of the folder with all parameters in name
    title = get_title(N,charge_arg,bound_arg,L,w,h,R,sor,tol,initial_kw,
                      use_half_grid)
    #get the full path such that the folder is inside the confinement folder
    path = get_path(title)
    if os.path.exists(path): #if solution already exists, no need to resolve
        sol = Solution_Viewer(title)
    else:
        #assuming the input of L,w,R are integers and h=0.1, there is a
        #canonical way to convert them to a grid with odd number of points
        num_z,num_y,num_R = canonical_length_num_conversion(L,w,R,h)
        #regradless of whether we use halfgrid, create a full grid object first
        DFG = Dipole_Full_Grid(num_z,num_y,num_R,h)
        #create the two sigma space critical point objects associated with 
        #the charge of the quarks and the boundary vacuum
        charge = Sigma_Critical(N,charge_arg)
        bound = Sigma_Critical(N,bound_arg)
        #create superpotential object, then use it to create laplcian function
        #which takes into account of whether half grid is in use, since
        #the source term in laplacian depends on half vs full grid.
        W = Superpotential(N)
        laplacian_function = W.create_laplacian_function(
            DFG,charge.real_vector, use_half_grid)
        #intitialize field, with kw input that is default to BPS
        #need path to store the proper BPS result and plots
        x_initial = initialize_field(N,DFG,charge,bound,initial_kw,path)
        x, error = relaxation_algorithm(x_initial,laplacian_function,
                                        DFG,sor,tol,use_half_grid,
                                        diagnose)
        #store all the parameters and results that are not user-defined objects
        store_solution(path,N,x,x_initial,charge_arg,bound_arg,L,w,h,R,sor,tol,
                       initial_kw,use_half_grid,error)
        sol = Solution_Viewer(title)
    #sol.display_all()
    return sol

def best_sor_for_N(N):
    #need to test using tol=e-5
    #still need to test 5, 6 for 1.97
    sor_dict = {2:1.96,3:1.96,4:1.96,5:1.96,6:1.96,7:1.97} 
    if N in sor_dict:
        return sor_dict[N]
    else:
        return 1.96 # guess

def get_title(N,charge,bound,L,w,h,R,sor,tol,initial_kw,use_half_grid):
    title =\
    ('CS(N={},charge={},bound={},L={},w={},h={},R={},sor={},'+ \
    'tol={},initial_kw={},use_half_grid={})').format(str(N),charge,
    bound,str(L),str(w),str(h),str(R),str(sor),str(tol),initial_kw,
    str(use_half_grid))
    return title

def get_path(title):
    path = "Confinement Solutions/"+title+"/"
    return path
    
def canonical_length_num_conversion(L,w,R,h):
    if isinstance(L,int) and isinstance(w,int) and isinstance(R,int) and h==0.1:
        num_z = int(L/h)+1 #this is guranteed to be odd if L is integer
        num_y = int(w/h)+1
        num_R = int(R/h)+1
        return num_z,num_y,num_R
    else:
        raise Exception("length arguments not canonical.")

def store_solution(path,N,x,x_initial,charge_arg,bound_arg,L,w,h,R,sor,tol,
                   initial_kw,use_half_grid,error):
    #store the core result in a dictionary
    #notice we avoid self created object and only store arrays, numbers, and 
    #strings, and booleans
    core_dict = {"N":N,
                 "field":x,
                 "initial field":x_initial,
                 "charge_arg":charge_arg, "bound_arg":bound_arg,
                 "L":L,"w":w,"h":h,"R":R,
                 "sor":sor,
                 "tol":tol,
                 "initial_kw":initial_kw,
                 "use_half_grid":use_half_grid,
                 "error":error}
    #create directory for new folder if it doesn't exist
    create_path(path)
    with open(path+"core_dict","wb") as file:
        pickle.dump(core_dict, file)

def initialize_field(N,DFG,charge,bound,initial_kw,path):
    if initial_kw == "constant":
        x0 = DFG.create_constant_vector_field(bound.imaginary_vector)
    elif initial_kw == "zero":
        x0 = DFG.create_zeros_vector_field(N-1)
    elif initial_kw == "BPS":
        x0 = _BPS_initial_field(N,DFG,charge,bound,path)
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
    
def _BPS_initial_field(N,DFG,charge,bound,path):
    #solve for the two BPS equations
    if str(bound) == "x1":
        #canonically, the outside vacua is x1=rho/N, and the inside is x0,
        #and wk, for the 1-wall solution
        #other special configuration not using wk need initial conditions
        #on a case by case basis, like below.
        y_half, x_top_half, x_bottom_half = \
            _get_BPS_from_ordered_vacua(str(bound),"x0",str(charge),
                                        str(bound),N,DFG,path)
    elif str(bound) == "x2" and str(charge) == "w1 -w2 +w3" and N==4:
        y_half, x_top_half, x_bottom_half = \
            _get_BPS_from_ordered_vacua(str(bound),"w2","w1+w3",str(bound),N,
                                        DFG,path)
    #combine the two BPS equations into a single vertical slice
    x_slice = _get_double_BPS_slice(x_top_half,x_bottom_half,N,DFG)
    #initialize result
    x0 = DFG.create_constant_vector_field(bound.imaginary_vector)
    #for the 2 columns between the two charges, set to x_slice
    for k in range(DFG.num_z):
        if DFG.left_charge_axis_number <= k <= DFG.right_charge_axis_number:
            x0[:,:,k] = x_slice
    return x0

def _get_double_BPS_slice(x_top_half,x_bottom_half,N,DFG):
    x_top_half_trans = x_top_half.T
    x_bottom_half_trans = x_bottom_half.T
    half_num = x_top_half_trans.shape[1]
    x_slice = np.zeros(shape=(N-1,DFG.num_y),dtype=complex)
    x_slice[:,-1-half_num:-1] = np.flip(x_top_half_trans,1)
    x_slice[:,0:half_num] = np.flip(x_bottom_half_trans,1)
    return x_slice

def _get_BPS_from_ordered_vacua(v1,v2,v3,v4,N,DFG,path):
    x_top_half,y_half,error = _call_BPS(
            top=True,vac0_arg=v1,vacf_arg=v2,N=N,DFG=DFG,path=path)
    x_bottom_half, y_half, error = _call_BPS(
            top=False,vac0_arg=v3,vacf_arg=v4,N=N,DFG=DFG,path=path)
    #ignore the BPS error array
    return y_half, x_top_half, x_bottom_half

def _call_BPS(top,vac0_arg,vacf_arg,N,DFG,path):
    #call the usual solve BPS function, with the number of points equal to 
    #half of number of points in y (plus 1 to include the midpoint)
    #this point will get replaced by one of the 2 BPS upon overlap, doesn't
    #reallly matter as this is just initial condition
    print("BPS call:")
    print("N=",N)
    print("vac0_arg",vac0_arg)
    print("vacf_arg",vacf_arg)
    print("num =",DFG.num_y_half+1)
    print("h =",DFG.h)
    print("sor =",1.5)
    print("tol=",1e-9)
    #for now, call BPS for initial condition purpose with only tol=1e-5
    return solve_BPS(N=N,vac0_arg=vac0_arg,vacf_arg=vacf_arg,
                     num=DFG.num_y_half+1,h=DFG.h,tol=1e-5,sor=1.5,
                     kw="kink with predicted width",plot=True,save_plot=True,
                     save_result=True,folder=path,separation_R=DFG.R,top=top)

"""
===============================================================================
                                    BPS
===============================================================================
"""
""" ============== subsection: solve BPS ==================================="""
@timeit
def solve_BPS(N,vac0_arg,vacf_arg,num,h=0.1,tol=1e-9,sor=1.5,plot=True,
              save_plot=True,save_result=True,folder="",
              separation_R=0,top=True,kw="special kink",kink_bd_distance=None,
              continue_kw="BPS_Energy",x0_given=None):
    #sor = successive overrelaxation parameter. For h=0.1 in 2D, the ideal
    #value is given by approximately 1.5. For 1D, it's different; just a guess.
    #create sigma cirtical point objects for iniitial and final bondary
    vac0 = Sigma_Critical(N,vac0_arg)
    vacf = Sigma_Critical(N,vacf_arg)
    #create linspace for physical space, z
    z0,zf,z_linspace = get_z_linspace(num,h)
    #initialize field using speical kink
    x0 = set_x0(vac0.imaginary_vector,vacf.imaginary_vector,num,N-1,z0,zf,
                 kink_bd_distance,separation_R,top,kw=kw,x0_given=x0_given)
    #generate second derivative function for relaxation
    BPS_second_derivative_function = generate_BPS_second_derivative_function(N)
    #generate continue condition that checks energy as well as error for
    #whether to continue relaxation loop
    if continue_kw == "BPS_Energy":
        continue_condition = generate_BPS_energy_continue_condition(
                N,num,vac0,vacf,z_linspace,h)
    elif continue_kw == "default":
        continue_condition = default_continue_condition
    #call everything in relaxation algorithm
    x, z_linspace, error = relaxation_1D_algorithm(
            g=BPS_second_derivative_function,num=num,f0=x0,h=h,tol=tol,sor=sor,
            continue_condition=continue_condition)
    #if folder is unspecified, save it in designated BPS solitons folder
    if folder == "":
        folder = "BPS Solitons/N={},vac0={},vacf={},num={},h={},tol={},sor={},final_error={}/".format(
                str(N),vac0_arg,vacf_arg,str(num),str(h),str(tol),str(sor),
                '{:0.1e}'.format(error[-1]))
    if plot:
        plot_BPS(N,z_linspace,x,num,h,vac0,vacf,save_plot,folder)
        plot_error(error,folder)
    if save_result:
        BPS_dict={"x":x,"z":z_linspace,"vac0":str(vac0),"vacf":str(vacf),
                  "error":error}
        create_path(folder)
        with open(folder+"BPS_dict","wb") as file:
            pickle.dump(BPS_dict, file)
    return x, z_linspace, error

def open_BPS_result(folder):
    pickle_in = open("BPS Solitons/"+folder+"/BPS_dict","rb")
    BPS_dict = pickle.load(pickle_in)
    x = BPS_dict["x"]
    z = BPS_dict["z"]
    error = BPS_dict["error"]
    return x,z,error

def get_z_linspace(num,h):
    #create z_linspace for future plotting
    zf = (num-1)*h/2
    z0 = -zf
    z_linspace = np.linspace(start=z0,stop=zf,num=num)
    return z0,zf,z_linspace

def generate_BPS_second_derivative_function(N):
    S = SU(N)
    def BPS_second_derivative_function(x):
        #returns the second derivative function:
        #(1/4) Sum_{a=1}^{N} e^{alpha.x} (2e^{alpha[a].x*}alpha[a] 
        # - e^{alpha[a-1].x*}alpha[a-1] - e^{alpha[a+1].x*}alpha[a+1])
        exp_alpha_x = np.exp(dot_field_with_all_alpha(S.alpha,x))
        exp_alpha_x_conj = np.exp(dot_field_with_all_alpha(S.alpha,np.conj(x)))
        summation = 0
        for a in range(0,N):
            a_minus_1 = (a-1) % N
            a_plus_1 = (a+1) % N
            a_term = list_of_costants_times_vector(
                    exp_alpha_x_conj[:,a], S.alpha[a])
            a_minus_1_term = list_of_costants_times_vector(
                    exp_alpha_x_conj[:,a_minus_1],S.alpha[a_minus_1])
            a_plus_1_term = list_of_costants_times_vector(
                    exp_alpha_x_conj[:,a_plus_1],S.alpha[a_plus_1])
            summation += list_of_costants_times_vector(exp_alpha_x[:,a],
                                (2*a_term - a_minus_1_term - a_plus_1_term))
        return summation/4
    return BPS_second_derivative_function

def dot_field_with_all_alpha(alpha,x):
    #take a vector field of shape (num,N-1), dot it with each of alpha[a],
    #where a goes from 1 to N
    #return result of shape (num,N), each row is a point, each column is a 
    #result with corresponding alpha
    return np.dot(alpha, x.T).T

#Warning: for some reasons the numba version of this is very slow.
# def numba_generate_BPS_second_derivative_function(N):
#     S = SU(N)
#     @jit(nopython=False)
#     def BPS_second_derivative_function(x):
#         #returns the second derivative function:
#         #(1/4) Sum_{a=1}^{N} e^{alpha.x} (2e^{alpha[a].x*}alpha[a] 
#         # - e^{alpha[a-1].x*}alpha[a-1] - e^{alpha[a+1].x*}alpha[a+1])
#         summation = np.zeros(shape=x.shape,dtype=complex)
#         x_conj = np.conj(x)
#         for a in range(0,N): #loop over alpha componenets
#             a_minus_1 = (a-1) % N
#             a_plus_1 = (a+1) % N
#             for i in range(x.shape[0]): #loop over points
#                 exp_alpha_x = exp_alpha_dot_x(a,i,S,x)
#                 exp_alpha_x_conj = exp_alpha_dot_x(a,i,S,x_conj)
#                 exp_alpha_x_conj_a_minus_1 = exp_alpha_dot_x(a_minus_1,i,S,x_conj)
#                 exp_alpha_x_conj_a_plus_1 = exp_alpha_dot_x(a_plus_1,i,S,x_conj)
#                 for j in range(x.shape[1]): #loop over field components
#                     a_term = exp_alpha_x_conj*S.alpha[a][j]
#                     a_minus_1_term = exp_alpha_x_conj_a_minus_1*S.alpha[a_minus_1][j]
#                     a_plus_1_term = exp_alpha_x_conj_a_plus_1*S.alpha[a_plus_1][j]
#                     summation[i][j] += exp_alpha_x * (
#                             2*a_term - a_minus_1_term - a_plus_1_term)
#         return summation/4
#     return BPS_second_derivative_function

def exp_alpha_dot_x(a,i,S,x):
    return np.exp(np.dot(S.alpha[a],x[i]))

def set_x0(vac0_vec,vacf_vec,num,m,z0,zf,kink_bd_distance,R=0,top=True,
            kw=None,x0_given=None):
    #m is number of fields, m = N-1
    if kw is None:
        x0 = np.ones(shape=(num,m),dtype=complex)
    elif kw == "special kink":
        x0 = np.ones(shape=(num,m),dtype=complex)
        half = num // 2
        x0[0:half,:] = vac0_vec
        x0[half:-1,:] = vacf_vec
    elif kw == "special kink customized center":
        x0 = np.ones(shape=(num,m),dtype=complex)
        ratio = kink_bd_distance/np.abs(zf - z0)
        if top:
            kink_pixel_number = int((1-ratio)*num)
        else:
            kink_pixel_number = int(ratio*num)
        x0[0:kink_pixel_number,:] = vac0_vec
        x0[kink_pixel_number:-1,:] = vacf_vec
    elif kw == "kink with predicted width":
        x0 = np.ones(shape=(num,m),dtype=complex)
        #predict the width to be d/2 = y(R/2) = ln(R/2+1)-R/(R+2)
        height = np.log(R/2+1)-R/(R+2) #equivalent to d/2
        ratio = height/np.abs(zf - z0)
        if top:
            kink_pixel_number = int((1-ratio)*num)
        else:
            kink_pixel_number = int(ratio*num)
        print("kpn = ",kink_pixel_number)
        print("num=",num)
        x0[0:kink_pixel_number,:] = vac0_vec
        x0[kink_pixel_number:-1,:] = vacf_vec
        
    if not kw == "x0_given":
        #always enforce boundary unless x0 is given
        x0[0]= vac0_vec
        x0[-1]= vacf_vec
    
    if kw == "x0_given":
        x0 = x0_given
    return x0

def generate_BPS_energy_continue_condition(N,num,vac0,vacf,z,h):
    def BPS_energy_continue_condition(error,tol,x):
        greater_than_tol = default_continue_condition(error,tol)
        loop = len(error)
        if loop > 10000 and loop % 1000 ==0:
            print("Relaxation 1D: loop=",loop,"error=",error[-1])
            #after 10,000 loops, check energy condition once every 1000 loops
            #this ensures we don't stop prematurely and don't waste time
            #computing energy too foten
            theoretic_energy, numeric_energy = BPS_Energy(N,num,vac0,vacf,x,z,h)
            energy_too_far = (np.abs(theoretic_energy - numeric_energy) > 0.005)
            #only continue loop if error is too large and energy too far from
            #theoretical energy
            return greater_than_tol and energy_too_far
        else:
            return greater_than_tol
    return BPS_energy_continue_condition
        
""" ============== subsection: relaxation 1D ==============================="""
def default_continue_condition(error,tol,*args):
    return error[-1]>tol

def relaxation_1D_algorithm(g,num,f0,h=0.1,tol=1e-9,sor=1.5,
                            continue_condition = default_continue_condition):
    """
    Solve the boundary value problem of a set of coupled, second order,
    ordinary differential equation with m components using relaxation method.
    
    The ODE has to be of the form
    d^2(f(z))/dz^2 = g(f(z))
    where f can be a vector.
    
    f0 is the initial field, which is assumed to have right boundary.
    """
    h_squared = h**2 #precalculate to save time
    z0,zf,z_linspace = get_z_linspace(num,h)
    if sor==1: #no successive over relaxation
        f, error = relaxation_1D_while_loop(g,f0,num,h_squared,tol,
                                            continue_condition) #run relaxation
    else:
        _validate_sor(sor)
        f, error = relaxation_1D_while_loop_with_sor(g,f0,num,h_squared,tol,
                                                     sor,continue_condition)
    return f, z_linspace, error

def relaxation_1D_while_loop(g,f,num,h_squared,tol,
                             continue_condition = default_continue_condition):
    error = [tol+1] #initialize a "fake" error so that while loop can run
    while continue_condition(error,tol,f):
        f_new = _realxation_1D_update(g,f,num,h_squared)
        new_error = np.max(np.abs(f_new - f))/np.max(np.abs(f_new))
        print(new_error)
        error.append(new_error)
        f = f_new
    del error[0] #delete the first, fake error
    error = np.array(error) #change error into an array
    return f, error

@jit(nopython=False)
def _realxation_1D_update(g,f_old,num,h_squared):
    # replace each element of f_old with sum of left and right neighbors,
    # plus a second derivative term
    f_new = deepcopy(f_old)
    second_derivative = g(f_new)
    for k in range(1,num-1): #skip boundaries
        #note f is of shape (num,m), where num is number of points,
        #m is number of components
        f_new[k] = (f_new[k-1] + f_new[k+1] - second_derivative[k]*h_squared)/2
    return f_new

def _validate_sor(sor):
    if not 1<sor<2:
        raise Exception("sor parameter must be between 1 and 2.")

def relaxation_1D_while_loop_with_sor(g,f,num,h_squared,tol,sor,
                             continue_condition = default_continue_condition):
    error = [tol+1] #initialize a "fake" error so that while loop can run
    one_minus_sor = 1-sor
    while continue_condition(error,tol,f):
        f_new = _realxation_1D_update_with_sor(g,f,num,h_squared,sor,
                                               one_minus_sor)
        error.append(np.max(np.abs(f_new - f))/np.max(np.abs(f_new)))
        f = f_new
    del error[0] #delete the first, fake error
    error = np.array(error) #change error into an array
    return f, error

@jit(nopython=False)
def _realxation_1D_update_with_sor(g,f_old,num,h_squared,sor,one_minus_sor):
    # replace each element of f_old with sum of left and right neighbors,
    # plus a second derivative term
    f_new = deepcopy(f_old)
    second_derivative = g(f_new)
    for k in range(1,num-1): #skip boundaries
        #note f is of shape (num,m), where num is number of points,
        #m is number of components
        f_new[k] = (f_new[k-1] + f_new[k+1] - second_derivative[k]*h_squared)/2
        #sor is the successive overrelaxation parameter, between 1 and 2
        f_new[k] = sor*f_new[k] + one_minus_sor*f_old[k]
    return f_new

""" ============== subsection: plot BPS ===================================="""
def plot_BPS(N,z,f,num,h,vac0,vacf,save_plot,folder):
    phi = []
    sigma = []
    for i in range(N-1):
        phi.append(np.real(f[:,i]))
        sigma.append(np.imag(f[:,i]))
    
    #get the theoretical derivative for comparison
    dx_theoretic = BPS_dx(N,f,vac0,vacf)
    dphi_theoretic = []
    dsigma_theoretic = []
    for i in range(N-1):
        dphi_theoretic.append(np.real(dx_theoretic[:,i]))
        dsigma_theoretic.append(np.imag(dx_theoretic[:,i]))
        
    #get numerical derivative
    dx_numeric = derivative_sample(f,h)
    dphi_numeric = []
    dsigma_numeric = []
    for i in range(N-1):
        dphi_numeric.append(np.real(dx_numeric[:,i]))
        dsigma_numeric.append(np.imag(dx_numeric[:,i]))
    
    fig = plt.figure(figsize=(17,10))
    ax1 = fig.add_subplot(221)
    for i in range(N-1):
        ax1.plot(z,phi[i],label="$\phi_{}$".format(str(i+1)))
    ax1.legend()
    ax1.legend(loc=(1.05, 0))
    
    ax2 = fig.add_subplot(222)
    for i in range(N-1):
        ax2.plot(z,sigma[i],label="$\sigma_{}$".format(str(i+1)))
    ax2.legend()
    ax2.legend(loc=(1.05, 0))
    
    ax3 = fig.add_subplot(223)
    for i in range(N-1):
        color=next(ax3._get_lines.prop_cycler)['color']
        ax3.plot(z, dphi_theoretic[i], '--',
                 label="$d\phi_{} theoretic$".format(str(i+1)),color=color)
        ax3.plot(z, dphi_numeric[i],
                 label="$d\phi_{} numeric$".format(str(i+1)),color=color)
    ax3.legend()
    ax3.legend(loc=(1.05, 0))
    
    ax4 = fig.add_subplot(224)
    for i in range(N-1):
        color=next(ax4._get_lines.prop_cycler)['color']
        ax4.plot(z, dsigma_theoretic[i], '--',
                 label="$d\sigma_{} theoretic$".format(str(i+1)),color=color)
        ax4.plot(z, dsigma_numeric[i],
                 label="$d\sigma_{} numeric$".format(str(i+1)),color=color)
    ax4.legend()
    ax4.legend(loc=(1.05,0))
    
    fig.subplots_adjust(wspace=0.7)
    
    #get and print energy
    theoretic_energy,numeric_energy = BPS_Energy(N,num,vac0,vacf,f,z,h)
    fig.text(x=0,y=0.05,s= r"$E_{theoretic}$= "+str(round(theoretic_energy,4))+
             "; $E_{numeric}$= "+str(round(numeric_energy,4)),size=16)
    
    title="BPS (N={}, {} to {})".format(str(N),str(vac0),(vacf))
    fig.suptitle(title,size=30)
    if save_plot:
        create_path(folder)
        fig.savefig(folder+title+".png", dpi=300)

def BPS_dx(N,x,vac0,vacf):
    """
    BPS equations
    """
    W = Superpotential(N)
    x_ = np.conj(x)
    dWdx_ = grad(W,x_)
    numerator = W(np.array([vacf.imaginary_vector])) - \
                W(np.array([vac0.imaginary_vector]))
    denominator = np.absolute(numerator)
    alpha = numerator/denominator
    return (alpha/2)*dWdx_

def BPS_Energy(N,num,vac0,vacf,x,z,h):
    theoretic_energy = get_BPS_theoretic_energy(N,vac0,vacf)
    numeric_energy = get_BPS_numeric_energy(N,x,z,h)
    return (theoretic_energy,numeric_energy)

def get_BPS_theoretic_energy(N,vac0,vacf):
    W=Superpotential(N)    
    theoretic_energy = np.abs(W(np.array([vacf.imaginary_vector])) -\
                              W(np.array([vac0.imaginary_vector])))[0][0]
    return theoretic_energy

def get_BPS_numeric_energy(N,x,z,h):
    W=Superpotential(N)    
    #get the derivaitve of field
    dxdz = derivative_sample(x,h)
    # initialize first term
    kin_sum = 0
    dxdz_absolute_square = np.abs(dxdz**2)
    for i in range(N-1):
        # integrate the absolute square of each complex field and sum them up
        kin_sum += trapz(dxdz_absolute_square[:,i],z)
    # initialize second term
    dWdx_absolute_square = np.real(W.dWdx_absolute_square(x))
    int_dWdx_absolute_square = trapz(dWdx_absolute_square,z)
    numeric_energy = kin_sum + int_dWdx_absolute_square/4
    return numeric_energy

def plot_error(error,folder):
    """
    Plot the error.
    """
    plt.figure()
    loops = np.arange(0,error.size,1)
    plt.plot(loops,np.log10(error))
    plt.ylabel("log(error)")
    plt.title("BPS Error")
    plt.savefig(folder+"Error.png")
    plt.show()

"""
===============================================================================
                                Solution Viewer
===============================================================================
"""    
class Solution_Viewer():
    """
    Analyzing and displaying the field solution.
    """
    def __init__(self,title):
        #get the path and file name from title (this makes it easy to call
        #the solution by just copy and paste the name of the folder)
        path = get_path(title)
        file_name = path+"core_dict"
        if os.path.exists(file_name):
            #read the core dictionary from pickle
            pickle_in = open(file_name,"rb")
            core_dict = pickle.load(pickle_in)
            self.folder_title = title
            self.path = path
            #unload the most important results and parameters first
            self.x, self.x_initial = \
                core_dict["field"], core_dict["initial field"]
            self.bound_arg, self.charge_arg = \
                core_dict["bound_arg"],core_dict["charge_arg"]
            # m is number of components of field
            self.N, self.m = core_dict["N"], self.x.shape[0]
            #unload all grid related parameters
            self.L,self.w,self.R,self.h  = \
                core_dict["L"], core_dict["w"], core_dict["R"], core_dict["h"]
            self.use_half_grid = core_dict["use_half_grid"]
            #unload error and number of loops
            self.error = core_dict["error"]
            self.max_loop = self.error.size
            self.sor = core_dict["sor"]
            self.tol = core_dict["tol"]
            self.initial_kw = core_dict["initial_kw"]
            #recreate grid object from parameters
            num_z,num_y,num_R = canonical_length_num_conversion(
                self.L,self.w,self.R,self.h)
            self.grid = Dipole_Full_Grid(num_z,num_y,num_R,self.h)
        else:
            raise Exception("Solution file does not exist.")
            
    def display_all(self):
        self.print_attributes()
        self.plot_potential_energy_density()
        self.plot_error()
        self.plot_x_all()
        self.plot_x_initial()
        self.plot_gradient_energy_density()
        self.plot_energy_density()
        self.plot_middle_cross_section_comparison()
        #self.plot_laplacian_all() #implementation needs to be updated
            
    def print_attributes(self):
        print()
        print("Attributes:")
        print("N =", str(self.N))
        print("charge_arg =", self.charge_arg)
        print("bound_arg =", self.bound_arg)
        print("L =", str(self.L))
        print("w =", str(self.w))
        print("h =", str(self.h))
        print("R =", str(self.R))
        print("use_half_grid =",str(self.use_half_grid))
        print("sor =", str(self.sor))
        print("tolerance =", str(self.tol))
        print("error =", str(self.error[-1]))
        print("max loop =", str(self.max_loop))
        print("initial keyword =",str(self.initial_kw))
        print("gradient energy =", self.get_gradient_energy())
        print("potential energy =", self.get_potential_energy())
        print("energy =", str(self.get_energy()))
        print("separation =", str(self.get_string_separation()))
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
        plt.figure()
        plt.plot(np.arange(0,self.max_loop,1),np.log10(self.error))
        plt.ylabel("log(error)")
        plt.title("Error")
        plt.savefig(self.path+"Error.png")
        plt.show()
        
    # def get_laplacian(self):
    #     #initialize second derivative in each direction
    #     d2xdz = np.zeros(shape=self.x.shape,dtype=complex)
    #     d2xdy = np.zeros(shape=self.x.shape,dtype=complex)
    #     for i in range(self.m): #loop over each layer
    #         for j in range(self.grid.num_y): #loop over each row
    #             for k in range(self.grid.num_z): #loop over each column
    #                 d2xdz[i][j][k] = self._get_d2xdz_ijk(i,j,k)
    #                 d2xdy[i][j][k] = self._get_d2xdy_ijk(i,j,k)
    #     return d2xdz + d2xdy

    # def plot_laplacian_all(self):
    #     """
    #     Plot and compare the numerical and theoretical Laplacian to verify
    #     that the solution actually solves the PDE
    #     """
    #     lap_num = self.get_laplacian()
    #     lap_theo = self._get_lap_theo()
    #     for n in range(self.m):
    #         self._plot_laplacian_n(n,lap_num,lap_theo)
    
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
        #Note: simps works best when there are odd number of points, which is
        #always the case for me here, by grid construction.
        gradient_energy = simps(simps(self.get_gradient_energy_density(), 
                             self.grid.z_linspace),self.grid.y_linspace)
        return gradient_energy
    
    def plot_gradient_energy_density(self):
        self._quick_plot(self.get_gradient_energy_density(),
                         "Gradient Energy Density",
                         "Gradient_Energy_Density",
                         cmap='jet')

    def get_potential_energy_density(self):
        W = Superpotential(self.N)
        ped = (1/4)*W.dWdx_absolute_square_on_grid(self.x)
        ped = np.real(ped) #it is real anyway, doing this for plotting
        return ped
               
    def get_potential_energy(self):
        return simps(simps(self.get_potential_energy_density(),self.grid.z_linspace),
                     self.grid.y_linspace)
    
    def plot_potential_energy_density(self):
        self._quick_plot(self.get_potential_energy_density(),
                         "Potential Energy Density",
                         "Potential_Energy_Density",
                         cmap='jet')
        
    def get_energy_density(self):
        return self.get_potential_energy_density() \
               + self.get_gradient_energy_density()
               
    def get_energy(self):
        return simps(simps(self.get_energy_density(),
                           self.grid.z_linspace),self.grid.y_linspace)
    
    
    def plot_energy_density(self):
        self._quick_plot(self.get_energy_density(),
                         "Energy Density (E={})".format(
                             str(round(self.get_energy(),3))),
                         "Energy_Density",
                         cmap='jet')
        
    def plot_x_initial(self):
        for n in range(self.m):
            phi_n = np.real(self.x_initial[n,:,:])
            sigma_n = np.imag(self.x_initial[n,:,:])
            self._quick_plot(phi_n,"initial phi_{}".format(str(n+1)),
                             "initial_phi_{}".format(str(n+1)))
            self._quick_plot(sigma_n,"initial sigma_{}".format(str(n+1)),
                             "initial_sigma_{}".format(str(n+1)))
            
    def get_cross_section_from_z_position(self,z):
        #takes the vertical cross section at z_position
        n_z = self.grid.z_position_to_z_number(z)
        return self.get_cross_section_from_z_number(n_z)
    
    def get_cross_section_from_z_number(self,n_z):
        #takes the vertical cross section of the fields at z_number
        #get all fields, at all rows, at a fixed column
        return self.x[:,:,n_z]
    
    def plot_cross_section_from_z_position(self,z):
        x_cross = self.get_cross_section_from_z_position(z)
        phi_cross = np.real(x_cross)
        sigma_cross = np.imag(x_cross)
        z_linspace = self.grid.z_linspace
        
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(121)
        for i in range(self.m):
            ax1.plot(z_linspace,phi_cross[i],
                     label=r"$\phi_{}$".format(str(i+1)))
        ax1.legend()

        ax2 = fig.add_subplot(122)
        for i in range(self.m):
            ax2.plot(z_linspace,sigma_cross[i],
                     label=r"$\sigma_{}$".format(str(i+1)))
        ax2.legend()

        fig.suptitle("Cross Section at z={}".format(str(z)),size=20)
        fig.savefig(self.path+"Cross_Section_at_z={}.png".format(str(z)), dpi=300)
        
            
    def plot_middle_cross_section_comparison(self):
        z_linspace = self.grid.z_linspace
        #cross section at center
        x_cross = self.get_cross_section_from_z_position(0)
        phi_cross = np.real(x_cross)
        sigma_cross = np.imag(x_cross)
        #BPS at center
        x_initial_cross = self.x_initial[:,:,self.grid.y_axis_number]
        phi_BPS = np.real(x_initial_cross)
        sigma_BPS = np.imag(x_initial_cross)
        
        fig = plt.figure(figsize=(20,20))
        ax1 = fig.add_subplot(221)
        for i in range(self.m):
            ax1.plot(z_linspace,phi_cross[i],
                     label=r"$\phi_{}$".format(str(i+1)))
        ax1.legend(fontsize=15)
        ax1.set_title("$\phi$ Cross Section at z=0",size=20)

        ax2 = fig.add_subplot(222)
        for i in range(self.m):
            ax2.plot(z_linspace,sigma_cross[i],
                     label=r"$\sigma_{}$".format(str(i+1)))
        ax2.legend(fontsize=15)
        ax2.set_title("$\sigma$ Cross Section at z=0",size=20)
        
        ax3 = fig.add_subplot(223)
        for i in range(self.m):
            ax3.plot(z_linspace,phi_BPS[i],
                     label=r"$\phi_{}$".format(str(i+1)))
        ax3.legend(fontsize=15)
        ax3.set_title("BPS $\phi$",size=20)

        ax4 = fig.add_subplot(224)
        for i in range(self.m):
            ax4.plot(z_linspace,sigma_BPS[i],
                     label=r"$\sigma_{}$".format(str(i+1)))
        ax4.legend(fontsize=15)
        ax4.set_title("BPS $\sigma$",size=20)
        
        fig.suptitle("SU({}), {}, R={}, Middle Cross Section Comparison".format(
            str(self.N),self.charge_arg,str(self.R)),size=30)
        fig.savefig(self.path+"Middle_Cross_Section_Comparison.png",dpi=300)
    
    def get_string_separation(self):
        pe = self.get_potential_energy_density()
        center = self.grid.y_axis_number
        pe_top = pe[0:self.grid.z_axis_number,center]
        pe_bottom = pe[self.grid.z_axis_number:,center]
        top_arg = np.argmax(pe_top)
        bottom_arg = np.argmax(pe_bottom) + pe_top.size
        d = self.grid.y_linspace[bottom_arg]-self.grid.y_linspace[top_arg]
        return d
    
    # def compare_slice_with_BPS(self):
    #     for n in range(self.m):
    #         plt.figure()
    #         #take a vertical slice through middle
    #         middle_col = int(self.grid.num_z/2)
    #         plt.plot(self.grid.y, self.get_phi_n(n)[:,middle_col],
    #                  label="final $\phi_{}$".format(str(n+1)))
    #         plt.plot(self.grid.y, np.real(self.BPS_slice[n,:]),
    #                  label="BPS $\phi_{}$".format(str(n+1)))
    #         plt.legend()
    #         plt.title("compare slice with BPS phi_{}".format(str(n+1)))
    #         plt.savefig(self.folder_title +
    #                     "compare_slice_with_BPS_phi_{}.png".format(str(n+1)))
    #         plt.show()
            
    #         plt.figure()
    #         plt.plot(self.grid.y, self.get_sigma_n(n)[:,middle_col],
    #                  label="final $\sigma_{}$".format(str(n+1)))
    #         plt.plot(self.grid.y, np.imag(self.BPS_slice[n,:]),
    #                  label="BPS $\sigma_{}$".format(str(n+1)))
    #         plt.legend()
    #         plt.title("compare slice with BPS sigma_{}".format(str(n+1)))
    #         plt.savefig(self.folder_title +
    #                 "compare_slice_with_BPS_sigma_{}.png".format(str(n+1)))
    #         plt.show()
            

    def _quick_plot(self,field,plot_title,file_title,cmap=None):
        plt.figure()
        plt.pcolormesh(self.grid.zv,self.grid.yv,field,cmap=cmap)
        plt.colorbar()
        plt.title(plot_title)
        plt.savefig(self.path+file_title+".png")
        plt.show()
        
    # def _quick_plot_laplacian(self,field,ax,title,fig):
    #     im = ax.pcolormesh(self.grid.zv,self.grid.yv,field)
    #     ax.set_title(title)
    #     fig.colorbar(im,ax=ax)
        
    # def _plot_laplacian_n(self,n,lap_num,lap_theo):
    #     #row= real & imag of fields; col= numeric vs theoretic
    #     fig, axs = plt.subplots(2, 2) 
    #     fig.subplots_adjust(hspace=0.7)
    #     fig.subplots_adjust(wspace=0.7)
    #     self._quick_plot_laplacian(np.real(lap_num[n,:,:]),axs[0, 0],
    #                     "$\\nabla^2 \phi_{}$ numeric".format(str(n+1)),
    #                                fig)
    #     self._quick_plot_laplacian(np.real(lap_theo[n,:,:]),axs[0,1],
    #                     "$\\nabla^2 \phi_{}$ theoretic".format(str(n+1)),
    #                                fig)
    #     self._quick_plot_laplacian(np.imag(lap_num[n,:,:]),axs[1, 0],
    #                     "$\\nabla^2 \sigma_{}$ numeric".format(str(n+1)),
    #                                fig)
    #     self._quick_plot_laplacian(np.imag(lap_theo[n,:,:]),axs[1,1],
    #                     "$\\nabla^2 \sigma_{}$ theoretic".format(str(n+1)),
    #                         fig)
    #     #add axis label such that repeated are avoided
    #     #for ax in axs.flat:
    #         #ax.set(xlabel='z', ylabel='y')
    #     # Hide x labels and tick labels for top plots and y ticks for right plots.
    #     #for ax in axs.flat:
    #         #ax.label_outer()
    #     fig.savefig(self.folder_title+"Laplacian_{}.png".format(str(n+1)))
            
    # def _get_lap_theo(self):
    #     #return theoretical laplacian
    #     charge = Sigma_Critical(self.N,self.charge_arg)
    #     bound = Sigma_Critical(self.N,self.bound_arg)
    #     relax = Relaxation(self.grid,self.N,bound,charge,
    #                        self.max_loop,x0=None,diagnose=False)
    #     return relax._full_grid_EOM(self.x)
    
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
        elif j==self.grid.z_axis_number-1 and \
            self.grid.left_charge_axis_number<= k <=self.grid.right_charge_axis_number:
            result = (self.x[i][j][k] - self.x[i][j-1][k])/self.grid.h
        elif j==self.grid.z_axis_number and \
            self.grid.left_charge_axis_number<= k <=self.grid.right_charge_axis_number:
            result = (self.x[i][j+1][k] - self.x[i][j][k])/self.grid.h
        else: #two sided derivative elsewhere
            result = (self.x[i][j+1][k] - self.x[i][j-1][k])/(2*self.grid.h)
        return result
    
    # def _get_d2xdz_ijk(self,i,j,k):
    #     if k == 0: #one sided second derivative on the edge (forward difference)
    #         result = (self.x[i][j][k+2] - 2*self.x[i][j][k+1] +
    #                   self.x[i][j][k])/(self.grid.h**2)
    #     elif k==self.grid.num_z-1: #one sided second derivative on the edge
    #         result = (self.x[i][j][k] - 2*self.x[i][j][k-1] +
    #                   self.x[i][j][k-2])/(self.grid.h**2)
    #     else: #two sided second derivative elsewhere
    #         result = (self.x[i][j][k+1] - 2*self.x[i][j][k] +
    #                   self.x[i][j][k-1])/(self.grid.h**2)
    #     return result
    
    # def _get_d2xdy_ijk(self,i,j,k):
    #     if j == 0: #one sided derivative on the edge
    #         result = (self.x[i][j+2][k] - 2*self.x[i][j+1][k] +
    #                   self.x[i][j][k])/(self.grid.h**2)
    #     elif j==self.grid.num_y-1: #one sided derivative on the edge
    #         result = (self.x[i][j][k] - 2*self.x[i][j-1][k] +
    #                   self.x[i][j-2][k])/(self.grid.h**2)
    #     else: #two sided derivative elsewhere
    #         result = (self.x[i][j+1][k] - 2*self.x[i][j][k] +
    #                   self.x[i][j-1][k])/(self.grid.h**2)
    #     return result
    
"""
===============================================================================
                            Analyzing Solutions Group
===============================================================================
After having solved some solutions, analyze a group of solutions in conjunction

Subsection:
Tensions
String Separation
===============================================================================
"""    
def get_all_CS_folder_names():
    folders = []
    for x in os.walk("Confinement Solutions"):
        #take out the big folder name in front
        folders.append(x[0].replace("Confinement Solutions\\",""))
    folders.remove("Confinement Solutions") #remove big folder name
    return folders

def get_all_fully_solved_N_p_folder_names(N,charge_arg,R_greater_than_five=False):
    folders = get_all_CS_folder_names()
    unwanted_folders = []
    important_strings = ["tol=1e-09",
                        "N={},".format(str(N)),
                        "charge={},".format(charge_arg)]
    for folder in folders:
        for string in important_strings:
            if string not in folder:
                #add a folder into garbage list if it doesn't contain
                #important key phrase
                unwanted_folders.append(folder)
            if R_greater_than_five:
                if _get_R_from_folder_title(folder) <= 5:
                    #add folder to grabage list if R is too small
                    unwanted_folders.append(folder)
    #the fullly solved folders are the set difference
    result = list(set(folders) - set(unwanted_folders))
    return result

def _get_R_from_folder_title(folder):
    #the index of the letter 'R' in 'R='
    index_R = folder.find("R=")
    index_equal_sign = index_R + 1
    #get the index of the comma immediately after the R number
    #start search at index_equal_sign
    index_comma = folder.find(",",index_equal_sign)
    R_str = folder[index_equal_sign+1:index_comma]
    R = int(R_str)
    return R

""" ============== subsection: Tensions ===================================="""
def get_R_and_energy_array(N,charge_arg):
    #initialze lists
    R_list = []
    energy_list = []
    folders = get_all_fully_solved_N_p_folder_names(N,charge_arg,
                                                    R_greater_than_five=True)
    print(folders)
    for folder in folders:
        print(folder)
        sol = Solution_Viewer(folder)
        R_list.append(sol.R)
        energy_list.append(sol.get_energy())
    #convert back to array
    R_array = np.array(R_list)
    energy_array = np.array(energy_list)
    return R_array, energy_array

#a dictionary that given N, gives the optimal L,w
N_Lw_dict = {3:[30,30],
             4:[30,30],
             5:[30,30],
             6:[30,30],
             7:[30,30]}

def get_R_and_energy_array_solve_new(N,charge_arg,L,w,R_min,R_max,R_interval):
    #initialze lists
    R_list = []
    energy_list = []
    for R in range(R_min,R_max,R_interval):
        sol = confining_string_solver(N=N,charge_arg=charge_arg,
                                      bound_arg="x1",L=L,
                                      w=w,R=R)
        R_list.append(R)
        energy_list.append(sol.get_energy())
    #convert back to array
    R_array = np.array(R_list)
    energy_array = np.array(energy_list)
    return R_array, energy_array

def plot_energy_vs_R(R_array,energy_array,m,dm,b,db,N,p,L):
    plt.figure()
    plt.scatter(x=R_array,y=energy_array)
    x = np.linspace(1,L,1000)
    y = linear_model(x,m,b)
    plt.plot(x,y,label=r"$E=TR+b$")
    #assumes BPS 1 wall theoretical energy
    vac0 = Sigma_Critical(N,"x0")
    vacf = Sigma_Critical(N,"x1")
    two_T_BPS = 2*get_BPS_theoretic_energy(N,vac0,vacf)
    plt.plot(x,linear_model(x,two_T_BPS,b),
             label=r"$E=2T_{BPS 1}R+b$",linestyle="--")
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    plt.text(x=20,y=y_min+0.3*(y_max-y_min),s="T = {} $\pm$ {}".format(
        str(round(m,3)),str(round(dm,3))))
    plt.text(x=20,y=y_min+0.2*(y_max-y_min),s=r"$2 T(BPS1) = {}$".format(
        str(round(two_T_BPS,3))))
    # plt.text(x=20,y=y[0],s="b = {} $\pm$ {}".format(
    #     str(round(b,3)),str(round(db,3))))
    plt.xlabel("R")
    plt.ylabel("Energy")
    plt.legend()
    plt.title("Energy vs Distance (N={}, p={})".format(str(N),str(p)))
    plt.savefig(
            "Tensions/Energy vs Distance (N={}, p={}).png".format(
            str(N),str(p)))
    plt.show()
    
def linear_model(x,m,b):
    return m*x + b

def compute_tension(N,p,solve_new=False,R_min=None,R_max=None,R_interval=None):
    #p is N-ality
    charge_arg='w'+str(p)
    L,w = N_Lw_dict[N]
    if solve_new:
        _validate_R_arguments(R_min,R_max,R_interval)
        R_array, energy_array = get_R_and_energy_array_solve_new(
            N,charge_arg,L,w,R_min,R_max,R_interval)
    else:
        R_array, energy_array = get_R_and_energy_array(N,charge_arg)
    potp, pcov = curve_fit(linear_model,xdata=R_array,ydata=energy_array)
    m,b = potp #slope and y-intercept
    dm = np.sqrt(pcov[0][0]) #standard deviation of slope
    db = np.sqrt(pcov[1][1])
    plot_energy_vs_R(R_array,energy_array,m,dm,b,db,N,p,L)
    #write to file in append mode
    # with open("Tensions/tensions.txt", "a") as text_file:
    #     text_file.write("{} {} {} {}\n".format(str(N),str(p),str(m),str(dm)))
    return m, dm

def _validate_R_arguments(R_min,R_max,R_interval):
    if (R_min is None) or (R_max is None) or (R_interval is None):
        raise Exception("If you want to solve new solutions,"+
                        " you must specified R_min,R_max,R_interval.")

""" ============== subsection: String Separation Log Growth ===============""" 
def plot_d_vs_R(R_array,d_array,a,da,b,db,c,dc,N,p,L):
    plt.figure()
    #scatter plot of data
    plt.scatter(x=R_array,y=d_array)
    #plot best fit model
    x = np.linspace(1,L,1000)
    y = log_model(x,a,b,c)
    plt.plot(x,y,label=r"$d = a \cdot \ln(R-b) + c $")
    #print parameters
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    plt.text(x=20,y=y_min+0.3*(y_max-y_min),s=r"a = {} $\pm$ {}".format(
        str(round(a,3)),str(round(da,3))))
    plt.text(x=20,y=y_min+0.2*(y_max-y_min),s=r"b = {} $\pm$ {}".format(
        str(round(b,3)),str(round(db,3))))
    plt.text(x=20,y=y_min+0.1*(y_max-y_min),s=r"c = {} $\pm$ {}".format(
        str(round(c,3)),str(round(dc,3))))
    plt.xlabel("R")
    plt.ylabel("d")
    plt.legend()
    plt.title("String Separation vs Distance (N={}, p={})".format(str(N),str(p)))
    plt.savefig(
            "String Separation/String Separation vs Distance (N={}, p={}).png".format(
            str(N),str(p)))
    plt.show()
    
def plot_d_vs_R_for_p1(R_array,d_array,N,p):
    plt.figure()
    #scatter plot of data
    plt.scatter(x=R_array,y=d_array)
    plt.xlabel("R")
    plt.ylabel("d")
    plt.title("String Separation vs Distance (N={}, p={})".format(str(N),str(p)))
    plt.savefig(
            "String Separation/String Separation vs Distance (N={}, p={}).png".format(
            str(N),str(p)))
    plt.show()
    

def get_R_and_d_array(N,charge_arg):
    #initialze lists
    R_list = []
    d_list = []
    folders = get_all_fully_solved_N_p_folder_names(N,charge_arg)
    for folder in folders:
        sol = Solution_Viewer(folder)
        print(sol.R)
        print(sol.get_string_separation())
        R_list.append(sol.R)
        d_list.append(sol.get_string_separation())
    #convert back to array
    R_array = np.array(R_list)
    d_array = np.array(d_list)
    return R_array, d_array

def log_model(x,a,b,c):
    return a*np.log(x-b) + c

def compute_string_separation(N,p):
    #p is N-ality
    charge_arg='w'+str(p)
    L,w = N_Lw_dict[N]
    R_array, d_array = get_R_and_d_array(N,charge_arg)
    if not p == 1:
        potp, pcov = curve_fit(log_model,xdata=R_array,ydata=d_array)
        a,b,c = potp
        da = np.sqrt(pcov[0][0])
        db = np.sqrt(pcov[1][1])
        dc = np.sqrt(pcov[2][2])
        plot_d_vs_R(R_array,d_array,a,da,b,db,c,dc,N,p,L)
    else:
        plot_d_vs_R_for_p1(R_array,d_array,N,p)
    return R_array,d_array

if __name__ == "__main__":
    # sol = confining_string_solver(N=3,charge_arg="w1",bound_arg="x1",
    #                               L=30,w=30,R=9,h=0.1,
    #                               initial_kw="BPS",
    #                               use_half_grid=True)
    # Np_list = [(4,1),(4,2),(5,1),(5,2),(6,1),(6,2),(6,3),(7,1),(7,2),(7,3)]
    # for tup in Np_list:
    #     N,p = tup
    #     compute_tension(N,p)
    #     compute_string_separation(N,p)
    pass
