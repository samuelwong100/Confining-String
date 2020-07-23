# -*- coding: utf-8 -*-
"""
File Name: tests.py
Purpose: 
Author: Samuel Wong
"""
import numpy as np
from confinepy.source import Dipole_Full_Grid, within_epsilon, relaxation_update_full_grid,\
relaxation_update_half_grid, relaxation_algorithm, Superpotential
import matplotlib.pyplot as plt
def test_dipole_full_grid():
    DFG = Dipole_Full_Grid(11,7,3)
    hg = DFG.half_grid
    DFG2 = Dipole_Full_Grid(101,301,1)
    hg2 = DFG2.half_grid
    
    assert DFG.num_z == 11
    assert DFG.num_z_half == 5
    assert DFG.num_y == 7
    assert within_epsilon(DFG.z0,-0.5)
    assert within_epsilon(DFG.zf,0.5)
    assert within_epsilon(DFG.y0,-0.3)
    assert within_epsilon(DFG.yf,0.3)
    assert within_epsilon(DFG.vertical_length,0.6)
    assert DFG.num_R == 3
    assert DFG.num_R_interval == 2
    assert DFG.R == 0.2
    assert DFG.left_charge_axis_number == 4
    assert DFG.right_charge_axis_number == 6
    assert within_epsilon(DFG.left_charge_z_position,-0.1)
    assert within_epsilon(DFG.right_charge_z_position,0.1)
    assert within_epsilon(DFG.zy_number_to_position(3,5),(-0.2,0.2))
    assert within_epsilon(DFG.get_nearest_position_on_grid(0.05,0.01),(0,0))
    assert within_epsilon(DFG.get_nearest_position_on_grid(0.06,0.01),(0.1,0))
    assert within_epsilon(DFG.get_nearest_position_on_grid(0.34,0.18),(0.3,0.2))
    
    assert hg.parent_grid is DFG
    assert hg.parent_grid.half_grid is hg
    assert hg.num_z == 6
    assert hg.z0 == -0.5
    assert hg.zf == 0
    assert hg.num_y == 7
    assert hg.num_R_half == 2
    assert hg.num_R_half_interval == 1
    assert within_epsilon(hg.horizontal_length,0.5)
    assert within_epsilon(DFG.get_nearest_position_on_grid(-0.34,0.18),(-0.3,0.2))
    assert hg.left_charge_axis_number == 4
    assert within_epsilon(hg.left_charge_z_position,-0.1)
    
    x=DFG2.create_vector_field(10)
    x[:,:,0] = 1
    x[:,:,-1] = 1
    y=hg2.create_vector_field(10)
    y[:,:,0] = 1
    x_new = hg2.reflect_vector_field(y)
    assert within_epsilon(np.max(x-x_new),0)

def laplacian_function(x):
    return 2*x

def test_relaxation_update():
    grid = Dipole_Full_Grid(3,3,1)
    x_old =  np.array([[[1.,2,3],[4,5,6],[7,8,9]],
                       [[1,2,3],[4,5,6],[7,8,9]],
                       [[1,2,3],[4,5,6],[7,8,9]]])
    x = relaxation_update_full_grid(x_old=x_old,laplacian_function=laplacian_function,
                             grid=grid)
    x_correct = np.array([[[1,2,3],[4,4.975,6],[7,8,9]],
                       [[1,2,3],[4,4.975,6],[7,8,9]],
                       [[1,2,3],[4,4.975,6],[7,8,9]]])
    assert within_epsilon(x,x_correct)
    
    y = relaxation_update_half_grid(x_old=x_old,laplacian_function=laplacian_function,
                             grid=grid)
    y_correct = np.array([[[1,2,2],[4,4.975,4.975],[7,8,8]],
                       [[1,2,2],[4,4.975,4.975],[7,8,8]],
                       [[1,2,2],[4,4.975,4.975],[7,8,8]]])
    assert within_epsilon(y,y_correct)
    
def test_half_grid_reflect():
    grid = Dipole_Full_Grid(3,3,1)
    y = np.array([[[1,2,2],[4,4.975,4.975],[7,8,8]],
                       [[1,2,2],[4,4.975,4.975],[7,8,8]],
                       [[1,2,2],[4,4.975,4.975],[7,8,8]]])
    y_flipped = np.array([[[1.,2.,2.,2.,1.],
                              [4.,4.975,4.975,4.975,4.],
                              [7.,8.,8.,8.,7.,]],
                                [[1.,2.,2.,2.,1.],
                              [4.,4.975,4.975,4.975,4.],
                              [7.,8.,8.,8.,7.,]],
                                [[1.,2.,2.,2.,1.],
                              [4.,4.975,4.975,4.975,4.],
                              [7.,8.,8.,8.,7.,]]])
    assert within_epsilon(grid.half_grid.reflect_vector_field(y),y_flipped)
    
def test_relaxation_algorithm():
    #the boundary value problem: laplacian(f) = e^(|x|), boundary =e^(bounday),
    # is solved by f = e^(|x|)        
    DFG = Dipole_Full_Grid(31,31,13)
    def exponential_laplacian(x):
        result = DFG.create_vector_field(x.shape[0],dtype=float)
        for m in range(x.shape[0]):
            for row in range(x.shape[1]):
                result[m][row] = np.exp(np.abs(DFG.z_linspace))
        return result
    x_initial = DFG.create_vector_field(2,dtype=float)
    x_correct=exponential_laplacian(x_initial)
    x_initial[:,:,0] =x_correct[:,:,0]
    x_initial[:,:,-1] =x_correct[:,:,-1]
    x_initial[:,0,:] =x_correct[:,0,:]
    x_initial[:,-1,:] =x_correct[:,-1,:]
    x,error,loop = relaxation_algorithm(x_correct, exponential_laplacian,
                         DFG,tol=1e-9,use_half_grid=True)
    for i in range(2):
        plt.figure()
        plt.pcolormesh(DFG.zv,DFG.yv,x_correct[i,:,:])
        plt.colorbar()
        plt.title("Correct solution: f(x,y) = e^-x")
        plt.show()
        
    for i in range(2):
        plt.figure()
        plt.pcolormesh(DFG.zv,DFG.yv,x[i,:,:])
        plt.colorbar()
        plt.title("my solution")
        plt.show()
    return x, error,loop
    
def test_Superpotential_potential_term_on_grid_numba():
    W=Superpotential(6)
    x = np.ones(shape=(5,301,301),dtype=complex)*complex(1.23,1.44)
    assert within_epsilon(W.potential_term_on_grid_numba(x),
                          W.potential_term_on_grid_vectorized(x))
    #%timeit W.potential_term_on_grid_vectorized(x)
    #%timeit W.potential_term_on_grid_numba(x)
    
def test_relaxation_full_grid_update_speed():
    DFG=Dipole_Full_Grid(301,301,101)
    W=Superpotential(6)
    charge_vec= Sigma_Critical(6,"w1").real_vector
    lap=W.create_laplacian_function(DFG,charge_vec,use_half_grid=False)
    x_old = np.ones(shape=(5,301,301),dtype=complex)
    _relaxation_update_full_grid(x_old,lap,DFG,5)
    %timeit _relaxation_update_full_grid(x_old,lap,DFG,5)
    
#test_dipole_full_grid()
#test_relaxation_update()
#test_half_grid_reflect()
#x, error,loop = test_relaxation_algorithm()