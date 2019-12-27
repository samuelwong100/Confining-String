# -*- coding: utf-8 -*-
"""
File Name: test_math.py
Purpose: test
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
from Math import SU, Superpotential
import numpy as np
from numpy import sqrt,pi,sin,cos
from Grid import Standard_Dipole_Grid
import matplotlib.pyplot as plt
"""
test class SU
"""
s = SU(3)
s4 = SU(4)

def test_SU_nu():
    nu1_theo = np.array([1/sqrt(2), 1/sqrt(6)])
    nu2_theo = np.array([-1/sqrt(2),1/sqrt(6)])
    nu3_theo = np.array([0,-2/sqrt(6)])
    print("For SU(3):")
    print("theoretical nu1 =", nu1_theo)
    print("my nu1 =", s.nu[0,:])
    print()
    print("theoretical nu2 =", nu2_theo)
    print("my nu2 =", s.nu[1,:])
    print()
    print("theoretical nu3 =", nu3_theo)
    print("my nu3 =", s.nu[2,:])
    print()
    nu1_theo = np.array([1/sqrt(2), 1/sqrt(6), 1/(2*sqrt(3))])
    nu2_theo = np.array([-1/sqrt(2),1/sqrt(6),1/(2*sqrt(3))])
    nu3_theo = np.array([0,-2/sqrt(6),1/(2*sqrt(3))])
    nu4_theo = np.array([0,0,-sqrt(3)/2])
    print("For SU(4):")
    print("theoretical nu1 =", nu1_theo)
    print("my nu1 =", s4.nu[0,:])
    print()
    print("theoretical nu2 =", nu2_theo)
    print("my nu2 =", s4.nu[1,:])
    print()
    print("theoretical nu3 =", nu3_theo)
    print("my nu3 =", s4.nu[2,:])
    print()
    print("theoretical nu4 =", nu4_theo)
    print("my nu4 =", s4.nu[3,:])
    print()

def test_SU_alpha():
    alpha1_theo = np.array([sqrt(2),0])
    alpha2_theo = np.array([-1/sqrt(2),3/sqrt(6)])
    alpha3_theo = np.array([-1/sqrt(2),-3/sqrt(6)])
    print("For SU(3):")
    print("theoretical alpha1 =", alpha1_theo)
    print("my alpha1 =", s.alpha[0,:])
    print()
    print("theoretical alpha2 =", alpha2_theo)
    print("my alpha2 =", s.alpha[1,:])
    print()
    print("theoretical alpha3 =", alpha3_theo)
    print("my alpha3 =", s.alpha[2,:])
    print()
    alpha1_theo = np.array([sqrt(2),0,0])
    alpha2_theo = np.array([-1/sqrt(2),3/sqrt(6),0])
    alpha3_theo = np.array([0,-2/sqrt(6),2/sqrt(3)])
    alpha4_theo = np.array([-1/sqrt(2),-1/sqrt(6),-2/sqrt(3)])
    print("For SU(4):")
    print("theoretical alpha1 =", alpha1_theo)
    print("my alpha1 =", s4.alpha[0,:])
    print()
    print("theoretical alpha2 =", alpha2_theo)
    print("my alpha2 =", s4.alpha[1,:])
    print()
    print("theoretical alpha3 =", alpha3_theo)
    print("my alpha3 =", s4.alpha[2,:])
    print()
    print("theoretical alpha4 =", alpha4_theo)
    print("my alpha4 =", s4.alpha[3,:])
    print()
    
def test_SU_w():
    w1_theo = np.array([1/sqrt(2), 1/sqrt(6)])
    w2_theo = np.array([0,2/sqrt(6)])
    print("For SU(3):")
    print("theoretical w1 =", w1_theo)
    print("my w1 =", s.w[0,:])
    print()
    print("theoretical w2 =", w2_theo)
    print("my w2 =", s.w[1,:])
    print()
    w1_theo = np.array([1/sqrt(2), 1/sqrt(6),1/(2*sqrt(3))])
    w2_theo = np.array([0,2/sqrt(6),1/sqrt(3)])
    w3_theo = np.array([0,0,sqrt(3)/2])
    print("For SU(4):")
    print("theoretical w1 =", w1_theo)
    print("my w1 =", s4.w[0,:])
    print()
    print("theoretical w2 =", w2_theo)
    print("my w2 =", s4.w[1,:])
    print()
    print("theoretical w3 =", w3_theo)
    print("my w3 =", s4.w[2,:])
    print()
    
def test_SU_rho():
    rho_theo = np.array([1/sqrt(2),3/sqrt(6)])
    print("For SU(3):")
    print("theoretical rho = ", rho_theo)
    print(s.rho)
    print()
    rho_theo = np.array([1/sqrt(2),3/sqrt(6),sqrt(3)])
    print("For SU(4):")
    print("theoretical rho = ", rho_theo)
    print(s4.rho)
    print()
    
"""
test class Superpotential
"""
w = Superpotential(3)
w4 = Superpotential(4)
def test_superpotential_x_min():
    print("For SU3:")
    x0_theo = np.array([0,0])
    x1_theo = np.array([complex(0,sqrt(2)*pi/3),complex(0,2*pi/sqrt(6))])
    x2_theo = np.array([complex(0,4*pi/(3*sqrt(2))),complex(0,4*pi/sqrt(6))])
    print("theoretical x0 =", x0_theo)
    print("my x0 =", w.x_min[0,:])
    print()
    print("theoretical x1 =", x1_theo)
    print("my x1 =", w.x_min[1,:])
    print()
    print("theoretical x2 =", x2_theo)
    print("my x2 =", w.x_min[2,:])
    print()
    print("For SU4:")
    x0_theo = np.array([0,0,0])
    rho = np.array([1/sqrt(2),3/sqrt(6),sqrt(3)])
    x1_theo = complex(0,pi/2)*rho
    x2_theo = complex(0,pi)*rho
    x3_theo = complex(0,3*pi/2)*rho
    print("theoretical x0 =", x0_theo)
    print("my x0 =", w4.x_min[0,:])
    print()
    print("theoretical x1 =", x1_theo)
    print("my x1 =", w4.x_min[1,:])
    print()
    print("theoretical x2 =", x2_theo)
    print("my x2 =", w4.x_min[2,:])
    print()
    print("theoretical x3 =", x3_theo)
    print("my x3 =", w4.x_min[3,:])
    print()
    
def test_superpotential_call():
    x=np.array([[0,0],[5,2j]])
    print("theoretical w(0,0) =",[[3],[1177.36]])
    print("my w(0,0) =",w(x))
    print()
    x=np.array([[1+2j,3+4j]])
    print("theoretical w(1+2j,3+4j) =", np.array([[-22.2-5.27j]]))
    print("my w(1+2j,3+4j) =", w(x))
    print()

#TODO: finish dwdx test
#def test_dWdx():
#    x = np.array([[1,1],[1,1]])
    

def test_ddWddx():
    x = np.array([[1,1],[1,1]])
    print("theoretical ddWddx = [[9.13797, 2.7344],[9.13797, 2.7344]]")
    print("numerical ddWddx = " + str(w.ddWddx(x)))
    
def test_ddW_dxb_dxa():
    x = np.array([[1,1],[1,1]])
    print("theoretical ddWdx1dx1 = [9.13797,9.13797]")
    print("numerical ddWdx1dx1 = " + str(w.ddW_dxb_dxa(x,0,0)))
    print()
    print("theoretical ddWdx1dx2 = [-1.32777,-1.32777]")
    print("numerical ddWdx1dx2 = " + str(w.ddW_dxb_dxa(x,0,1)))
    print()
    print("theoretical ddWdx2d1 = [-1.32777,-1.32777]")
    print("numerical ddWdx2dx1= " + str(w.ddW_dxb_dxa(x,1,0)))
    print()
    print("theoretical ddWdx2dx2 = [2.7344,2.7344]")
    print("numerical ddWdx2dx2 = " + str(w.ddW_dxb_dxa(x,1,1)))
    
def test_sum_over_dWdxa_ddWdxbdxa_conj():
    #x is a 2-dimensional field with two sample points
    x = np.array([[1,1],[1,1]])
    print("For b=1:")
    print("theoretical sum = [38.879,38.879]")
    print("numerical sum =" + str(w.sum_over_dWdxa_ddWdxbdxa_conj(x,0)))
    
def test_sum_over_dWdxa_ddWdxbdxa_conj_on_grid1():
    x = np.ones(shape=(2,5,5))
    print(w.sum_over_dWdxa_ddWdxbdxa_conj_on_grid(x))
    #should see first layer being 38.9
    #should see second layer being -0.877
    
def test_sum_over_dWdxa_ddWdxbdxa_conj_on_grid2():
    S=SU(2)
    W=Superpotential(2)
    grid = Standard_Dipole_Grid(L=1,w=1,h=0.1,R=0.5)
    #I worked out that the function on SU(2) acting on purely imaginary x
    #is given by this equation
    def correct_scalar(sigma):
        #sigma is real
        return 1j*8*sqrt(2)*sin(sqrt(2)*sigma)*cos(sqrt(2)*sigma)
    
    print("Test on homogeneous field 1:")
    x = np.ones(shape=(1,grid.num_y,grid.num_z),dtype=complex)*1j
    print("x = ",x)
    print()
    x_new = W.sum_over_dWdxa_ddWdxbdxa_conj_on_grid(x)
    print("potential(x) = ",x_new)
    print()
    print("analytic potential(x) = ",correct_scalar(1))
    print()
    
    print("Test on homogeneous field 2:")
    x = np.ones(shape=(1,grid.num_y,grid.num_z),dtype=complex)*1j*2*np.pi*S.w[0]
    print("x = ",x)
    print()
    x_new = W.sum_over_dWdxa_ddWdxbdxa_conj_on_grid(x)
    print("potential(x) = ",x_new)
    print()
    print("analytic potential(x) = ",correct_scalar(2*np.pi*S.w[0]))
    print()
    
    print("Test on discontinuous field:")
    x = np.zeros(shape=(1,grid.num_y,grid.num_z),dtype=complex)
    x[:,:,grid.left_axis:grid.right_axis]=1j*2*np.pi*S.w[0]
    x_new = W.sum_over_dWdxa_ddWdxbdxa_conj_on_grid(x)
    plt.figure()
    plt.pcolormesh(grid.zv,grid.yv,np.imag(x[0,:,:]))
    plt.colorbar()
    plt.title("x")
    plt.show()
    plt.figure()
    plt.pcolormesh(grid.zv,grid.yv,np.imag(x_new[0,:,:]))
    plt.colorbar()
    plt.clim(vmin=0,vmax=1)
    plt.title("potential(x)")
    plt.show()

def test_dWdx_absolute_squared():
    W=Superpotential(3)
    x = np.array([[0,0],[1,1]])
    print("expected: [ 0.        +0.j 24.02876087+0.j]")
    print("result: ",W.dWdx_absolute_square(x))
    
def test_dWdx_absolute_square_on_grid():
    W=Superpotential(3)
    x = np.array([[[1,1,1],[1,1,1],[0,0,0]],[[1,1,1],[1,1,1],[0,0,0]]])
    print("expected: [[24.02876087+0.j 24.02876087+0.j 24.02876087+0.j]\
 [24.02876087+0.j 24.02876087+0.j 24.02876087+0.j]\
 [ 0.        +0.j  0.        +0.j  0.        +0.j]]")
    print("result: ",W.dWdx_absolute_square_on_grid(x))

#test_SU_nu() #pass
#test_SU_alpha() #pass
#test_SU_w() #pass
#test_SU_rho() #pass

#test_superpotential_x_min() #pass
#test_superpotential_call() #pass
#test_ddWddx() #pass
#test_ddW_dxb_dxa() #pass
#test_sum_over_dWdxa_ddWdxbdxa_conj()
#test_sum_over_dWdxa_ddWdxbdxa_conj_on_grid1() #pass
#test_sum_over_dWdxa_ddWdxbdxa_conj_on_grid2() #pass
#test_dWdx_absolute_squared() #pass
#test_dWdx_absolute_square_on_grid() #pass