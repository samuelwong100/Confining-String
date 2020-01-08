import sys
sys.path.append("../BPS_Package")
from Sigma_Critical import Sigma_Critical
import numpy as np
from BPS import BPS, SU, Superpotential

s = SU(3)
w = Superpotential(3)
s4 = SU(4)

def test_BPS_dx_boundary():
    xmin0 = Sigma_Critical(3,"x0")
    xminf = Sigma_Critical(3,"x1")
    b=BPS(3,xmin0.imaginary_vector,xminf.imaginary_vector)
    print("dx(-infinity) = ", b.dx(b.xmin0))
    print("dx(infinity) = ", b.dx(b.xminf))
    print()
    
def test_BPS_dx():
    xmin0 = Sigma_Critical(3,"x0")
    xminf = Sigma_Critical(3,"x1")
    b=BPS(3,xmin0.imaginary_vector,xminf.imaginary_vector)
    x = np.array([[1+3j,2+4j]])
    print("theotretical dx(1+3j,2+4j) = [[-2.13527 -2.5800j,3.4782 -0.55049j]]")
    print("my dx(1+3j,2+4j) =", b.dx(x))
    print()
    
def test_BPS_ddx():
    xmin0 = Sigma_Critical(3,"x0")
    xminf = Sigma_Critical(3,"x1")
    b=BPS(3,xmin0.imaginary_vector,xminf.imaginary_vector)
    x = np.array([[1+3j,2+4j]])
    print("theoretical ddx(1+3j,2+4j) = [[-0.0950115 + 12.2985j, 19.2257 -7.25826j]]")
    print("my ddx(1+3j,2+4j) =", b.ddx(x))
    print()
    
test_BPS_dx_boundary() #passed
test_BPS_dx() #passed
test_BPS_ddx() #passed