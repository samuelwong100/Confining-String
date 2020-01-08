from equations_tools import *

s = SU(3)
w = Superpotential(3)

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
    
def test_superpotential_x_min():
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
    
def test_superpotential_call():
    x=np.array([[0,0],[5,2j]])
    print("theoretical w(0,0) =",[[3],[1177.36]])
    print("my w(0,0) =",w(x))
    print()
    x=np.array([[1+2j,3+4j]])
    print("theoretical w(1+2j,3+4j) =", np.array([[-22.2-5.27j]]))
    print("my w(1+2j,3+4j) =", w(x))
    print()

def test_BPS_dx_boundary():
    b=BPS(3,0,1)
    print("dx(-infinity) = ", b.dx(b.xmin0))
    print("dx(infinity) = ", b.dx(b.xminf))
    print()
    
def test_BPS_dx():
    b=BPS(3,0,1)
    x = np.array([[1+3j,2+4j]])
    print("theotretical dx(1+3j,2+4j) = [[-2.13527 -2.5800j,3.4782 -0.55049j]]")
    print("my dx(1+3j,2+4j) =", b.dx(x))
    print()
    
def test_BPS_ddx():
    b=BPS(3,0,1)
    x = np.array([[1+3j,2+4j]])
    print("theoretical ddx(1+3j,2+4j) = [[-0.0950115 + 12.2985j, 19.2257 -7.25826j]]")
    print("my ddx(1+3j,2+4j) =", b.ddx(x))
    print()
    
def test_grad():
    def f(x):
        m = x.shape[0]
        result = np.zeros(shape=(m,1))
        for row in range(m):
            result[row] = x[row][0]*(x[row][1])**2
        return result
    
    x = np.array([[1,2],[3,4]])
    print("thereotical df(x) = [[4,4],[16,24]]")
    print("my df(x) =", grad(f,x))
    print()
    
#test_SU_nu() #passed
#test_SU_alpha() #passed
#test_SU_w() #passed
test_SU_rho() #passed
#test_superpotential_x_min() #passed
#test_grad() #passed
#test_superpotential_call() #passed
#test_BPS_dx_boundary() #passed
#test_BPS_dx() #passed
#test_BPS_ddx() #passed