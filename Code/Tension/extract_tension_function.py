# -*- coding: utf-8 -*-
"""
File Name: extract_tension_function.py
Purpose: extract the dependence of tension on N and p
Author: Samuel Wong
"""
import numpy as np
import matplotlib.pyplot as plt

class tensions_manager():
    def __init__(self,data):
        self.N=data[:,0]
        self.p=data[:,1]
        self.T=data[:,2]
        self.dT=data[:,3]
        self.point_list = []
        for point in data:
            self.point_list.append(tension_point(*point))
    
    def display_all_points(self):
        for point in self.point_list:
            point.display()            
        
class tension_point():
    def __init__(self,N,p,T,dT):
        self.N = N
        self.p = p
        self.T = T
        self.dT = dT
        
    def display(self):
        print("N={}, p={}, T={}, dT={}".format(self.N,self.p,self.T,self.dT))
    

#load data
data = np.loadtxt('../Results/Tensions/tensions.txt')
#create a tension dictionary
tension_dict = {}
for point in data:
    N,p,T,dT = point
    tension_dict[(N,p)] = (T,dT)
    
#functions for tension and tension uncertainty
def T(N,p):
    return tension_dict[(N,p)][0]

def dT(N,p):
    return tension_dict[(N,p)][1]

#create dictionary for f function
f_dict = {}
df_dict = {}
for arg in tension_dict:
    N,p = arg
    f_dict[(N,p)] = T(N,p)/T(N,1)
    df_dict[(N,p)] = (T(N,p)/T(N,1))* np.sqrt((dT(N,p)/T(N,p))**2 + \
           (dT(N,1)/T(N,1))**2)
    
#create f function
def f(N,p):
    return f_dict[(N,p)]

def df(N,p):
    return df_dict[(N,p)]

N1_list = np.array([2,3,4,5,6])
T1_list = np.array([T(2,1),T(3,1),T(4,1),T(5,1),T(6,1)])
dT1_list = np.array([dT(2,1),dT(3,1),dT(4,1),dT(5,1),dT(6,1)])

N2_list = np.array([4,5,6])
T2_list = np.array([T(4,2),T(5,2),T(6,2)])
dT2_list = np.array([dT(4,2),dT(5,2),dT(6,2)])

N3_list = np.array([6])
T3_list = np.array([T(6,3)])
dT3_list = np.array([dT(6,3)])

plt.figure()
plt.errorbar(x=N1_list,y=T1_list,yerr=dT1_list,fmt='o',ls='-',label='p=1')
plt.errorbar(x=N2_list,y=T2_list,yerr=dT2_list,fmt='o',ls='-',label='p=2')
plt.errorbar(x=N3_list,y=T3_list,yerr=dT3_list,fmt='o',ls='-',label='p=3')
plt.xlabel('N')
plt.ylabel('Tension')
plt.title('Tension vs N')
plt.legend()
plt.savefig('Tension vs N.png')
plt.show()

N2_list = np.array([4,5,6])
f2_list = np.array([f(4,2),f(5,2),f(6,2)])
df2_list = np.array([df(4,2),df(5,2),df(6,2)])

N3_list = np.array([6])
f3_list = np.array([f(6,3)])
df3_list = np.array([df(6,3)])

plt.figure()
plt.errorbar(x=N2_list,y=f2_list,yerr=df2_list,fmt='o',ls='-',label='p=2')
plt.errorbar(x=N3_list,y=f3_list,yerr=df3_list,fmt='o',ls='-',label='p=3')
plt.xlabel('N')
plt.ylabel('f')
plt.title('f(N,p)')
plt.legend()
plt.savefig('f(N,p).png')
plt.show()

#
##compute f function, which is ratio with tension of p=1
##f(N,p) = T(N,p)/T(N,1)
#f2= m[p2_mask]/m[p1_mask]
#f3= m[p3_mask]/m[p1_mask]
#
#plt.figure()
#plt.errorbar(x=N[p2_mask],y=f2[p2_mask],yerr=dm[p2_mask],fmt='o',ls='-',label='p=2')
#plt.errorbar(x=N[p3_mask],y=m[p3_mask],yerr=dm[p3_mask],fmt='o',ls='-',label='p=3')
#plt.xlabel('N')
#plt.ylabel('Tension')
#plt.title('Tension vs N')
#plt.legend()
#plt.savefig('Tension vs N.png')
#plt.show()

#plt.figure()
#plt.errorbar(x=N[p1_mask],y=m[p1_mask],yerr=dm[p1_mask],fmt='o',label='p=1')
#plt.xlabel('N')
#plt.ylabel('Tension')
#plt.title('Tension vs N')
#plt.legend()
#plt.show()
#
#plt.figure()
#plt.errorbar(x=N[p2_mask],y=m[p2_mask],yerr=dm[p2_mask],fmt='o',label='p=2')
#plt.xlabel('N')
#plt.ylabel('Tension')
#plt.title('Tension vs N')
#plt.legend()
#plt.show()
#
#plt.figure()
#plt.errorbar(x=N[p3_mask],y=m[p3_mask],yerr=dm[p3_mask],fmt='o',label='p=3')
#plt.xlabel('N')
#plt.ylabel('Tension')
#plt.title('Tension vs N')
#plt.legend()
#plt.show()