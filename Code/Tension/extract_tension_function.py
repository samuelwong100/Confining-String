# -*- coding: utf-8 -*-
"""
File Name: extract_tension_function.py
Purpose: extract the dependence of tension on N and p
Author: Samuel Wong
"""
import numpy as np
import matplotlib.pyplot as plt
        
#load data
data = np.loadtxt('../Results/Tensions/tensions.txt')
N=data[:,0]
p=data[:,1]
m=data[:,2]
dm=data[:,3]

#get the mask of N-ality series
p1_mask = (p==1)
p2_mask = (p==2)
p3_mask = (p==3)

plt.figure()
plt.errorbar(x=N[p1_mask],y=m[p1_mask],yerr=dm[p1_mask],fmt='o',ls='-',label='p=1')
plt.errorbar(x=N[p2_mask],y=m[p2_mask],yerr=dm[p2_mask],fmt='o',ls='-',label='p=2')
plt.errorbar(x=N[p3_mask],y=m[p3_mask],yerr=dm[p3_mask],fmt='o',ls='-',label='p=3')
plt.xlabel('N')
plt.ylabel('Tension')
plt.title('Tension vs N')
plt.legend()
plt.savefig('Tension vs N.png')
plt.show()

#compute f function, which is ratio with tension of p=1
#f(N,p) = T(N,p)/T(N,1)
f2= m[p2_mask]/m[p1_mask]
f3= m[p3_mask]/m[p1_mask]

plt.figure()
plt.errorbar(x=N[p2_mask],y=f2[p2_mask],yerr=dm[p2_mask],fmt='o',ls='-',label='p=2')
plt.errorbar(x=N[p3_mask],y=m[p3_mask],yerr=dm[p3_mask],fmt='o',ls='-',label='p=3')
plt.xlabel('N')
plt.ylabel('Tension')
plt.title('Tension vs N')
plt.legend()
plt.savefig('Tension vs N.png')
plt.show()
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