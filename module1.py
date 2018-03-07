# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:00:32 2018

@author: s139248
"""

## @package module1.py
#  This module contains question a

import numpy as np
from scipy.integrate import dblquad
from myLinAlglib import LinearSystem
from myFElib import *


## https://math.stackexchange.com/questions/516219/finding-out-the-area-of-a-triangle-if-the-coordinates-of-the-three-vertices-are

## Calculate Cross Section
def calculate_cross_section(mesh):
    X = mesh.get_nodal_coordinates()
    C = mesh.get_connectivity()

    Area_i=0
    Area=0
    for i in range(1, len(C)):
        Area_i=0.5*abs(     X.item(C.item(i-1,0),0)*X.item(C.item(i-1,1),1) +  
                            X.item(C.item(i-1,1),0)*X.item(C.item(i-1,2),1) + 
                            X.item(C.item(i-1,2),0)*X.item(C.item(i-1,0),1) - 
                            X.item(C.item(i-1,0),0)*X.item(C.item(i-1,2),1) -
                            X.item(C.item(i-1,2),0)*X.item(C.item(i-1,1),1) -
                            X.item(C.item(i-1,1),0)*X.item(C.item(i-1,0),1)     )
        print(Area_i)
        Area+=Area_i
    
    print("cross-sec. area [m^2] : ", Area)
    return Area

## Calculate Average velocity
def calculate_average_velocity(mesh, sol):
    X = mesh.get_nodal_coordinates()
    A = calculate_cross_section(mesh)
    C = mesh.get_connectivity()
        
    u_sum = 0
    for i in range(0, len(C)-1):
        xi, w = StandardTriangle.get_integration_scheme ( self, 'gauss', 3)
        print("Intigration points:", xi)
        u_i = 0
        for j in range(0, get_nr_of_nodes - 1):
            N_j = get_shapes ( xi(j) )
            n_j = #Obtain the nodal indices of the nodes in the current element (might be better to do outside loop)
            u_i_j = [ ] #Insert value of speed in nodes 0, 1 and 2
            u_i += np.dot(N_j, u_i_je)           
        u_sum += u_i

#    u=(1/A)*u_sum
#    print("velocity average [m/s] : ", u)
#    print(sol)