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
#from myFElib import *
import math


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
        Area+=Area_i
    
    return Area

# Calculate Average velocity
def calculate_average_velocity(mesh, sol):
    X = mesh.get_nodal_coordinates()
    A = calculate_cross_section(mesh)

    u_sum = 0
    i = 0
    for element in mesh:
        xis, ws = element.get_integration_scheme( 'gauss', 3 )   
        C = mesh.get_connectivity() 
        n_j =  C[i,:]
        i+=1

        u_i = 0
        for xi, w in zip( xis, ws ):
            N_j = element.get_shapes( xi ) 
            if len(N_j) == 3:
                u_i_j = [sol.item(n_j[0]) , sol.item(n_j[1]) , sol.item(n_j[2])]
            elif len(N_j) == 6:
                u_i_j = [sol.item(n_j[0]) , sol.item(n_j[1]) , sol.item(n_j[2]), sol.item(n_j[3]) , sol.item(n_j[4]) , sol.item(n_j[5])]
            u_i += w * np.dot(N_j, u_i_j)           
        u_sum += u_i
    
    u=(1/A)*u_sum

    return u
    
## Determine Boundary Node Values
def Boundary_Nodes(mesh, cons):
    C = mesh.get_connectivity()
    B = cons
    
    Boundary_Node=[]
    for i in range(0, len(C)):
        boolarr = np.in1d(C[i,:],B)
        if np.sum(boolarr) > 1:
            position = np.where(np.in1d(C[i,:],B))
            place = position[0]
            Boundary_Node.append([C.item(i,place[0]) , C.item(i,place[1])])

    return Boundary_Node


## Calculate circumference
def calculate_circumference(mesh, cons):
    X = mesh.get_nodal_coordinates()
    Z = np.array(Boundary_Nodes(mesh, cons))

    length=0
    for i in range(0, len(Z)):
        length_i = math.sqrt(( X.item(Z[i,0],0) - X.item(Z[i,1],0))**2 + 
                            ( X.item(Z[i,0],1) - X.item(Z[i,1],1))**2  )
        length += length_i
        
    return length

## Calculate the normal vector on the element
def calculate_normal(mesh, cons, nodes):
    X = mesh.get_nodal_coordinates()    

    dx = X.item(nodes[0], 0) - X.item(nodes[1], 0)
    dy = X.item(nodes[0], 1) - X.item(nodes[1], 1)
        
    if (X.item(nodes[0], 0) > 0 and X.item(nodes[1], 0) > 0):
        if np.sign([dy]) > 0:
            norm = [dy, -dx]
        else:
            norm = [-dy, dx]
    elif (X.item(nodes[0], 0) < 0 and X.item(nodes[1], 0) < 0):
        if np.sign([dy]) < 0:
            norm = [dy, -dx]
        else:
            norm = [-dy, dx]        
    elif (X.item(nodes[0], 1) > 0 and X.item(nodes[1], 1) > 0):
        if np.sign([dx]) < 0:
            norm = [dy, -dx]
        else:
            norm = [-dy, dx]        
    elif X.item(nodes[0], 1) < 0 and X.item(nodes[1], 1) < 0:
        if np.sign([dx]) > 0:
            norm = [dy, -dx]
        else:
            norm = [-dy, dx]
    else:
        norm = [0, 0]
    
    return norm


## Calculate the cross sectional area
def cross_sectional_area(mesh, cons):
    X = mesh.get_nodal_coordinates()
    Z = np.array(Boundary_Nodes(mesh, cons))
    
    Ac = 0
    for i in range(0, len(Z)):
        norm = calculate_normal(mesh, cons, Z[i, 0:2])
        coor = (X[Z[i, 0], 0:2] + X[Z[i, 1], 0:2]) / 2
        Ac_i = np.dot(coor, norm)  
        Ac += Ac_i
    
    Ac = Ac/2
    
    return Ac
    
    ## Geometry Factor
def geometry_factor(mesh, sol, cons, params):
    u = calculate_average_velocity(mesh, sol)
    lc = calculate_circumference(mesh, cons)
    Ac_old = calculate_cross_section(mesh)
    Ac_new = cross_sectional_area(mesh, cons)

    mu    = params['viscosity']
    s     = params['pressure_drop']/params['length']

    gm = (32/u)*((Ac_new/lc)**2)*(s/mu)

    print("cross-sec. area old     [m^2] : ", Ac_old)
    print("cross-sec. area new     [m^2] : ", Ac_new)
    print("velocity average     [m/s] : ", u)
    print("circumference        [m]   : ", lc)
    print("geometry factor      [-]   : ", gm)