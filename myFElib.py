## @package myFElib
#  This module contains the basic finite element data structures

import numpy as np
import math

## Finite element node
class Node:

    ## Constructor
    #  @param ID    Node ID
    #  @param coord Node coordinate
    #  @param dof   Dof index
    def __init__ ( self, ID, coord, dof ):
        self.__ID = ID
        self.__dof = dof
        self.set_coordinate( coord )

    ## String function
    def __str__ ( self ):
        return str(self.__ID)
    
    ## Get the Node ID
    def get_ID ( self ):
        return self.__ID
    
    ## Get the Dof index
    def get_dof( self ):
        return self.__dof
    
    ## Set the coordinate
    #  @param coord Node coordinate   
    def set_coordinate( self, coord ):
        assert isinstance( coord, np.ndarray )
        assert coord.ndim==1
        assert coord.dtype==float
        self.__coord = coord

    ## Get the Node coordinate
    def get_coordinate( self ):
        return self.__coord
        
## Finite element triangular parent element
#        
#  Linear triangle parent element with local coordinates (0,0), (1,0), (0,1)         
class StandardTriangle:
    
    ## Dictionary of integration schemes
    #
    #  See e.g. 'http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf' for details
    __ischemes = {
                   ('gauss',3 ) : ( np.array([[1./6.,1./6.],
                                                 [2./3.,1./6.],
                                                 [1./6.,2./3.]]),
                                    np.array([1./6.,1./6.,1./6.]) )
                 }

    ## Number of nodes
    __nnodes = 3#6 
    
    ## Length function
    def __len__ ( self ):
        return self.get_nr_of_nodes()
    
    ## Get the number of nodes    
    def get_nr_of_nodes ( self ):
        return self.__nnodes
    
    ## Get the shape functions
    #  @param  xi Local coordinate vector
    #  @return    Vector of shape functions
    def get_shapes ( self, xi ):
        ## Linear 
        return np.array([1-xi[0]-xi[1],xi[0],xi[1]])
#        ## Quadratic
#        return np.array([(1 - xi[0] - xi[1])*(2*(1 - xi[0] - xi[1]) - 1), 
#                            4*(1 - xi[0] - xi[1])*xi[0], 
#                            xi[0]*(2*xi[0] - 1), 
#                            4*xi[0]*xi[1], 
#                            xi[1]*(2*xi[1] - 1), 
#                            4*(1 - xi[0] - xi[1])*xi[1]])  
    
    ## Get the shape functions gradient
    #  @param  xi Local coordinate vector
    #  @return    Matrix of shape function gradients
    def get_shapes_gradient ( self, xi ):
        ## Linear
        return np.array([[-1.,-1.], [ 1., 0.], [ 0., 1.]])
#        ## Quadratic
#        return np.array([[-3 + 4*(xi[0] + xi[1]), -3 + 4*(xi[0] + xi[1])],
#                            [-8*xi[0] - 4*xi[1], -4*xi[0]],
#                            [4*xi[0], 0],
#                            [4*xi[1], 4*xi[0]],
#                            [0, 4*xi[1]],
#                            [-4*xi[1], -8*xi[1] - 4*xi[0]]])  

    ## Get the integration scheme
    #  @param  name The type of integration scheme (e.g. 'gauss')
    #  @param  npts The number of integration points
    #  @return      Matrix of integration point coordinates
    #  @return      Vector of integration point weights    
    def get_integration_scheme ( self, name, npts ):
        xis, ws = self.__ischemes[ (name,npts) ]
        return xis, ws
        
## Isoparametric finite element
#
#  Maps a standard (parent) element using a node-based parametric map             
class Element:
    
    ## Constructor
    #  @param ID     Element ID
    #  @param parent Standard/parent element
    #  @param nodes  List of finite element Nodes
    def __init__ ( self, ID, parent, nodes ):
        self.__ID     = ID
        self.__nodes  = nodes
        self.__parent = parent
        self.__nnodes = len(parent)
        assert len(parent)==len(nodes)

    ## Iterator function
    def __iter__ ( self ):
        return iter(self.__nodes)    

    ## Length function
    def __len__ ( self ):
        return self.get_nr_of_nodes()
    
    ## Get item function
    def __getitem__ ( self, index ):
        return self.__nodes[index]
    
    ## String function    
    def __str__ ( self ):
        s  = 'Element %d with nodes: ' % self.__ID 
        s += ', '.join(str(node) for node in self.__nodes)
        return s

    ## Get the number of nodes    
    def get_nr_of_nodes ( self ):
        return self.__nnodes    

    ## Get the vector of Dof indices
    def get_dofs( self ):
        return [node.get_dof() for node in self]

    ## Get the matrix of nodal coordinates
    def get_coordinates( self ):
        return np.array([node.get_coordinate() for node in self])
    
    ## Get the global coordinate
    #  @param  xi Local coordinate vector    
    #  @return    Global coordinate vector
    def get_coordinate ( self, xi ):
        coords     = self.get_coordinates()
        std_shapes = self.__parent.get_shapes( xi )
        return coords.T.dot( std_shapes )
    
    ## Get the integration scheme
    #  @param  name The type of integration scheme (e.g. 'gauss')
    #  @param  npts The number of integration points
    #  @return      Matrix of integration point coordinates
    #  @return      Vector of integration point weights    
    def get_integration_scheme ( self, name, npts ):
        xis, ws = self.__parent.get_integration_scheme( name, npts )
        ws = np.array([w*np.abs(np.linalg.det(self.__get_jacobian( xi ))) for xi, w in zip(xis,ws)])
        return xis, ws
        
    ## Get the shape functions
    #  @param  xi Local coordinate vector
    #  @return    Vector of shape functions
    def get_shapes ( self, xi ):
        return self.__parent.get_shapes( xi )
        
    ## Get the shape functions gradient
    #  @param  xi Local coordinate vector
    #  @return    Matrix of shape function gradients
    def get_shapes_gradient ( self, xi ):
        J_inv = np.linalg.inv( self.__get_jacobian( xi ) )
        std_shapes_grad = self.__parent.get_shapes_gradient(xi)
        return std_shapes_grad.dot( J_inv )
        
    ## Get the jacobian
    #  @param  xi Local coordinate vector    
    #  @return    Jacobian matrix 
    def __get_jacobian ( self, xi ):
        coords          = self.get_coordinates()
        std_shapes_grad = self.__parent.get_shapes_gradient(xi)
        return coords.T.dot( std_shapes_grad )
    

## Finite element mesh
class Mesh:

    ## Constructor
    #  @param nodes list of finite element nodes
    #  @param elems list of finite elements
    def __init__ ( self, nodes, elems ):
       self.__nodes = nodes
       self.__elems = elems

    ## Iterator function    
    def __iter__ ( self ):
        return iter(self.__elems)        
        
    ## Length function    
    def __len__ ( self ):
        return len(self.__elems)

    ## String function
    def __str__ ( self ):
        s  = 'Number of nodes   : %d\n' % self.get_nr_of_nodes()
        s += 'Number of elements: %d\n' % len(self)
        return s

    ## Get a node
    #  @param ID Node ID
    def get_node ( self, ID ):
        for node in self.__nodes:
            if node.get_ID()==ID:
                return node
        raise RuntimeError( 'Node ID %d not found' % ID )
    
    ## Get all nodal coordinates
    #  @return Matrix of nodal coordinates
    def get_nodal_coordinates ( self ):
        return np.array([node.get_coordinate() for node in self.__nodes])
    
    ## Get the element connectivity table
    #  @return Matrix (int) with element-Dof connectivities
    def get_connectivity ( self ):
        return np.array([[node.get_dof() for node in elem] for elem in self],dtype=int)
        
    ## Get the number of nodes      
    def get_nr_of_nodes ( self ):
        return len(self.__nodes)

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
def calculate_average_velocity(mesh, sol, cons):
    A = cross_sectional_area(mesh, cons)
    C = mesh.get_connectivity() 
    u_sum = 0
    i = 0
    for element in mesh:
        xis, ws = element.get_integration_scheme( 'gauss', 3 )   

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
            if np.sum(boolarr) == 2:
                Boundary_Node.append([C.item(i,place[0]) , C.item(i,place[1])])
            elif np.sum(boolarr) == 3:
                Boundary_Node.append([C.item(i,place[0]) , C.item(i,place[1]), C.item(i,place[2])])
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
def calculate_normal(mesh, nodes):
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
    print("Nodal coordinates:",X)
    print("Boundary nodes",Z)
    Ac = 0
    for i in range(0, len(Z)):
        norm = calculate_normal(mesh, Z[i, :])
        coor = (X[Z[i, 0], :] + X[Z[i, 1], :]) / 2
        Ac_i = np.dot(coor, norm)  
        Ac += Ac_i
    
    Ac = Ac/2
    
    return Ac
     ## Geometry Factor
def geometry_factor(mesh, sol, cons, params):   
    u = calculate_average_velocity(mesh, sol, cons)
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