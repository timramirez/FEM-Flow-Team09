## @package myFElib
#  This module contains the basic finite element data structures

import numpy

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
        assert isinstance( coord, numpy.ndarray )
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
                   ('gauss',3 ) : ( numpy.array([[1./6.,1./6.],
                                                 [2./3.,1./6.],
                                                 [1./6.,2./3.]]),
                                    numpy.array([1./6.,1./6.,1./6.]) )
                 }

    ## Number of nodes
    __nnodes = 3
    
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
        return numpy.array([1-xi[0]-xi[1],xi[0],xi[1]])
    
    ## Get the shape functions gradient
    #  @param  xi Local coordinate vector
    #  @return    Matrix of shape function gradients
    def get_shapes_gradient ( self, xi ):
        return numpy.array([[-1.,-1.],
                            [ 1., 0.],
                            [ 0., 1.]])

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
        return numpy.array([node.get_coordinate() for node in self])
    
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
        ws = numpy.array([w*numpy.abs(numpy.linalg.det(self.__get_jacobian( xi ))) for xi, w in zip(xis,ws)])
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
        J_inv = numpy.linalg.inv( self.__get_jacobian( xi ) )
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
        return numpy.array([node.get_coordinate() for node in self.__nodes])
    
    ## Get the element connectivity table
    #  @return Matrix (int) with element-Dof connectivities
    def get_connectivity ( self ):
        return numpy.array([[node.get_dof() for node in elem] for elem in self],dtype=int)
        
    ## Get the number of nodes      
    def get_nr_of_nodes ( self ):
        return len(self.__nodes)
