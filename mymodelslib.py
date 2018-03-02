## @package mymodelslib
#  This module contains the finite element fluid flow model

import numpy
from myLinAlglib import LinearSystem

## Fluid flow finite element model
class PipeFlow:
    
    ## Constructor
    #  @param params Dictionary of model parameters
    #  @param mesh   Finite element mesh
    #  @param cons   Indices of constrained degrees of freedom
    def __init__ ( self, params, mesh, cons ):
        
        assert isinstance( params['length'], float ) and params['length'] > 0.
        assert isinstance( params['pressure_drop'], float )
        assert isinstance( params['viscosity'], float ) and params['viscosity'] > 0.
        
        self.__mu    = params['viscosity']
        self.__s     = params['pressure_drop']/params['length']
        self.__mesh  = mesh
        self.__cons  = cons

    ## Assemble the finite element system
    #  @return Linear system of equations
    def assemble ( self ):
        
        #Initialize the linear system
        linsys = LinearSystem( self.__mesh.get_nr_of_nodes(), self.__cons )
               
        for element in self.__mesh:
        
            elhs = numpy.zeros( (len(element),)*2 )
            erhs = numpy.zeros( (len(element),)   )
            
            xis, ws = element.get_integration_scheme( 'gauss', 3 )
        
            for xi, w in zip( xis, ws ):
        
               N = element.get_shapes( xi )
               G = element.get_shapes_gradient( xi )
               
               x = element.get_coordinate( xi ) 
               
               erhs += w * self.__s * N
               elhs += w * self.__mu * numpy.dot( G, G.T )
        
            linsys.add( erhs, elhs, element.get_dofs() )
            
        return linsys
