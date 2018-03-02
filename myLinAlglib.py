## @package myLinAlglib
#  This module contains the linear system class

import numpy

## Linear system of equations
class LinearSystem:
    
    ## Constructor
    #  @param size     Number of degrees of freedom
    #  @param zerocons Indices of constrained degrees of freedom
    def __init__ ( self, size, zerocons ):
        self.__size = size
        self.__rhs = numpy.zeros( size )
        self.__lhs = numpy.zeros( (size,size) )
        self.__cons = numpy.zeros( size, dtype=bool )
        self.__cons[zerocons] = True
    
    ## Length function  
    def __len__ ( self ):
        return self.__size
    
    ## Add contribution to the linear system
    #  @param vec   Vector to be added to the right-hand-side
    #  @param mat   Matrix to be added to the left-hand-side
    #  @param rdofs Row degrees of freedom to add to
    #  @param cdofs Column degrees of freedom to add to
    def add ( self, vec, mat, rdofs, cdofs=None ):
        self.add_to_rhs( vec, rdofs )
        self.add_to_lhs( mat, rdofs, cdofs )
        
    ## Add contribution to the right-hand-side
    #  @param vec   Vector to be added to the right-hand-side
    #  @param rdofs Row degrees of freedom to add to
    def add_to_rhs ( self, vec, rdofs ):
        self.__rhs[rdofs] += vec
        
    ## Add contribution to the left-hand-side
    #  @param mat   Matrix to be added to the left-hand-side
    #  @param rdofs Row degrees of freedom to add to
    #  @param cdofs Column degrees of freedom to add to
    def add_to_lhs ( self, mat, rdofs, cdofs=None ):
        if not cdofs:
            cdofs = rdofs
        self.__lhs[numpy.ix_(rdofs,cdofs)] += mat

    ## Solve the constrained linear system of equations
    #  @return Solution vector
    def solve ( self ):
        lhs_free = self.__lhs[numpy.ix_(~self.__cons,~self.__cons)]
        rhs_free = self.__rhs[~self.__cons]
        sol = numpy.zeros( len(self) )
        sol[~self.__cons] = numpy.linalg.solve( lhs_free, rhs_free )
        return sol
