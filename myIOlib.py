## @package myIOlib
#  This module contains a mesh reader and basic plotting function

from myFElib import *
import numpy
## Mesh file reader
#
#  @param  fname Name of the mesh file
#  @return       Finite element mesh
#  @return       Indices of constrained degrees of freedom
def read_from_txt ( fname ):

    #Open the file to read
    fin = open( fname )
    
    #Read the number of nodes
    line     = fin.readline()
    linelist = line.strip().split()
    assert linelist[0]=='NNODES'
    nnodes = int(linelist[1])

    #Read the nodes
    nodes   = []
    nodeIDs = []
    for dof in range( nnodes ):

        linelist = fin.readline().strip().split()
        nodeID   = int(linelist[0])
        coord    = numpy.array(linelist[1:],dtype=float)
        node     = Node( nodeID, coord, dof )
        
        nodeIDs.append( nodeID )
        nodes  .append( node   )

    #Empty line
    fin.readline()

    #Read the number of elements
    linelist = fin.readline().strip().split()
    assert linelist[0]=='NELEMS'
    nelems = int(linelist[1])

    #Read the elements
    elems = []
    std_triangle = StandardTriangle()
    for ielem in range( nelems ):
        
        linelist    = fin.readline().strip().split()
        elemID      = int(linelist[0])
        elemnodeIDs = numpy.array(linelist[1:],dtype=int)
        elemnodes   = [ nodes[nodeIDs.index(elemnodeID)] for elemnodeID in elemnodeIDs ]
        elem        = Element( elemID, std_triangle, elemnodes )
        
        elems.append( elem )

    mesh = Mesh( nodes, elems )

    #Empty line
    fin.readline()

    #Read the zero constraints
    assert fin.readline().strip()=='ZEROCONS'        
    linelist = fin.readline().strip().split()
    nodeIDs  = numpy.array(linelist,dtype=int)        
    dofs     = [ mesh.get_node( nodeID ).get_dof() for nodeID in nodeIDs ]
    cons = numpy.array(dofs,dtype=int)
    
    fin.close()
    
    return mesh, cons
 

import matplotlib.pyplot as plt
import matplotlib.tri as tri

## Plot the solution on a finite element mesh
#
#  @param mesh    Finite element mesh
#  @param sol     Solution vector
#  @param outfile Name of the output file
def plot_solution( mesh, sol, outfile ):

    #Create the Triangulation
    X = mesh.get_nodal_coordinates()
    C = mesh.get_connectivity()
    
    triang = tri.Triangulation( X[:,0], X[:,1], C )
    
    #Plotting
    plt.figure()
    plt.tripcolor(triang, sol, edgecolors='k' )
    
    #Plot configuration
    plt.axis('off')
    plt.colorbar()

    #Save the figure to the output file
    plt.savefig( outfile )
    
    print( 'Output written to {}'.format( outfile ) )

    ## Geometry Factor
def getdata(mesh, sol, cons, params):
    geometry_factor(mesh, sol, cons, params)
    
    