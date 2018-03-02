from myIOlib import read_from_txt, plot_solution
from mymodelslib import PipeFlow

#Define the model parameters
meshfile = 'meshes/circle_coarse_p1.txt'
outfile  = 'output/output.png'
params   = { 'length'        : 1.,
             'pressure_drop' : 1.,
             'viscosity'     : 1e-3 }

#Read the mesh and constraints from a text file
mesh, cons = read_from_txt( meshfile )

#Construct the finite element model
femodel = PipeFlow( params, mesh, cons )

#Assemble the linear system of equations
linsys = femodel.assemble()

#Solve the system of equations    
sol = linsys.solve()

#Post-processing
plot_solution( mesh, sol, outfile )
