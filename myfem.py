from myIOlib import read_from_txt, plot_solution
from myFElib import *
from mymodelslib import PipeFlow
import time

t0 = time.time()

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

##Calculate geometry factor
process_data(mesh, sol, cons, params)

t1=time.time()
total=t1-t0
print("CPU time            [s]    : ", total)

#Plot the sollution
plot_solution( mesh, sol, outfile )