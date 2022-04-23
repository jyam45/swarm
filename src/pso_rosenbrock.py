import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

# Rosenbrock function
def rosenbrock(x, a, b, c=0):
	f = ( a - x[:,0]) **2 + b * (x[:,1] - x[:,0]**2 ) **2 + c
	return f

# setup hpyerparameters
options = {'c1':0.5, 'c2':0.3, 'w':0.9 }

# create bounds
max_bound = 10 * np.ones(2)
min_bound = -max_bound
bounds = (min_bound, max_bound)

# PSO instance
optimizer = ps.single.GlobalBestPSO( n_particles = 10, dimensions=2, options=options, bounds=bounds )

# performe optimization
cost, pos = optimizer.optimize( rosenbrock, iters=1000, a=1, b=100, c=0 )

#kwargs={"a":1.0, "b":100.0, "c":0}
#cost, pos = optimizer.optimize( rosenbrock, iters=1000, **kwargs )

