import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

# setup hpyerparameters
options = {'c1':0.5, 'c2':0.3, 'w':0.9 }

# create bounds
max_bound = 5.12 * np.ones(2)
min_bound = -max_bound
bounds = (min_bound, max_bound)

# PSO instance
optimizer = ps.single.GlobalBestPSO( n_particles = 10, dimensions=2, options=options, bounds=bounds )

# performe optimization
cost, pos = optimizer.optimize( fx.rastrigin, iters=1000 )

