import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

# setup hpyerparameters
options = {'c1':0.5, 'c2':0.3, 'w':0.9 }

# PSO instance
optimizer = ps.single.GlobalBestPSO( n_particles = 10, dimensions=2, options=options )

# performe optimization
cost, pos = optimizer.optimize( fx.sphere, iters=1000 )

