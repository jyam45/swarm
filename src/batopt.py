# Swarm Interigence : Bat Optimizer

import numpy as np
import copy as cp
import math

# To create an animation gif
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image


# objective function ( Double-well Potential Energy V(x) )
def local_func( x ):
	# V(x) = a*x^4 + b*x^2 +c 
	# dV/dx = 4a*x^3 + 2b*x = 4a*x*(x^2+b/2a) -> b/2a = -d^2 -> d = sqrt(-b/2a)
	a =  1
	b = -2
	c =  1
	x2 = np.linalg.norm(x,ord=2)
	f = a*x2*x2 + b*x2 + c
	return f

def find_best(x):
	n = len(x)
	ybest = local_func(x[0])
	ibest = 0
	for i in range(1,n):
		ytmp = local_func(x[i])
		if ytmp < ybest :
			ybest = ytmp
			ibest = i
	return ibest

def func( x ):
	n = len(x)
	r = 0
	for i in range(0,n):
		r += local_func(x[i])
	return r	

def main():

	# loop controllers
	maxite       = 1000
	maxite_local = 100
	threshold    = 0.85 # under limit of roudness
	
	# scalars
	n    = 50    # number of bats
	dim  = 3     # size of dimension
	fmin = 0.00  # minimum frequency
	fmax = 1.00  # maximum frequency
	Amax = 10    # maximum roudness
	
	# list of vectors
	x = -1 + ( 1 + 1 ) * np.random.rand(n,dim) # initial bat's positions
	v = np.zeros((n,dim))                      # initial bat's velocities
	r = np.random.rand(n)                      # initial pulse emission rates
	A = Amax*np.random.rand(n)                 # initial roudness
	r0= cp.deepcopy(r)
	
	# constant parameters
	alpha  = 0.8                 # roundness decreasing rate (constant)
	gamma  = 0.5                 # pulse rate control parameter
	maxeps = 0.9                 # sound amplitude control parameter

	# figure	
	fig = plt.figure(figsize=(8,6))
	imgs= []
	
	# Find the best solution
	ibest = find_best(x)
	xbest = cp.deepcopy(x[ibest])
	
	# Bat Algorithm
	xnew = cp.deepcopy(x)
	vnew = cp.deepcopy(v)
	for t in range(0,maxite):
		# Generate new solution by adjusting frequency(xnew)
		beta = np.random.rand(n)
		f = fmin + ( fmax - fmin ) * beta
		for i in range(0,n):
			v[i] = v[i] + ( x[i] - xbest ) * f[i]
			xnew[i] = x[i] + v[i]
	
		Aavg = np.mean(A)
		ravg = np.mean(r)
		ybest= local_func(xbest)
		ytotal = func(x)
		rbest  = np.linalg.norm(xbest,ord=2)
		print(t, ybest, ytotal, rbest, Aavg, ravg)
		#print(t, ybest, ytotal, Aavg, ravg, xnew, v)
		#print(t, ybest, Aavg, ravg, ibest, xbest)
		#print(t, ybest, ytotal, Aavg, ravg)

		fig.clear()
		ax  = fig.add_subplot(111,projection='3d')	
		ax.scatter(x[:,0],x[:,1],x[:,2],color='blue')
		ax.scatter(xbest[0],xbest[1],xbest[2],color='red')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.set_xlim(-1.5,1.5)
		ax.set_ylim(-1.5,1.5)
		ax.set_zlim(-1.5,1.5)
		fig.text(0,0,'t={:6}'.format(t))
		buf = BytesIO()
		fig.savefig(buf)
		imgs.append(Image.open(buf))
		#fig.savefig('img/bat{:0>8}.png'.format(t))

		# stoping criteria
		if Aavg < threshold : break
	
		rand = np.random.randn();
	
		vnew = v
		for i in range(0,n):
			# Perform local search around global best best(xnew)	
			if rand > r[i]:
				# local search
				yold =  local_func(x[i])
				xibest= cp.deepcopy(x[i])
				for k in range(0,maxite_local):
					epsv  = np.random.randn(dim)
					leps  = np.linalg.norm(epsv,ord=2)
					if leps > maxeps : epsv = (maxeps/leps)*epsv
					#xtmp = cp.deepcopy(x[i]) + epsv*Aavg
					xtmp = cp.deepcopy(x[i]) + epsv*A[i]
					ynew = local_func(xtmp)
					if ynew < yold :
						yold  = ynew
						xibest = xtmp
				xnew[i] = xibest
				#print(i,rand,r[i],A[i],yold,xnew)
	
		for i in range(0,n):
			if rand < A[i]:
				yold = func(x)
				ynew = func(xnew)
				if ynew < yold:
					x[i] = xnew[i]
					r[i] = r0[i]*(1-math.exp(-gamma*t)) 
					A[i] = alpha*A[i]
				else:
					xnew[i] = x[i]
	
		# Update the best solution
		ibest = find_best(x)
		ybest_old = local_func(xbest)
		ybest_new = local_func(x[ibest])
		#print(ybest_old,xbest,ybest_new,x[ibest])
		if ybest_new < ybest_old :
			#xbest = cp.deepcopy(x[ibest])
			xbest = x[ibest]


	print(xbest)


	imgs[0].save('output.gif',save_all=True,append_images=imgs[1:],duration=100,loop=0)

	#anim = ani.ArtistAnimation(fig,imgs,interval=100,blit=True,repeat_delay=1000)
	#anim.save("optim.gif",writer="pillow")

	#ax.clear()
	#ax.scatter(x[:,0],x[:,1],x[:,2],color='blue')
	#ax.scatter(xbest[0],xbest[1],xbest[2],color='red')
	#fig.savefig("batopt_result.png")
	#plt.show()


main()

