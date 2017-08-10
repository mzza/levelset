'''level set segmentation

	the code refer to the paper "distance regularized level set 
	evolution and its application to image segemntation" Chunming Li 2010 TIP

	2017.7.29
'''
from __future__ import division
import fuzzy_kmeans
import random
import numpy as np
from sympy import *
##################################

def random_region_init(X):
	''' select a constant rectangle region randomly
		Input:
		-----------
		X: image matrix

		returns :
		----------
		random_region : matrix
			the region label matrix, the value of pixel in the region is 1,while othes are 0
	'''
	x=random.sample(range(X.shape[0]),2)
	y=random.sample(range(X.shape[1]),2)
	x.sort()
	y.sort()
	random_region=np.zeros(X.shape)
	for i in range(X.shape[0]):
	 	for j in range(X.shape[1]):
	 		if (x[0]<=i<=x[1])&(y[0]<=j<=y[1]):
	 			random_region[i,j]=1
	 		else:
	 			random_region[i,j]=0
	return random_region

def region_init(X,InitRegion_state):
	'''get the init region 

		Inputs::
		------------
		X: the image matrix
		init_state : string,['random', 'fuzzy_kmeans']

		returns:
		---------
		region : matrix
			the region label matrix, the value of pixel in the region corsspond 1,while othes are 0

	'''
	if InitRegion_state=='random':
		region=random_region_init(X)
		return region
	else:
		pass

def LSF_init(X,InitRegion_state ,c):
	'''initialization of level set function
	Inputs:
	------------
	X ,init_state
	c:a constant for the binary step function

	returns:
	---------
	LST_init : matrix
		the discrete iniliazation of level set function
	'''
	region=region_init(X,InitRegion_state)
	LSF=np.where(region==1,c,(-1)*c)
	return LSF

def ZeroCross_set(phi,narrowband_old):
	'''find all the zero crosing ponits on the old narrowband
	 Input : 
	 ------
	 phi :matrix 
	  	the discrete level set function
	  narrowband_old :matrix
		the value of the pix in the narrowband is 1

	 Output:
	 --------
	 sets :matrix
	 	the value of zero crossing is 1,while othes are 0
	'''
	sets=np.zeros(phi.shape)
	for i in range(1,phi.shape[0]-1):
		for j in range(1,phi.shape[1]-1):
			if phi[i-1,j]*phi[i+1,j]<0:
				sets[i,j]=1
			if phi[i,j-1]*phi[i,j+1]<0:
				sets[i,j]=1

	sets=np.where(sets==0,2,sets)
	sets=np.where(sets==narrowband_old,1,0)

	return sets

def narrowband(phi,narrowband_old,r):
	''' get the new narrowband

	Input:
	------
	phi :matrix
	narrowband_old :matrix
		the value of the pix in the narrowband is 1
	r : int,default:1
		the radius of the narrowband squar

	Output:
	-------
	narrowband_new :matrix 
	'''
	narrowband_new=np.zeros(phi.shape)
	narrowband_update=np.zeros(phi.shape)

	sets=ZeroCross_set(phi,narrowband_old)
	for i in range(phi.shape[0]):
		for j in range(phi.shape[1]):
			if (sets[i,j]==1):
				narrowband_new[i,j-r:j+r]=1
				narrowband_new[i-r:i+r,j]=1

	for i in range(phi.shape[0]):
		for j in range(phi.shape[1]):
			if (narrowband_old[i,j]==1)|(narrowband_new[i,j]==1):
				narrowband_update[i,j]=1

	return narrowband_update


def Hamition(direction,phi,i,j):
	'''
		Inputs:
		-----------
		direction :string [ringht,left,up,down]

		phi :matrix
			the discrete level set function

		i,j:the dsicrete coordinate

		returns:
		-------
		a list :[diff_x,diff_y]
		'''
	if direction=='right':
		diff_x=phi[i+1,j]-phi[i,j]
		diff_y=(phi[i,j+1]-phi[i,j-1])/2
	elif direction=='left':
		diff_x=phi[i,j]-phi[i-1,j]
		diff_y=(phi[i,j+1]-phi[i,j-1])/2
	elif direction=='up':
		diff_x=(phi[i+1,j]-phi[i-1,j])/2
		diff_y=(phi[i,j+1]-phi[i,j])
	else :
		diff_x=(phi[i+1,j]-phi[i-1,j])/2
		diff_y=phi[i,j]-phi[i,j-1]

	return [diff_x,diff_y]

def diffusion_rate(direction,phi,i,j):
	'''
		Parameters:
		-----------
		direction :string [ringht,left,up,down]

		phi :matrix
			the discrete level set function

		i,j:the dsicrete coordinate
	'''
	nabla=Hamition(direction,phi,i,j)
	len_nabla=sqrt(nabla[0]**2+nabla[1]**2)

	if len_nabla>=1:
		diff_rate=(len_nabla-1)/len_nabla
	elif 0<len<1:
		diff_rate=sin(2*pi*len_nabla)/(2*pi*len_nabla)
	else:
		diff_rate=1

	return diff_rate


def len_term_rate(direction,phi,i,j):
	nabla=Hamition(direction,phi,i,j)
	len_nabla=sqrt(nabla[0]**2+nabla[1]**2)

	g=1/(1+len_nabla**2)
	if len_nabla==0:
		len_nabla=pow(10,-3)
	return g/len_nabla

def delta_epsilon(x,epsilon):
	if (x<=epsilon)|(x>-epsilon):
		return (1+cos(pi*x/epsilon))/(2*epsilon)
	else :
		return 0

def len_term_coef(phi,lambda_,epsilon,i,j):
	'''
	Inputs:
	--------
	lambda_,phi,epsilon,i,j
	Returns:
	-------
	coef :float
	'''
	return delta_epsilon(phi[i,j],epsilon)*lambda_

def area_term(phi,epsilon,i,j):
	'''
	Inputs :
	-------
	phi,alpha,i,j
	returns:
	-------
	area :
	the discrete area teem
	'''
	diff_x=(phi[i+1,j]-phi[i-1,j])/2
	diff_y=(phi[i,j+1]-phi[i,j-1])/2

	len_nabla=sqrt(diff_y**2+diff_x**2)

	g=1/(1+len_nabla**2)

	return g*delta_epsilon(phi[i,j],epsilon)

def PDE_diff(phi,delta_t,mu,narrowband,lambda_,epsilon,alpha):
	'''
	Input
	----------
	phi :the level set function ,only contain the x and y.
	delta_t :the time step
	mu :a const

	returns:
	--------
	fi_update:the updated fi

	'''
	delta_phi=np.zeros(phi.shape)

	for i in range(1,phi.shape[0]-1):
		for j in range(1,phi.shape[1]-1):
			if narrowband[i,j]==1:
				diff_potential =(diffusion_rate('right',phi,i,j)*(phi[i+1,j]-phi[i,j])\
					+diffusion_rate('up',phi,i,j)*(phi[i,j+1]-phi[i,j])\
					-diffusion_rate('left',phi,i,j)*(phi[i,j]-phi[i-1,j])\
					-diffusion_rate('down',phi,i,j)*(phi[i,j]-phi[i,j-1]))*mu*delta_t
			
				diff_len=((len_term_rate('right',phi,i,j)*(phi[i+1,j]-phi[i,j])\
					+len_term_rate('up',phi,i,j)*(phi[i,j+1]-phi[i,j])\
					-len_term_rate('left',phi,i,j)*(phi[i,j]-phi[i-1,j])\
					+len_term_rate('down',phi,i,j))*(phi[i,j]-phi[i,j-1]))*len_term_coef(phi,lambda_,epsilon,i,j)*delta_t
			
				diff_area=area_term(phi,epsilon,i,j)*alpha*delta_t

				delta_phi[i,j]=diff_potential+diff_len+diff_area
	phi_update=phi+delta_phi

	return phi_update

def PDE_update(phi,iter,delta_t,mu,r,lambda_,epsilon,alpha):
	''' start evolution
	 	Inputs:
	 	---------
	 	phi : matrix
	 		the level set function
	 	returns:
	 	--------
	 	LSF_result : matrix
	 		 the evolution result of levle set function
	'''
	sets=ZeroCross_set(phi,np.ones(phi.shape))
	narrowband_new=np.zeros(phi.shape)

	for i in range(phi.shape[0]): 
		for j in range(phi.shape[1]):
			if (sets[i,j]==1):
				narrowband_new[i,j-r:j+r]=1
				narrowband_new[i-r:i+r,j]=1

	iter_time=1
	while iter_time<=iter:
		phi=PDE_diff(phi,delta_t,mu,narrowband_new,lambda_,epsilon,alpha)
		narrowband_old=narrowband_new
		narrowband_new=narrowband(phi,narrowband_old,r)

		for i in range(phi.shape[0]):
			for j in range(phi.shape[1]):
				if (narrowband_old[i,j]!=1)&(narrowband_new[i,j]==1):
					phi[i,j]=r+1

		print("iteration times:%d"%iter_time)
		iter_time=iter_time+1
	return phi

class levelset(object):
	"""levelset segmentation
		
		paremeters :
		------------
		InitRegion_state : string,['random','fuzzy_kmeans'],default:'random'
			the way for initialization of region

		c :float 
			a constant for initilization the LST

		mu:float
			the coefficient of the potential term

		delta_t : float
			the time step ,mu*delt_t<=1/4

		iter : int
			the iteration times

		r :int ,dafault:1

		lambda_ :float
			THE coef of the len term

		epsilon : float 
			smooth the delta and Heaviside function

		alpha :
			the coef the the area term

		Attribute:
		----------
		LSF_result :numpy
			level set function
		labels: matrix
			the value of the pixel in the region is 1,while other are 0


	"""
	def __init__(self,c,mu,delta_t,InitRegion_state,iter,r,lambda_,epsilon,alpha):
		self.c=c
		self.InitRegion_state=InitRegion_state
		self.c=c
		self.mu=mu
		self.delta_t=delta_t
		self.iter=iter
		self.r=r
		self.lambda_=lambda_
		self.epsilon=epsilon
		self.alpha=alpha

	def fit(self,X):
		''' the level set evolution

			Parameter:
			---------
			X : the image matrix

			returns :
			---------
			self
		'''
		#LSF=LSF_init(X,self.InitRegion_state,self.c)
		LSF=np.zeros(X.shape)
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				LSF[i,j]=-3
		LSF[20:30,20:30]=3

		self.LSF_result=PDE_update(LSF,self.iter,self.delta_t,self.mu,self.r,self.lambda_,self.epsilon,self.alpha)
		return self

	def predict(self):
		''' predict the label

			returns:
			--------
			labels
		'''
		self.labels=np.where(self.LSF_result>=0,1,0)
		return self.labels
