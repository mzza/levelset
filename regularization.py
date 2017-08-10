from __future__ import division
import cv2
import random
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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

def potential_term(phi):
	'''calucate the Regulation term

		Input:phi
		Output: potential term value
	'''
	[phi_x,phi_y]=np.gradient(phi)
	s=np.sqrt(np.power(phi_x,2)+np.power(phi_y,2))
	dps=np.where(s<=1,np.sin(2*np.pi*s)/(2*np.pi),s-1)
	potential_value=div(np.multiply(dps,phi_x)-phi_x,np.multiply(dps,phi_y)-phi_y)+cv2.Laplacian(phi,cv2.CV_64F)
	return potential_value

def edge_indicate(phi):
	[diff_x,diff_y]=np.gradient(phi)
	f=np.sqrt(np.square(diff_x)+np.square(diff_y))
	g=1/(1+f)
	return g
	
def div(nx,ny):
	'''calculate the div
	'''
	[nxx,junk]=np.gradient(nx)
	[junk,nyy]=np.gradient(ny)
	f=nxx+nyy
	return f

def delta_epsilon(phi,epsilon):
	'''delta function
	'''
	temp=np.where(phi>=epsilon,epsilon,phi)
	return (1+np.cos(np.pi*phi/epsilon))/(2*epsilon)

def len_term(phi,epsilon,g):
	'''calculate the len_term value

		Input:phi,epsilon,g
		Output:len term value
	'''
	[phi_x,phi_y]=np.gradient(phi)
	s=np.sqrt(np.power(phi_x,2)+np.power(phi_y,2))
	smallnumber=pow(10,-10)
	nx=phi_x/(s+smallnumber)
	ny=phi_y/(s+smallnumber)

	curvature=div(nx,ny)
	[vx,vy]=np.gradient(g)
	LenTerm_value=np.multiply(vx,nx)+np.multiply(vy,ny)+g*curvature

	return np.multiply(LenTerm_value,delta_epsilon(phi,epsilon)) 

def area_term(phi,epsilon,g):
	'''
		Inputs :phi,epsilon,g
		returns:area value
	'''
	return np.multiply(g,delta_epsilon(phi,epsilon))

def PDE_update(X,LSF_init,iter,delta_t,mu,lambda_,epsilon,alpha):
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
	g=edge_indicate(X)
	phi=LSF_init
	x=np.arange(0,X.shape[0],1)
	y=np.arange(0,X.shape[1],1)
	x,y=np.meshgrid(y,x)
	fig=plt.figure(figsize=(16,8),dpi=100)
	iter_time=1
	while iter_time<=iter:
		if iter_time%10==1:
			ax=fig.add_subplot(4,5,int(iter_time//10)+1,projection='3d')
			surf=ax.plot_surface(x,y,phi,cmap=cm.coolwarm)

		phi=phi+delta_t*(mu*potential_term(phi)+lambda_*len_term(phi,epsilon,g)+alpha*area_term(phi,epsilon,g))

		print("iteration times:%d"%iter_time)
		if iter_time==iter:
			plt.show()
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
	def __init__(self,c,mu,delta_t,InitRegion_state,iter,lambda_,epsilon,alpha):
		self.InitRegion_state=InitRegion_state
		self.c=c
		self.mu=mu
		self.delta_t=delta_t
		self.iter=iter
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
		LSF_init=np.ones(X.shape)*(-self.c)
		LSF_init[25:50,20:50]=self.c

		self.LSF_result=PDE_update(X,LSF_init,self.iter,self.delta_t,self.mu,self.lambda_,self.epsilon,self.alpha)
		return self

	def predict(self):
		''' predict the label

			returns:
			--------
			labels
		'''
		self.labels=np.where(self.LSF_result<0,1,0)
		return self.labels
