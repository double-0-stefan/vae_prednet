import math 
from utils import model_utils as mutils
import numpy as np
from torch.nn import functional as F
from torch.nn import ModuleList
from torch.autograd import Variable, Function
from torch.nn import Linear, Module, LSTM, Parameter, BatchNorm2d, Conv2d, ConvTranspose2d, Softmax, RNN, RNNBase, RNNCell, RNNCellBase
from torch import zeros,zeros_like, ones_like, cat, ByteTensor, FloatTensor, rand, log, sigmoid
from torch import add,tanh,squeeze,Tensor,float,stack, argmax
from torch import max as torchmax
from torch import min as torchmin
from utils import transform_utils as trutils
import torch 


class pc_conv_layer(Module):
	def __init__(self,p,l):
		super(pc_conv_layer, self).__init__()
		self.dim = p['ldim'][l]
		self.conv_trans = ConvTranspose2d(p['ldim'][l+1], p['ldim'][l], p['nf'][l])

		phi # descending prior, only comes in forward

		self.Sigma_u = torch.nn.Parameter(torch.zeros([self.dim,self.dim]))
		self.u = torch.nn.Parameter(torch.zeros([self.dim])) # unsure if need 1 here
		self.e = torch.nn.Parameter(torch.zeros([self.dim])) 

		self.dt = 1/p['inf_iter'] # integration step based on n interations

	def loss(self, phi):
		# Free energy
		F = 0.5*(	#- torch.logdet(Sigma_p)
					#- torch.matmul(
					#	torch.matmul(
					#		torch.transpose(phi-vp,0,1),
					#		torch.inverse(Sigma_p)
					#	),
					#	phi-vp
					#)
					# just one layer - take phi as fixed here
					- torch.logdet(self.Sigma_u)
					- torch.matmul(
						torch.matmul(						
							torch.transpose(self.u - self.conv_trans(phi),0,1),   # g(phi,Theta) = Theta*h(phi) = conv_trans(h(phi))
							torch.inverse(self.Sigma_u)				# leaving h aside for now
						),
						self.u - self.conv_trans(phi) # leaving h aside for now
					)
				)
					
		# vp (descending prediction from above this layer) have already gone through
		# h and trans_conv from layer above

		# torch.matmul:
		# If the first argument is 2-dimensional and the second argument is 
		# 1-dimensional, the matrix-vector product is returned.
		# nb matmul is association: (AB)C = A(BC)

	def inference(self, phi):
		for i in range(1/self.dt):
			F = loss(self,phi)

			F.backward()

            
			# get loss

			# equation 53 in bogacz 2017:
			# the 2 is the h' (h=x^2)
			
			#phi = phi + self.dt * (-error_p + error_u * dh * phi)

			#phi = phi + self.dt * (-error_i + dh(phi) .* transpose(Theta_im1) * error_im1 )

			# Equation 53 - can learn conv and T(conv) seperately as in spratling
			phi = phi + self.dt * (-error_i + dh(phi) .* self.conv_trans(error_im1))

			error_i = phi 


				# In the model of Rao and Ballard (1999) the sparse coding was achieved through
				# introduction of additional prior expectation that most φ i are close to 0, but the
				# sparse coding can also be achieved by choosing a shape of function h such that h (v i )
				# are mostly close to 0, but only occasionally significantly different from zero (Friston,
				# 2008).



			# i reckon dh is done on output of conv_trans
			# as h is done prior to weights (but could be other way i guess)

			# but could i just do descending then let backprop take care of ascending?
			# trans_conv -> conv/inverse would be taken care of automatically
			# objective F calculated on layer-by layer basis


			# better to start with conv_trans then get gradient?
			# actually, the trans_conv weights are Theta in Bogacz
			# function h could be anything - even an entire neural network
			# i guess that's one thing PC doesn't have - learning the functions
			# these are given in all of Friston's DEM examples
			# could import a function or have it learn one

	def learning(self,): 



	def forward(self, mean_phi, sigma_phi, p_above, pe_below, v1, v2):


		# Bogacz paper example:
		# function exercise3
		# 	v_p = 3; % mean of prior distribution of food size
		# 	Sigma_p = 1; 	% variance of prior distribution
		# 	Sigma_u = 1;  % variance of sensory noise
		# 
		# 	
		# 	u = 2; % observed light intensity
		# 	DT = 0.01;
		# 	MAXT = 5; % integration step
		# 	% maximum time considered
		# 	phi (1) = v_p;
		# 	% initializing the best guess of food size
		# 	error_p (1) = 0; % initializing the prediction error of food size
		# 	error_u (1) = 0; % initializing the prediction error of sensory input
		# 	for i = 2: MAXT/DT
			# 	phi(i) = phi(i -1) + DT * (- error_p (i -1) + error_u (i -1) * (2* phi(i -1)));
			# 	error_p (i) = error_p (i -1) + DT * (phi(i -1) - v_p - Sigma_p * error_p (i -1));
			# 	error_u (i) = error_u (i -1) + DT * (u - phi(i -1)^2 - Sigma_u * error_u (i -1));
		# 	end
		# 	plot ([ DT:DT:MAXT], phi , ’k ’);
		# 	hold on

		# p_above = prediction from above
		# pe_below = pe from below
		# y_lo = 
		# y_hi

		# top down 1: compare p_above, which is already in correct format, to y_hi

		pe_above = p_above - y_hi
		

		# bottom up 1: pe_below updates y_lo

		y_lo = y_lo * y_lo_precision - pe_below
		
		pred = self.conv_trans(desc_pred)
		pe   = y - pred


		asc_pe = self.conv(pe)
