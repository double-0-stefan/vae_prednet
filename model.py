import os 
import math 
import yaml 
import numpy as np
from utils import model_utils as mutils 
from modules import * 
from os.path import join
from pprint import pprint, pformat
from scipy.io import savemat
from torch import nn
from logging import getLogger, shutdown
from torch.nn import MSELoss, Module, CrossEntropyLoss, L1Loss
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image
from torch import nn, optim, zeros, save, Tensor, FloatTensor, cuda, no_grad, isnan, load, min, max
from torch.autograd import Variable, set_detect_anomaly, detect_anomaly
from argparse import ArgumentParser
import torch
import glob 
import scipy.io as sio
from torchvision.utils import save_image
from torch.optim import Adam
from torch.optim import SGD
from gpuinfo import GPUInfo
import torch.nn.functional as F
# import torch_xla
# import torch_xla.core.xla_model as xm

class sym_conv2D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
		padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
		super(sym_conv2D, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.bias = bias
		self.padding_mode = padding_mode



		# number of unique weights over channels
		n_uwc = in_channels
		for m in range(in_channels):
			n_uwc = sum([m, n_uwc])
		
		w = torch.rand(int((self.kernel_size +1)/2), n_uwc)

		print(n_uwc)

		self.weight_values = nn.Parameter(w)

		self.generate_filter_structure()

	def generate_filter_structure(self):

		filter_weights = torch.zeros(self.out_channels,int(self.in_channels/self.groups),
			self.kernel_size, self.kernel_size)

		m = -1
		mm = -1
		full = -1

		for i in range(self.out_channels):
			m += 1
			mm = -1
			for n in range(m, self.out_channels):
				mm += 1

				# reversed so stuff outside of 'field' gets overwritten
				for j in reversed(range(int((self.kernel_size +1)/2))):
					# left/top side -  first so centre 'cross' gets overwritten
					filter_weights[i,n,j,:] = self.weight_values[j, full +mm]
					filter_weights[i,n,:,j] = self.weight_values[j, full +mm]

					# right/bottom side
					if j > 0:
						filter_weights[i,n,-j,:] = self.weight_values[j, full +mm]
						filter_weights[i,n,:,-j] = self.weight_values[j, full +mm]
			full += mm

		self.expanded_weight = filter_weights

	def forward(x):

		return F.conv2d(x, weight=self.expanded_weight, bias=self.bias, stride=self.stride,
			padding=self.padding, dilation=self.dilation, groups=self.groups)







class pc_conv_network(nn.Module):
	def __init__(self,p):
		super(pc_conv_network, self).__init__()

		l=0 # sb add for conveience/testing
		self.latents = sum(p['z_dim'][l:l+2]) 
		self.hidden	  = p['enc_h'][l] 

		self.err_plot_flag = True
		self.plot_errs = []
		self.enc_mode = False
		
		self.p = p
		self.p['ldim'] = [[1,1,32,32]]
		self.p['imdim_'] = self.p['imdim'][1]
		self.bs = p['bs']
		self.iter = p['iter']
		self.nlayers = p['nblocks']
		self.chan = p['chan']

		self.init_conv_trans(p)

		self.imchan = p['imchan']
		self.init_covariance(p)
		self.init_latents(p)
		self.optimizer = None
		print(self)
		
	def init_latents(self, p):

		# Initialise Distributions
		self.prior_dist, self.q_dist, self.x_dist= mutils.discheck(p)
		
		p['z_params']	= self.q_dist.nparams
		self.z_pc = nn.Parameter(torch.rand(self.p['bs'],self.latents*2),requires_grad = True)

		fc1= Linear(self.phi[-1].size(1), self.hidden)
		fc2 = Linear(self.hidden, int(self.latents*2))

		lin = []
		lin.append(fc1)
		lin.append(fc2)
		self.lin_up = nn.ModuleList(lin)

		# descending
		fc1 = Linear(int(self.latents), self.hidden)
		fc2 = Linear(self.hidden, self.phi[-1].size(1))
		lin = []
		lin.append(fc1)
		lin.append(nn.ReLU())
		lin.append(fc2)
		lin.append(nn.ReLU())
		self.lin_down = nn.Sequential(*lin)

	def init_conv_trans(self, p): 

		x = torch.zeros([p['bs'],p['imchan'],self.p['imdim_'],self.p['imdim_']])

		self.p['dim'] = []
		self.conv_trans = []
		phi = []
		Precision = []
		weights = []
		P_chol = []

		if self.p['include_precision']:

			## Precision as cholesky factor -> ensure symetric positive semi-definite
			a = torch.eye(p['imchan']*p['imdim_']*p['imdim_'])/10 + 0.001 * torch.rand(p['imchan']*p['imdim_']*p['imdim_'],p['imchan']*p['imdim_']*p['imdim_'])
			a = torch.cholesky(a)
			P_chol.append(nn.Parameter(a))
			Precision.append(nn.Bilinear(p['imchan']*p['imdim_']*p['imdim_'], p['imchan']*p['imdim_']*p['imdim_'], 1, bias=False))
			Precision[0].weight = nn.Parameter(a)

		# Create network
		for j in range(len(p['ks'])):
			conv_trans_block = []
			conv_block = []
			dim_block = []
			for i in reversed(range(len(p['ks'][j]))):
				if i == 0:
					if j == 0: # ie lowest level
						conv_trans_block.append(ConvTranspose2d(p['chan'][0][0], p['imchan'], p['ks'][j][i], stride=1, padding=p['pad']))
						conv_block.append(Conv2d(p['imchan'], p['chan'][0][0], p['ks'][j][i], stride=1, padding=p['pad']))
					else: 
						conv_trans_block.append(ConvTranspose2d(p['chan'][j][i], p['chan'][j-1][-1], p['ks'][j][i], stride=1, padding= p['pad']))
						conv_trans_block.append(nn.BatchNorm2d(p['chan'][j-1][-1]))
						conv_trans_block.append(nn.ReLU())
						conv_block.append(Conv2d(p['chan'][j-1][-1], p['chan'][j][i], p['ks'][j][i], stride=1, padding=p['pad']))
				else:
					conv_trans_block.append(ConvTranspose2d(p['chan'][j][i], p['chan'][j][i-1], p['ks'][j][i], stride=1, padding=p['pad']))
					conv_trans_block.append(nn.BatchNorm2d(p['chan'][j][i-1]))
					conv_trans_block.append(nn.ReLU())
					conv_block.append(Conv2d(p['chan'][j][i-1], p['chan'][j][i], p['ks'][j][i], stride=1, padding=p['pad']))
			

			for i in reversed(range(len(p['ks'][j]))):
				x = conv_block[i](x)
				dim_block.append(x.size(2))

			## CREATE PHI ABOVE EACH BLOCK ##
			phi.append(nn.Parameter((torch.rand_like(x)).view(self.bs,-1),requires_grad = True))

			self.conv_trans.append(nn.Sequential(*conv_trans_block))
			self.p['dim'].append(dim_block)

		self.phi = nn.ParameterList(phi)
		self.dim = self.p['dim']
		self.conv_trans = nn.ModuleList(self.conv_trans)
		# top level phi
		if not self.p['vae']:
			phi.append(nn.Parameter((torch.rand_like(x)).view(self.bs,-1)))

	def init_covariance(self, p): 

		Precision = []

		# will prob be better with class of Precision CNN that enforces requirements

		for i in range(len(self.phi)):

			if self.p['conv_precision']: 

				Precision.append(

					sym_conv2D(in_channels=self.p['chan'][i][-1], out_channels=self.p['chan'][i][-1], # do this as 2D over all channels

						# if kernel is odd, centroid is central pixel at current channel for input and output

						kernel_size= self.p['cov_size'][i],
						stride=1, 
						padding= (self.p['cov_size'][i]-1)/2,
						dilation=1, groups=1, bias=False, padding_mode='zero')
					)

		self.Precision = nn.ModuleList(Precision) 


	def logdet_block_tridiagonal(self, l):
		'''
		Implements Molinari 2008 method to find determinant of block tridiagonal matrix

		Still need to crack placement of weights in large precision atrix
		'''
		# Covariance is between components of phi
		# need to determine pattern of placement of conv weights
		# in order to calculate log determinant

		phi = self.phi[l].view(self.bs,self.p['chan'][l][-1],self.dim[l][-1],self.dim[l][-1])
		ps = phi.size(1), phi.size(2), phi.size(3)


		print(self.Precision[l].expanded_weight.permute([2,3,0,1]))
		# check order of input/output
		w = self.Precision[l].expanded_weight.permute([2,3,0,1])
		ws = w.size() # 5 5 64 64
		v = w.view(-1,w.size(2),w.size(3))
		vs = v.size() # 25, 64 64


		# TRY ANOTHER WAY
		b = torch.zeros([v.size(1),(1+v.size(0))*(v.size(1))])

		# size of (section of) phi is 25*  64
		# size of weights is 25*64*64
		
		# stuff to be added after centre (ie that is never on leading diag) - zero out centre
		# fi = torch.ones_like(v[:,:,0])
		# fi[int((vs[0]-1)/2),:] = 0
		# fi = fi == 1

		# tiled matrix with lots of zeros
		for j in range(vs[1]):
			# central - leading diag
			b[j,j] = v[int((ws[0]-1)/2),j,j]

			# all other weights inputting to output layer j
			# stuff to be added after centre (ie that is never on leading diag) - zero out centre
			fi = torch.ones_like(v[:,:,0])
			fi[int((vs[0]-1)/2),j] = 0
			fi = fi == 1
			# print(v.size())
			# print(b.size())
			vv = v[:,:,j]
			other_weights = vv[fi]
			# print(other_weights.numel())
			b[j, (j+1):j+1+other_weights.numel()] = other_weights

		print(b[0,:])
		# paste this into matrix, length(centres) number of times,
		# add zeros to make square
		# take upper triangle, do transpose -> raw materials for block is done!

		block = torch.cat([b, torch.zeros_like(b)],dim=1)

		for i in range(v.size(1)):

			extraL = torch.zeros([v.size(1)*2**i, v.size(1)*2**i])
			block = torch.cat([
				block,
				torch.cat([extraL,block[:,:-v.size(1)*2**i]],dim=1),
				],dim=0)

			if block.size(0) >= block.size(1):
				break

		print(block)
		# get matrices for determinant algorithm
		A = block[:v.size(0)*v.size(1),:v.size(0)*v.size(1)]
		A = A.triu()
		A += A.t()

		B = block[:v.size(0)*v.size(1), v.size(0)*v.size(1):v.size(0)*v.size(1)*2]
		print(B)
		
		C = B.t()
		print(C)
		B_inv = torch.inverse(B)

		# lengths
		m = A.size(0)
		n = round(self.phi[l].view(self.bs,-1).size(1)/A.size(0))


		Im_Zm = torch.cat([torch.eye(m), torch.zeros(m, m)])

		T1 = torch.stack([
			torch.cat([-A, -C]), 
			Im_Zm
			])

		T2 = torch.matrix_power(
			torch.stack([
				torch.cat([	-torch.mm([B_inv,A]), -torch.mm([B_inv,C]) ]),
				Im_Zm
				]),
			length*self.dim[l][-1] -2)

		T3 = torch.stack([
			torch.cat([-torch.mm([B_inv,A]), -B_inv]), 
			Im_Zm
			])

		T = torch.chain_matmul([T1,T2,T3])

		T11 = torch.rot90(torch.triu(torch.rot90(T,1,[1,0])), 1, [0,1])

		# need to set up weights to be symmetric around centres
		B1n = torch.matrix_power(B, n)

		logdetM = (-1)**(n*A.size(0)) + torch.logdet(T11) + torch.logdet(B1n)

		return logdetM


	def plot(self, i, input_image, plot_vars):
	
		z, pred = plot_vars
		pdir = os.path.join(self.p['plot_dir'], self.p['model_name'])
		matsdir = os.path.join(self.p['plot_dir'], self.p['model_name'], 'mats')
		os.makedirs(pdir) if not os.path.exists(pdir) else None
		os.makedirs(matsdir) if not os.path.exists(matsdir) else None
		save_image(pred[0].data.cpu(), pdir+'/p{}.png'.format(i))
		save_image(input_image[0].data.cpu(), pdir+'/b{}.png'.format(i))
		z	= z.data.cpu().numpy()
		sio.savemat(os.path.join(matsdir,'z_{}.mat'.format(i)), {'r':z})


	def reset(self):

		self.F = None
		self.F_last = None
		self.baseline = None

	def latent_sample(self):
		latent_sample = []
		norm_sample = self.q_dist.sample_normal(params=self.z_pc, train=self.training)
		latent_sample.append(norm_sample)
		z = torch.cat(latent_sample, dim=-1) 
		return z

	def vae_loss(self, curiter, z_pc):

		loss 			   = 0.		
		train_loss 		   = [] 
		train_norm_kl_loss = []
		train_cat_kl_loss  = [] 
		layer_loss 		   = 0.
	
		kloss_args	= (z_pc,   # mu, sig
					   self.p['z_con_capacity'][0], # anealing params
					   curiter)	# data size
					   
		norm_kl_loss = self.q_dist.calc_kloss(*kloss_args) #/ self.p['b']

		return norm_kl_loss#, metrics

	def decode(self,latent_samples, ff=0):
		# need to include Precisions
		#print(latent_samples.size())
		x = self.lin_down(latent_samples).view(-1, self.chan[-1][-1], self.dim[-1][-1], self.dim[-1][-1])

		for i in reversed(range(len(self.conv_trans))):
			x = self.conv_trans[i](x)

		self.pred = x
		return x

	def loss(self, i, learn=0):

		loss = 0.
		f = 0.
		kl_loss = None

		# if top layer - latents:
		if i == self.nlayers -1:
			# get kl_loss
			kl_loss  = self.vae_loss(self.iteration, self.z_pc)
			# get sample
			x = self.lin_down(self.latent_sample()).view(self.bs,-1)

		# if not top layer - phi from layer above:
		else:
			x = self.conv_trans[i+1](self.phi[i+1].view(self.bs, self.chan[i+1][-1], self.dim[i+1][-1], self.dim[i+1][-1])).view(self.bs,-1)
		
		# calculate PE
		if i == -1:
			PE = self.images - x
			# do image prediction if required
			if self.eval_:
				self.pred = x.view(self.bs,1,32,32)
		else:
			PE = self.phi[i] - x



		# calculate free energy - negative of usual formula so works with loss
		if self.p['include_precision']:
			# calculate inverse covariance matrix from cholesky factorisation
			# NB using transpose (rather than conjugate transpose) ensures P is symmetric
			P = torch.mm(torch.tril(self.P_chol[i+1]), torch.tril(self.P_chol[i+1]).t())

			f = 0.5*sum(sum(
				- torch.logdet(P) # -ve here because more precise = good (nb will need to balance over layers somehow)
				+ torch.mm(
					torch.mm(PE,P), PE.t()
					)
				))
			# for testing purposes:
			print(i)
			print(self.P_chol[i+1])
		
		elif self.p['conv_precision']:


			f = 0.5*sum(sum(
				

				- self.logdet_block_tridiagonal(i) # -ve here because more precise = good (nb will need to balance over layers somehow)
				

				+ torch.mm(PE, (self.Precision[i](PE.view(self.phi[i].size())) ).view(-1).t())


				))

			print(f)


		else:
			f = 0.5*sum(sum(
				torch.mm(PE, PE.t())
				))


		# update activation parameters
		if learn == 0:
			if i < self.nlayers - 1:
				self.opt_phi[i+1].zero_grad()
				f.backward()
				self.opt_phi[i+1].step()
			else:
				f += kl_loss
				self.opt_z_pc.zero_grad()
				f.backward()
				self.opt_z_pc.step()
		# update synaptic parameters
		else:
			if self.p['include_precision']:
				self.opt_P[i+1].zero_grad() 
			if i < self.nlayers - 1:
				self.opt_ct[i+1].zero_grad()
				f.backward()
				# print(i)
				# print(f)
				self.opt_ct[i+1].step()
			else:
				f += kl_loss
				self.opt_lin.zero_grad()
				f.backward()
				# print(i)
				# print(f)
				# print(kl_loss)
				self.opt_lin.step()
			if self.p['include_precision']:
				self.opt_P[i+1].step()
		return f

		
	def inference(self):
		for j in range(self.p['iter_outer']):
			for i in range(self.iter):
				loss = 0.
				
				if i < 9*self.iter/10:
					learn = 0
				else:
					learn = 1
				for l in range(-1, self.nlayers):
					loss += self.loss(l, learn)

				if i == 0:
					print(loss)
				
			print(loss)

	def forward(self, iteration, images, act=None, eval=False):
		
		# reset activations
		for i in range(len(self.phi)):
			self.phi[i] = nn.Parameter(torch.rand_like(self.phi[i])/100,requires_grad=True)
		self.z_pc = nn.Parameter(torch.rand_like(self.z_pc)/100,requires_grad=True)
		

		torch.set_printoptions(threshold=50000)
		self.eval_ = eval

		# if not self.optimizer:
			# self.optimizer = Adam(self.parameters(), lr=self.p['lr'])#, weight_decay=1e-5)
		self.opt_phi = [None] * len(self.phi)
		self.opt_ct = [None] * len(self.phi)
		self.opt_P = [None] * len(self.phi)
		for i in range(len(self.phi)):
			self.opt_phi[i] = Adam([self.phi[i]], lr=self.p['lr'])
			self.opt_ct[i]  = Adam(self.conv_trans[i].parameters(), lr=self.p['lr'])
			if self.p['include_precision']:
				self.opt_P[i]   = Adam([self.P_chol[i]], lr=self.p['lr'])
			elif self.p['bilinear_precision']:
				# self.opt_P[i]   = Adam([torch.tril(squeeze(self.Precision[i].weight))], lr=self.p['lr'])
				self.opt_P[i]   = Adam(self.Precision[i].parameters(), lr=self.p['lr'])
			elif self.p['conv_precision']:
				self.opt_P[i]   = Adam(self.Precision[i].parameters(), lr=self.p['lr'])
				print(self.Precision[i].parameters())

		self.opt_z_pc = Adam([self.z_pc], lr=self.p['lr'])
		self.opt_lin  = Adam(self.lin_down.parameters(), lr=self.p['lr'])
		
		self.iteration = iteration
		torch.cuda.empty_cache()
		
		self.F = 0

		self.iter = self.p['iter']

		if self.p['xla']:
			self.images = images.view(self.bs, -1).to(xm.xla_device())
		else:	
			self.images = images.view(self.bs, -1).cuda()

		print(iteration)
		self.inference()
		

		if eval:
			return self.z_pc, self.pred












































class pc_conv_network_old(nn.Module):
	def __init__(self,p):
		super(pc_conv_network, self).__init__()

		l=0 # sb add for conveience/testing
		self.latents = sum(p['z_dim'][l:l+2]) 



		self.hidden	  = p['enc_h'][l] 

		self.err_plot_flag = True
		self.plot_errs = []
		self.enc_mode = False
		
		self.p = p
		self.p['ldim'] = [[1,1,32,32]]
		self.p['imdim_'] = self.p['imdim'][1]
		self.bs = p['bs']
		self.iter = p['iter']
		self.nlayers = p['nblocks']
		self.chan = p['chan']

		self.init_conv_trans(p)

		# self.init_phi(p)
		# if p['conv_precision']:
		# 	self.init_conv_precision(p)
		# else:
		# 	self.init_precision(p)

		self.imchan = p['imchan']

		# self.kl_loss = None
		# self.F = None
		# self.F_last = None
		# self.baseline = None

		self.init_latents(p)

		self.optimizer = None

		print(self)
		
	def init_latents(self, p):

		# Initialise Distributions
		self.prior_dist, self.q_dist, self.x_dist= mutils.discheck(p)
		
		p['z_params']	= self.q_dist.nparams

		# layer configuration 
		# self.phi_top = nn.Parameter(torch.zeros(self.bs,self.phi[-1].size(1))) # doesn't have to be this
		# ascending
		self.z_pc = nn.Parameter(torch.rand(self.p['bs'],self.latents*2))

		# mu, logsigma = torch.chunk(params, 2, dim=-1)

		fc1= Linear(self.phi[-1].size(1), self.hidden)
		fc2 = Linear(self.hidden, int(self.latents*2)) # not divided by 2 here!

		lin = []
		lin.append(fc1)
		lin.append(fc2)
		self.lin_up = nn.ModuleList(lin)


		# descending
		fc1 = Linear(int(self.latents), self.hidden)
		fc2 = Linear(self.hidden, self.phi[-1].size(1))
		lin = []
		lin.append(fc1)
		lin.append(fc2)
		self.lin_down = nn.ModuleList(lin)

		
		
# phi.append(nn.Parameter(torch.rand(self.bs,self.latents*2))) 

		# self.has_con = p['nz_con'][l] is not None 
		# self.z_con_dim = 0;
		# if self.has_con: 
		# 	self.z_con_dim = p['nz_con'][l] 

		# self.z_dim = self.z_con_dim
		# enc_h = p['enc_h'][l] 

		# out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		# #self.imdim = np.prod(p['ldim'][l]) 
		# self.constrained = l < p['layers']-1 

		# # first fc - output of final conv layer
		# self.fc1 = Linear(len(self.phi[-1]), enc_h) 
		# if self.has_con: 
		# 	# features to continuous latent	 
		# 	self.fc_zp = Linear(enc_h, out_dim) 


	def init_conv_trans(self, p): # does conv, phi and precision
	
		x = torch.zeros([p['bs'],p['imchan'],self.p['imdim_'],self.p['imdim_']])

		self.p['dim'] = []
		self.conv_trans = []
		phi = []
		Precision = []
		weights = []
		P_chol = []


		# # Image level - needs Precision
		if self.p['include_precision']:

			## Precision as cholesky factor -> ensure symetric positive semi-definite
			a = torch.eye(p['imchan']*p['imdim_']*p['imdim_'])/10 + 0.001 * torch.rand(p['imchan']*p['imdim_']*p['imdim_'],p['imchan']*p['imdim_']*p['imdim_'])
			#a = torch.mm(a,a.t())
			a = torch.cholesky(a)
			# do this as vectorised lower tri?
			P_chol.append(nn.Parameter(a))

			Precision.append(nn.Bilinear(p['imchan']*p['imdim_']*p['imdim_'], p['imchan']*p['imdim_']*p['imdim_'], 1, bias=False))
			Precision[0].weight = nn.Parameter(a)

		#print(P_chol)

		for j in range(p['nblocks']):
			conv_trans_block = []
			conv_block = []
			dim_block = []

			for i in range(len(p['ks'][j]) ): # -1 because interaction between blocks done at start of higher block, not top of lower

			## DEFINE CONVOLUTIONS ##
				# Conv_trans and conv blocks
				if i == 0: # interaction with block below
					if j == 0: # ie lowest level
						conv_trans_block.append(ConvTranspose2d(p['chan'][0][0], p['imchan'], p['ks'][j][i], stride=1, padding=p['pad']))
						conv_block.append(Conv2d(p['imchan'], p['chan'][0][0], p['ks'][j][i], stride=1, padding=p['pad']))
					else: 
						conv_trans_block.append(ConvTranspose2d(p['chan'][j][i], p['chan'][j-1][-1], p['ks'][j][i], stride=1, padding= p['pad']))
						conv_block.append(Conv2d(p['chan'][j-1][-1], p['chan'][j][i], p['ks'][j][i], stride=1, padding=p['pad']))
				else:
					conv_trans_block.append(ConvTranspose2d(p['chan'][j][i], p['chan'][j][i-1], p['ks'][j][i], stride=1, padding=p['pad']))
					conv_block.append(Conv2d(p['chan'][j][i-1], p['chan'][j][i], p['ks'][j][i], stride=1, padding=p['pad']))
				
				x = conv_block[i](x)
				dim_block.append(x.size(2))

			## CREATE PHI ABOVE EACH BLOCK ##
			phi.append(nn.Parameter((torch.rand_like(x)).view(self.bs,-1)))

			if self.p['include_precision']:
				## Precision as cholesky factor -> ensure symetric positive semi-definite
				a = torch.eye(p['chan'][j][-1]*x.size(2)*x.size(2))/10 + 0.001 * torch.rand(p['chan'][j][-1]*x.size(2)*x.size(2),p['chan'][j][-1]*x.size(2)*x.size(2))
				#a = torch.mm(a,a.t())
				a = torch.cholesky(a)
				P_chol.append(nn.Parameter(a))

				Precision.append(nn.Bilinear(p['chan'][j][-1]*x.size(2)*x.size(2), p['chan'][j][-1]*x.size(2)*x.size(2), 1, bias=False))
				Precision[j+1].weight = nn.Parameter(a)

			
			

		## APPEND NEW BITS TO MAIN ##
			self.conv_trans.append(nn.ModuleList(conv_trans_block))
			self.p['dim'].append(dim_block)

		# top level phi
		if not self.p['vae']:
		# 	phi.append(nn.Parameter(torch.rand(self.bs,self.latents*2)))   # how does mean/sd work with this??
		# else:
			phi.append(nn.Parameter((torch.rand_like(x)).view(self.bs,-1)))

		#self.Precision = nn.ModuleList(Precision)
		self.P_chol = nn.ParameterList(P_chol)
		self.Precision = [None] * len(P_chol) # empty list for chol multiplications
		#self.weights = nn.ParameterList(weights)
		self.phi = nn.ParameterList(phi)
		self.dim = self.p['dim']
		self.conv_trans = nn.ModuleList(self.conv_trans)

		# 	#  phi same size as output as block - x sticks around to be input to next block
		# 	imdim = []
		# 	for i in range(len(p['ks'][j])):
		# 		print(i)
		# 		print(x.size())
		# 		x = conv_block[i](x)
		# 		imdim.append(x.size(2))
		# 	self.imdim.append(imdim)
	

		
		# # self.imdim = imdim
		# self.p = p
		# self.phi = nn.ParameterList(phi)
		# self.Precision = nn.ModuleList(Precision)
		# self.conv_trans = nn.ModuleList(self.conv_trans)
		#print(self.Precision)
		# if p['xla']:
		# self.conv_trans = ModuleList(
		# 	[ConvTranspose2d(p['chan'][i+1], p['chan'][i], p['ks'][i], 1,p['pad'][i])#.to(xm.xla_device())
		# 	for i in range(self.nlayers)])

		# else:
		# 	self.conv_trans = ModuleList(
		# 		[ConvTranspose2d(p['chan'][i+1], p['chan'][i], p['ks'][i], 1,p['pad'][i]).cuda() 
		# 		for i in range(self.nlayers)])

		

	def init_phi(self,p):
		# only need phi's where there are precisions
		conv = ModuleList(
			[Conv2d(p['chan'][i], p['chan'][i+1], p['ks'][i], 1,p['pad'][i])
			for i in range(self.nlayers)])
		if p['imdim']:
			x = torch.zeros(self.bs,1,p['imdim'],p['imdim'])
		else:
			x = torch.zeros(self.bs,1,32,32)
		phi = []
		imdim = [x.size(2)]
		for i in range(self.nlayers):
			x = conv[i](x) # mnist
			imdim.append(x.size(2))
			phi.append(nn.Parameter((torch.rand_like(x)).view(self.bs,-1)))
		phi.append(nn.Parameter((torch.rand_like(x)).view(self.bs,-1))) # top level
		if p['xla']:
			self.phi = nn.ParameterList(phi)#.to(xm.xla_device())
		else:
			self.phi = nn.ParameterList(phi).cuda()
		self.imdim = imdim
		
	def init_precision(self,p):
		# need one of these for each *prediction error*
		# so one more than the phi's - additional one at level of image itself
		if p['xla']:
			self.Precision = ModuleList(
				[nn.Bilinear(self.chan[i]*self.imdim[i]*self.imdim[i], self.chan[i]*self.imdim[i]*self.imdim[i], 1, bias=False)#.to(xm.xla_device())
				for i in range(self.nlayers+1)])
		else:
			self.Precision = ModuleList(
			[nn.Bilinear(self.chan[i]*self.imdim[i]*self.imdim[i], self.chan[i]*self.imdim[i]*self.imdim[i], 1, bias=False).cuda()
			for i in range(self.nlayers+1)])

		for i in range(self.nlayers+1):
			weights = torch.exp(torch.tensor(8.)) * torch.eye(self.chan[i]*self.imdim[i]*self.imdim[i]).unsqueeze(0)
			if p['xla']:
				self.Precision[i].weight = nn.Parameter(weights)#.to(xm.xla_device())
			else:
				self.Precision[i].weight = nn.Parameter(weights).cuda()

	def plot(self, i, input_image, plot_vars):
	
		z, pred = plot_vars
		pdir = os.path.join(self.p['plot_dir'], self.p['model_name'])
		matsdir = os.path.join(self.p['plot_dir'], self.p['model_name'], 'mats')
		
		os.makedirs(pdir) if not os.path.exists(pdir) else None
		os.makedirs(matsdir) if not os.path.exists(matsdir) else None
		
		
		# save_image(pred.data.cpu(), pdir+'/p{}.png'.format(i))
		save_image(pred[0].data.cpu(), pdir+'/p{}.png'.format(i))
		save_image(input_image[0].data.cpu(), pdir+'/b{}.png'.format(i))
		
		#if p['vae']:
		#	mu	= m['z_pc'][l].select(-1, 0).data.cpu().numpy()
		#	var = m['z_pc'][l].select(-1, 1).data.cpu().numpy()
		#	sio.savemat(os.path.join(matsdir,'mu_{}.mat'.format(i)), {'r':mu})
		#	sio.savemat(os.path.join(matsdir,'var_{}.mat'.format(i)), {'r':var})

		z	= z.data.cpu().numpy()
		sio.savemat(os.path.join(matsdir,'z_{}.mat'.format(i)), {'r':z})

		# alternative precision
		# can assume covariance will be the same everywhere
		# so weight sharing of some sort should work

		# 3D conv to same size? then x times (conv(x))' --> xAx'
		# need to look at form of weights for logdet

		# conv output size:
		#   o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
		#   p = (k+(k-1)*(d-1) - i + (o-1)s)/2
		# trans conv output size:
		#   o = (i -1)*s - 2*p + k + output_padding 
		#   p = ((i -1)*s - o + k + output_padding)/2

		# o = output
		# p = padding
		# k = kernel_size
		# s = stride
		# d = dilation

	def init_conv_precision(self,p):
		# only 1 channel as channels treated as depth
		self.Precision = ModuleList(
			[nn.Conv3d(1,1, kernel_size= (p['chan'][i],self.imdim[i],self.imdim[i]), stride=1, groups=1, bias=True, 
				padding= ((p['chan'][i]-1)/2, (self.imdim[i]-1)/2, (self.imdim[i]-1)/2)) 
			for i in range(self.nlayers+1)])

		# need to set weights so positive definite -> need Toeplitz matrix

		# or make 2D -> concat height/width into 1 dimensionS

		# start with 3D but only inter-channel connections -> 2D block matrix with lots of zeros

		#kernel bigger than image with padding?????

		self.Precision = ModuleList(
			[nn.Conv3d(1,1, kernel_size= (p['chan'][i]*2+1,self.imdim[i]*2+1), stride=1, groups=1, bias=True, 
				padding= ((p['chan'][i]-1)/2, (self.imdim[i]-1)/2, (self.imdim[i]-1)/2)) 
			for i in range(self.nlayers+1)])

	def reset(self):

		self.F = None
		self.F_last = None
		self.baseline = None

	def loss(self, i):
		loss = 0.0
		kl_loss = 0.0

		if self.p['vae']:

			# NEED TO ADD PRECISIONS IN HERE TOO! NO WONDER THEY WEREN'T BEING LEARNED #

			##### do block above #####

			# latents #
			if i == self.nlayers -1:
				# top block - where self.phi['i+1'] is latents

				# Encoding - p(z2|x) or p(z1 |x,z2)
				# self.z_pc = F.relu(self.lin_up[0](self.phi_top))
				# self.z_pc = F.relu(self.lin_up[1](self.z_pc))

				kl_loss  = self.vae_loss(self.iteration, self.z_pc) 

				# z_pc is the means and sds of the coordinates in latent sapce
				# could ake this like phi - with covariance matrix

				# extra PE here for updating z_pc over course of inference?

				# Latent Sampling
				latent_sample = []

				# Continuous sampling 
				norm_sample = self.q_dist.sample_normal(params=self.z_pc, train=self.training)
				latent_sample.append(norm_sample)

				z = torch.cat(latent_sample, dim=-1) 


				# Decoding - p(x|z)
				x = F.relu(self.lin_down[0](z))
				x = F.relu(self.lin_down[1](x))


			# or phi above #
			else:
				x = self.phi[i+1].view(self.bs, self.chan[i+1][-1], self.dim[i+1][-1], self.dim[i+1][-1])
			
				# process through if not latent
				for j in reversed(range(len(self.p['ks'][i+1]))):
					x = F.relu(self.conv_trans[i+1][j](x))
					#print(x)
			
			# get PE
			PE_1 = self.phi[i] - x.view(self.bs,-1) # this currently just phi[-1] -> vae -> phi[-1]
			# print('pe1')
			# print(x)
			# print(self.phi[i])


			##### do lower block #####
			x = self.phi[i].view(self.bs, self.chan[i][-1], self.dim[i][-1], self.dim[i][-1])

			# print('processing')
			# print(x)
			
			for j in reversed(range(len(self.p['ks'][i]))):
				# top - done below
				# if i == len(self.p['ks']) -1 and j = len(self.p['ks'][i]) -1:
				# 	x = F.relu(self.fc1(x))
				# 	x = F.relu(self.fc2(x))
				# 	x = F.relu(self.conv_trans[i][j](x))

				x = F.relu(self.conv_trans[i][j](x))
				# print('processing')
				# print(x)

			if i == 0:
				PE_0 = self.images   - x.view(self.bs,-1)
				# print(sum(sum(PE_0)))
				# ffs = (x.view(self.bs,-1))
				# print('pe0')
				# print(x)
				# print(self.images.view(self.bs,1,32,32)[0,0,11:15,11:15])

				# print(sum(sum(x.view(self.bs,-1))))
				# print(sum(sum(self.images)))
				if self.eval_:
					self.pred = x.view(self.bs,1,32,32)
			else:
				PE_0 = self.phi[i-1] - x.view(self.bs,-1)

			

		# Standard version
		else:
			# do block
			x = self.phi[i].view(self.bs, self.chan[i][-1], self.dim[i][-1], self.dim[i][-1])
			for j in reversed(range(len(self.p['ks'][i]))):
				
				x = self.conv_trans[i][j](F.relu(x))
				# print(x)

			if i == 0:

				PE_0 = self.images   - x.view(self.bs,-1)

			else:
				PE_0 = self.phi[i-1] - x.view(self.bs,-1)

			# do block above
			if i == self.nlayers-1:
				PE_1 = self.phi[i] - self.phi[i+1]
			else:
				x = self.phi[i+1].view(self.bs, self.chan[i+1][-1], self.dim[i+1][-1], self.dim[i+1][-1])
				for j in reversed(range(len(self.p['ks'][i+1]))):
					x = self.conv_trans[i+1][j](F.relu(x))
				PE_1 = self.phi[i] - x.view(self.bs,-1)
		


		# if i == 0:
		# 	self.PE_0 = self.images   - (self.conv_trans[i](F.relu(self.phi[i].view(self.bs, self.chan[i+1], self.imdim[i+1], self.imdim[i+1])))).view(self.bs,-1)
		# else:
		# 	self.PE_0 = self.phi[i-1] - (self.conv_trans[i](F.relu(self.phi[i].view(self.bs, self.chan[i+1], self.imdim[i+1], self.imdim[i+1])))).view(self.bs,-1)

		# if i == self.nlayers-1:
		# 	self.PE_1 = self.phi[i] - self.phi[i+1]
		# else:
		# 	self.PE_1 = self.phi[i] - (self.conv_trans[i+1](F.relu(self.phi[i+1].view(self.bs, self.chan[i+2], self.imdim[i+2], self.imdim[i+2])))).view(self.bs,-1)


		# if self.p['conv_precision']:
		# 	self.F += - 0.5*(
		# 		# logdet cov = -logdet precision
		# 		  torch.logdet(torch.squeeze(self.Precision[i+1].weight))

		# 		- sum(torch.matmul(self.PE_1, 
		# 			(self.Precision[i+1](self.PE_1.view(self.bs,1,self.chan[i+1],self.imdim[i+1],self.imdim[i+1]))).view(self.bs,-1)))

		# 		+ torch.logdet(torch.squeeze(self.Precision[i].weight))

		# 		- sum(torch.matmul(self.PE_0, 
		# 			(self.Precision[i](self.PE_0.view(self.bs,1,self.chan[i],self.imdim[i],self.imdim[i]))).view(self.bs,-1)))
		# 		)
		#else:

		

		# self.F +=  0.5*(
		# 	# logdet cov = -logdet precision
		# 	- torch.logdet(P1)

		# 	+ torch.matmul(torch.matmul(PE_1,P1),PE_1.t())

		# 	- torch.logdet(P0)

		# 	+ torch.matmul(torch.matmul(PE_0,P0),PE_0.t())
		# 	)
		


		# normalise (so equal precision at all levels)
		#ratio = sum(sum(torch.matmul(PE_1,PE_1.t())))/sum(sum(torch.matmul(PE_0,PE_0.t())))
		#print(ratio)
		# print(sum(sum(torch.matmul(PE_1,PE_1.t()))))
		# print(sum(sum(torch.matmul(PE_0,PE_0.t()))))

		#print(self.phi[0])  - issue is precision-weighting!!

		if not self.p['include_precision']:
			# print(i)
			# print(self.nlayers -1)
			# if i < self.nlayers -1 :

			f =  0.5*sum(sum((
				# logdet cov = -logdet precision
				#- torch.logdet(P1)

				torch.matmul(PE_1,PE_1.t())

				#- torch.logdet(P0)

				+ torch.matmul(PE_0,PE_0.t())
				)))
			# else:
			# 	f =  0.5*sum(sum((
			# 		# logdet cov = -logdet precision
			# 		#- torch.logdet(P1)

			# 		# torch.matmul(PE_1,PE_1.t())

			# 		#- torch.logdet(P0)

			# 		torch.matmul(PE_0,PE_0.t())
			# 		)))
				# print(sum(sum(PE_0)))
			print(f) 
				# print('weights')
				# print(sum(sum(self.conv_trans[0][0].weight)))



		else:
			if not self.update_phi_only or self.i == 0:
			## for Cholesky-based precision
			# tril: ensure upper tri doesn't get involved
				P1 = torch.mm(torch.tril(self.P_chol[i+1]), torch.tril(self.P_chol[i+1]).t())
				P0 = torch.mm(torch.tril(self.P_chol[i]), torch.tril(self.P_chol[i]).t())

				self.Precision[i+1] = P1
				self.Precision[i]   = P0

			f =  0.5*sum(sum((
				# logdet cov = -logdet precision
				- torch.logdet(P1)

				+ torch.matmul(torch.matmul(PE_1,P1),PE_1.t())

				- torch.logdet(P0)

				+ torch.matmul(torch.matmul(PE_0,P0),PE_0.t())
				)))


		loss = f + kl_loss
		loss.backward()
		if self.i == 0 or self.i == self.iter - 1:
			print(loss)
		
		return loss

		
		#else:
			# self.F +=  0.5*(
			# 	# logdet cov = -logdet precision
			# 	- torch.logdet(torch.squeeze(self.Precision[i+1].weight))

			# 	+ sum(self.Precision[i+1](PE_1, PE_1))

			# 	- torch.logdet(torch.squeeze(self.Precision[i].weight))

			# 	+ sum(self.Precision[i](PE_0, PE_0))
			# 	)



		# self.F += - 0.5*(
		# 	# logdet cov = -logdet precision
		# 	  torch.logdet(torch.squeeze(self.Precision[i+1].weight))

		# 	- sum(self.Precision[i+1](PE_1, PE_1))

		# 	+ torch.logdet(torch.squeeze(self.Precision[i].weight))

		# 	- sum(self.Precision[i](PE_0, PE_0))
		# 	)


	def vae_loss(self, curiter, z_pc):

		loss 			   = 0.		
		train_loss 		   = [] 
		train_norm_kl_loss = []
		train_cat_kl_loss  = [] 
		layer_loss 		   = 0.

		
		#err_loss = self.mse(image, pred)			
	
		kloss_args	= (z_pc,   # mu, sig
					   self.p['z_con_capacity'][0], # anealing params
					   curiter)	# data size
					   
		norm_kl_loss = self.q_dist.calc_kloss(*kloss_args) #/ self.p['b']
		
		# kloss_args	 = (z_pd,  # alpha
		# 				self.p['z_dis_capacity'][0],  # anneling params 
		# 				self.p['nz_dis'][0], # nclasses per categorical dimension
		# 				curiter)	# data size
					  
		# cat_kl_loss = self.cat_dist.calc_kloss(*kloss_args) #/ self.p['b']

		# if self.p['elbo_loss']:	
		# 	layer_loss = norm_kl_loss + cat_kl_loss + err_loss
		# else:
		# 	layer_loss = err_loss 

		# loss += layer_loss

		# if self.p['dataset'] == 'mnist':
		# 	loss /= np.prod(self.p['imdim'][1:])
		
		
		# metrics = (err_loss.item(), norm_kl_loss.item(), cat_kl_loss.item())

		return norm_kl_loss#, metrics

	def decode(self,latent_samples, ff=0):
		# need to include Precisions
		#print(latent_samples.size())
		x = F.relu(self.lin_down[0](latent_samples)) # get rid of 'top phi', call z or somewthign
		x = F.relu(self.lin_down[1](x))

		x = x.view(-1, self.chan[-1][-1], self.dim[-1][-1], self.dim[-1][-1])
		for i in reversed(range(len(self.conv_trans))):

			for j in reversed(range(len(self.conv_trans[i]))):
				x = F.relu(self.conv_trans[i][j](x))

		return x


		
	def inference(self):

		# for l in range(0, self.nlayers):
		# 	for m in range(len(self.conv_trans[l])):
		# 		self.conv_trans[l][m].requires_grad_(False)
		# 		#self.Precision[l].requires_grad_(False)
		# 	self.P_chol[l].requires_grad_(True)
		# 	self.phi[l].requires_grad_(True)
		#self.optimizer.lr = self.p['lr']
		
		self.update_phi_only = True
		self.phi.requires_grad_(True)
		self.z_pc.requires_grad_(True)
		# self.z_pc.requires_grad_(True)
		self.lin_up.requires_grad_(False)
		self.lin_down.requires_grad_(False)
		self.conv_trans.requires_grad_(False)

		for i in range(self.iter):
			self.i = i
			if i == self.iter - 1 and self.eval_ == False:
				self.update_phi_only = False
				# learn = 1
				self.phi.requires_grad_(False)
				self.z_pc.requires_grad_(False)
				self.conv_trans.requires_grad_(True)
				# self.z_pc.requires_grad_(False)
				self.lin_up.requires_grad_(True)
				self.lin_down.requires_grad_(True)

			# self.optimizer.zero_grad()
			# self.F_old = self.F
			# self.F = 0#nn.Parameter(torch.zeros(1))
			# self.phi_old = self.phi

			for l in range(self.nlayers):
				
				self.optimizer.zero_grad()
				loss = self.loss(l)
				self.optimizer.step()
			# predictive coding and reconstruction loss

			# self.kl_loss = 0
			# if self.p['vae']:
			# 	# KL loss   -> z_pc is encoded latents - phi uppermost in this implementation?? HOW IS MEAN/SD MANAGED?
			# 	latent_sample = []
			# 	# Continuous sampling 
			# 	norm_sample = self.q_dist.sample_normal(params=self.z_pc, train=self.training)   # may need to implement self.training
			# 	latent_sample.append(norm_sample)
			# 	self.z = torch.cat(latent_sample, dim=-1)
			# 	self.kl_loss  = self.vae_loss(self.iteration, self.z_pc) 
			
				# if i > 0:
				# 	if self.F >= self.F_old:
				# 		# self.F = self.F_old
				# 		# self.phi = self.phi_old

				# 		self.i += 1
				# #print(self.phi[0])
				#print(self.phi[1])
				# print(torch.max(sum(self.phi[0])))

				#print(self.phi[0])
				#print(self.z_pc)



				#print(self.kl_loss)
				# print(self.F.size())
				#self.F = self.F + self.kl_loss #torch.sum(torch.tensor(self.kl_loss))
				# if i < self.iter-1:
				# 	self.F.backward(retain_graph=True)
				# else:

				# self.F.backward()
				# self.F.detach()

				#print(i)
				#print(self.F)
				# xm.optimizer_step(self.optimizer)#.step()
				# self.optimizer.step()
			#print(self.F)
			# end inference if starting to diverge
		#print(loss)
					#break
			#self.i = i
			# print(self.F)
			# print(torch.sum(self.images-self.F_old))



		# self.learn()

		

	def learn(self):

		# update Precision weights
		# for l in range(0,self.nlayers):
		# 	self.Precision[l].weight = torch.matrix_power(self.weights[l],2)

		#self.weights.requires_grad_(True)
		self.conv_trans.requires_grad_(True)
		# self.Precision.requires_grad_(True)
		for i in range(len(self.P_chol)):
			#self.P_chol[i].requires_grad_(True)
			self.phi[i].requires_grad_(False)
		#self.optimizer.lr = 0.001

		self.optimizer.zero_grad()
		self.F_old = self.F
		self.F = 0
		# last_Precision = self.Precision
		# last_conv_trans = self.conv_trans

		for l in range(0,self.nlayers):
			self.loss(l,learn=1)

		# if torch.isnan(self.F):
		# 	self.Precision = last_Precision
		# 	self.conv_trans = last_conv_trans
		# 	self.optimizer = Adam(self.parameters(), lr=self.p['lr']/100, weight_decay=1e-5)
		# 	#self.inference()
		# 	self.learn()
		# 	self.optimizer = Adam(self.parameters(), lr=self.p['lr'], weight_decay=1e-5)
		# 	return
		self.F=torch.sum(self.F)	

		self.P_chol_old = self.P_chol

		self.F.backward()
		# xm.optimizer_step(self.optimizer, barrier=False)
		self.optimizer.step()

		# if i > 0:
		# 	if self.F >= self.F_old:
		# 		self.F = self.F_old
		# 		self.P_chol = self.P_chol_old
				

	def forward(self, iteration, images, act=None, eval=False):
		torch.set_printoptions(threshold=50000)

		#self.cuda()
		self.eval_ = eval

		if not self.optimizer:
			self.optimizer = Adam(self.parameters(), lr=self.p['lr'])#, weight_decay=1e-5)
		#self.optimizer = optimizer = optim.SGD(self.parameters(), lr=self.p['lr'])

		#self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
		# self.optimizer = torch.optim.RMSprop(params=self.parameters(),lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=True)
		# self.optimizer = torch.optim.ASGD(params=self.parameters(), lr=0.0001, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
		
		# random latents
		#self.z_pc = nn.Parameter(torch.rand(self.bs,self.latents))
		#print(sum(sum(sum(images))))
		self.iteration = iteration
		# del self.F

		torch.cuda.empty_cache()
		
		self.F = 0
		# self.F_last = self.F
		#self.lin.requires_grad_(True)
		# self.fc2.requires_grad_(True)
		if iteration == 0:
			self.iter = 1
		else:
			self.iter = self.p['iter']

		# images.half()
		if self.p['xla']:
			self.images = images.view(self.bs, -1).to(xm.xla_device())
		else:	
			self.images = images.view(self.bs, -1).cuda()

		# put weights into bilinear for inference and see if faster (no update done)
		for i in range(len(self.phi)):
			# reset wactivations
			# self.z_pc = torch.zeros(self.bs,2*self.latents)
			self.phi[i] = nn.Parameter(torch.rand_like(self.phi[i]))
		# 	#self.Precision[i].weight = torch.nn.Parameter(torch.mm(self.P_chol[i],self.P_chol[i].t()).unsqueeze(0))
		self.z_pc = nn.Parameter(torch.rand_like(self.z_pc))
		
		self.inference()
		print(iteration)
		# print(self.phi[0])
		# print(self.images)
		# print(self.i)
		# print(self.kl_loss)
		#print(self.F)
		# print(self.phi[-1])
		# if learn == 1:
		# print(GPUInfo.gpu_usage())

		if eval:
			return self.z_pc#, self.pred

#		del self.optimizer
		
























class GenerativeModel(Module):
	# Use this model for inference 

	def __init__(self, p):
		super(GenerativeModel,self).__init__()
		
		self.p = p
		self.log	  = getLogger('model')
		self.err_log  = getLogger('errlog')
		
		self.mse  = MSELoss().cuda() if p['gpu'] else MSELoss()
		self.xent = CrossEntropyLoss().cuda() if p['gpu'] else CrossEntropyLoss()
		
		# Initialise Distributions
		self.prior_dist, self.q_dist, self.x_dist, self.cat_dist = mutils.discheck(p)
		
		p['z_params']	= self.q_dist[0].nparams
		
		for l in range(p['layers']):
			self.register_buffer('prior_params_{}'.format(l), zeros(sum(p['z_dim'][l:l+2]), 2))

		# Calc RF masks for dataset 
		masks, p = mutils.calc_rf(p)
		full_mask, reduce_mask = masks

		# Initialise Modules - learnable
		if p['conv'] and p['exp_name'] == 'stl_freq':
			self.f_enc = mutils.ListModule([ConvEncoder_hi(p,0), ConvEncoder_lo(p,1)])
			# need to amend loss function to run gaussian smoother on original image
			# will need 2 loss functions
			self.g_dec = mutils.ListModule(*[ConvDecoder(p,l)  for l in range(p['layers'])])
			self.link = mutils.ListModule([Link_lo_hi(p,0)])
		elif p['conv']:
			self.f_enc = mutils.ListModule(*[ConvEncoder(p,l)  for l in range(p['layers'])])
			self.g_dec = mutils.ListModule(*[ConvDecoder(p,l)  for l in range(p['layers'])])
		else:
			self.f_enc = mutils.ListModule(*[Encoder(p,l)  for l in range(p['layers'])])
			self.g_dec = mutils.ListModule(*[Decoder(p,l)  for l in range(p['layers'])])
		self.f_enc = self.f_enc.cuda() if p['gpu'] else self.f_enc
		
		if p['use_lstm']:
			self.lstm  = mutils.ListModule(*[DynamicModel(p,l) for l in range(p['layers'])])
			if p['foveate']:
				self.a_net	= ActionNet(sum(p['z_dim'][0:2]),sum(p['z_dim'][l:l+2]),p['action_dim'], p['lstm_l'])
				self.retina = Retina(p['patch_size'], p['num_patches'], p['glimpse_scale'])
				
		# Initialise Modules - non-learnable
		self.g_obs = mutils.ListModule(*[ObsModel(p,masks,l) for l in range(p['layers'])])
		self.e_err = mutils.ListModule(*[ErrorUnit(p,full_mask,l) for l in range(p['layers'])])
		
		# Initialise Tensors
		self.reset()

		if p['gpu']:
			self.cuda()
		
	def plot(self,i,b):
		# pass all tensors to (utils) plot function
		m = {k:v for k, v in self.__dict__.items() if not (k.startswith('__') and k.endswith('__'))}
		mutils.plot(m,self.p,i,b)	
	
	def reset(self):
		# reset / init all model variables 
		# call before each batch
		self.error = [zeros(self.p['b'], *ld) for ld in self.p['ldim']]

		self.targets = [zeros(self.p['b'], *ld) for ld in self.p['ldim']]
		
		self.pred = [zeros(self.p['b'], *ld) for ld in self.p['ldim']]

		self.obs = [zeros(self.p['b'], *ld) for ld in self.p['ldim']]

		self.mod_elbo = [zeros(self.p['b'], 1) for l in range(self.p['layers'])]

		self.elbo = [zeros(self.p['b'], 1) for l in range(self.p['layers'])]

		self.z_pd = [] ; self.z_pc	= []
		for l in range(self.p['layers']):
			self.z_pd.append(zeros(self.p['b'], 1, sum(self.p['nz_dis'][l])))
			self.z_pc.append(zeros(self.p['b'], 1, sum(self.p['nz_con'][l:l+2])*2))
		
		
		self.z = [zeros(self.p['b'], sum(self.p['z_dim'][l:l+2])) for l in range(self.p['layers'])]

		self.z_pred = [zeros(self.p['b'], sum(self.p['z_dim'][l:l+2])) for l in range(self.p['layers'])]
		
		if self.p['err_noise']:
			self.noise = [Variable(torch.zeros(self.p['b'], *ld)) for ld in self.p['ldim']]
		
		self.rnn_lossvec = [0, 0]
		
		if self.p['use_lstm']:
			for l in range(self.p['layers']):
				self.lstm[l].reset()
			if self.p['foveate']:
				self.action	  =	 zeros(self.p['b'], self.p['action_dim'], self.p['n_actions'])
				self.action = self.action.cuda() if self.p['gpu'] else self.action
				
	def del_vars(self):
		# prob won't need this 
		del self.error,	self.pred, self.obs, self.x_p
		del self.mod_elbo, self.elbo	  
		del self.z_p, self.z		  
		
	def _get_prior_params(self, batch_size, layer):
		prior_params = getattr(self, 'prior_params_{}'.format(layer))
		expanded_size = (batch_size,) + prior_params.size()
		prior_params = Variable(prior_params.expand(expanded_size), requires_grad=True)
		prior_params = prior_params.cuda() if self.p['gpu'] else prior_params
		return prior_params
	
	def encode(self, up, down, l_r, image = None):

		# encode - p(z2|x) or p(z1 |x,z2)


		self.z_pc[l_r], self.z_pd[l_r] = self.f_enc[l_r](up, down)
		
		cdist = self.q_dist[l_r].sample(params=self.z_pc[l_r])
		# Dfferentiaable sampling of categorical latents
		ddist = self.cat_dist[l_r].gumbel_softmax(self.z_pd[l_r])
		
		#if self.p['use_lstm']:
		#	self.z[l_r]	  = torch.cat((cdist, ddist), dim=-1).detach()
		#else:
		#	self.z[l_r]	  = torch.cat((cdist, ddist), dim=-1)
		
		return torch.cat((cdist, ddist), dim=-1)
	
	
	def decode(self, z, l):
		return self.g_dec[l](z).data
	
	def forward(self, bottom_up_input, eval=False): # when model called bottom_up_input is the data
		
		time_taken = False ; _rnn_loss = 0.; rnn_metrics = [[], []]

		# low/high frequency split: 2nd level encodes/decodes low-pass filtered greyscale version of input image (magnocellular)
		# provides prior to 1st level high frequency bit
		

		for t in range(self.p['model_inner']):
			bu = bottom_up_input[:,t] #without t?
			#image = bu

			# foveate input 
			if self.p['foveate']:
				bu = self.retina.foveate(bu, self.action)
				bottom_up_input[:,t] = bu # for plotting

			# Top down
			for l_r in reversed(range(self.p['layers'])):
				
				# low freq encoder potentially has prediction errors from two sources: the high freq one and the image
				# is there a way to make those from the high frequency very low precision until later in training
				# or similar in all cases so uninformative - annealing params?
				
				if self.p['exp_name'] == 'stl10_freq' and l_r == 1:
					#bu = self.error[l_r] if self.p['prednet'] else self.pred[l_r]
					# presuming that the below is the image at the start:
					bu = self.error[0] if self.p['prednet'] else self.pred[0]
				else:
					bu = self.error[l_r] if self.p['prednet'] else self.pred[l_r] # does this imply that non-prednet heirarchical model possible in this setup?

				# this is the same regardless
				td = self.z[l_r+1] if l_r < self.p['layers']-1 else None
				
				#if self.p['exp_name'] == 'stl10_freq' and l_r == 1:
				#	self.z[l_r] = self.encode(bu, td, l_r, image)
				#else:
				self.z[l_r] = self.encode(bu, td, l_r)
				
				# predict next hidden state
				if self.p['use_lstm']:
					
					if t == self.self.p['model_inner']-1:
						_rnn_out = self.rnn_loss(l)
						_rnn_loss += _rnn_out[0]
						self.rnn_lossvec[0] += _rnn_out[1][0]
						self.rnn_lossvec[1] += _rnn_out[1][1]
							
					self.z_pred[l_r], done = self.lstm[l_r](self.z[l_r], self.action)
					
				# decode - p(x|z)
				self.pred[l_r] = self.g_dec[l_r](self.z[l_r])

			# Bottom up
			for l in range(self.p['layers']):
				
				if p['exp_name'] == 'stl10_freq' and l == 1:
					self.obs[l] = self.g_obs[l](bu if l<2 else self.error[l-1])
				else:
					self.obs[l] = self.g_obs[l](bu if l==0 else self.error[l-1])

				if l == 1 and self.p['exp_name'] == 'stl10_freq' :
					self.error[l] = self.e_err[l](self.obs[l], self.pred[l], self.noise[l], 1)
				else:
					self.error[l] = self.e_err[l](self.obs[l], self.pred[l], self.noise[l])

			# next saccade location
			if self.p['foveate']:
				#print('action in ', self.action.shape) 
				self.action		 = self.a_net(self.z[0], self.lstm[l].lstm_h, self.action)
				#print('action out ', self.action.shape) 
				
		if eval:
			return (self.z_pc, self.z_pd), self.pred

class TransitionModel(Module):
	def __init__(self, p):
		super(TransitionModel,self).__init__()
		
		self.p = p
		
		self.mse  = MSELoss().cuda() if p['gpu'] else MSELoss()
		self.xent = CrossEntropyLoss().cuda() if p['gpu'] else CrossEntropyLoss()
		
		# Initialise Distributions
		self.lstm  = mutils.ListModule(*[DynamicModel(p,l) for l in range(p['layers'])])
		if p['foveate']:
			self.a_net	= ActionNet(sum(p['z_dim'][0:2]),sum(p['z_dim'][0:2]),p['action_dim'], p['lstm_l'])
		
		# Initialise Tensors
		self.reset()

		if p['gpu']:
			self.cuda()
		
	def reset(self):
		# reset / init all model variables 
		# call before each batch
		
		self.z		= [zeros(self.p['b'], self.p['n_steps'], sum(self.p['z_dim'][l:l+2])).cuda() for l in range(self.p['layers'])]

		self.z_pred = [zeros(self.p['b'], self.p['n_steps'], sum(self.p['z_dim'][l:l+2])).cuda() for l in range(self.p['layers'])]
		
		self.d_pred = [zeros(self.p['b'], self.p['n_steps'], 1) for l in range(self.p['layers'])]
		
		self.r_pred = [zeros(self.p['b'], self.p['n_steps'], 1) for l in range(self.p['layers'])]
		
		self.rnn_lossvec = [0, 0]
		
		if self.p['use_lstm']:
			for l in range(self.p['layers']):
				self.lstm[l].reset()
			
	def plot(self, layer, images=None, iteration=None):
	
		def _render(_data, pdir, datachar, l, iteration, t):
			_dir = os.path.join(pdir, datachar)
			os.makedirs(_dir) if not os.path.exists(_dir) else None
			save_image(_data.data.cpu(), _dir+'/{}_{}_{}.png'.format(l,iteration,t))
		
		pdir = self.p['plot_dir']
		os.makedirs(pdir) if not os.path.exists(pdir) else None
		
		for t in range(self.p['n_steps']-1):
			for l in range(self.p['layers']):
				
				_render(self.z[l][0], pdir, 'real',	 l, iteration, t)
				_render(self.z_pred[l][0],	pdir, 'pred',  l, iteration, t)
	
	def loss(self, l):
		"""
		Input:	l	
		Output: loss, (metrics)
		l [int] - layer for which to calculate loss 
		loss[scalar] - loss for current layer
		metrics [tuple] - loss (detached) for discrete & continuous kl
		"""

		layer_loss = 0.
		train_loss = [[] for i in range(self.p['layers'])] 
		train_norm_kl_loss = [[] for i in range(self.p['layers'])]
		train_cat_kl_loss = [[] for i in range(self.p['layers'])]
		
		for l in range(self.p['layers']):		
			
			pred = self.z_pred[l]	;	real = self.z[l]
			
			splitdim = [self.p['nz_con'][l], sum(self.p['nz_dis'][l])]
			
			con_real, cat_real = torch.split(real, splitdim, dim=-1) 
			con_pred, cat_pred = torch.split(pred, splitdim, dim=-1) 
			
			cat_real = torch.max(cat_real, 1)[1].type(torch.LongTensor).cuda()
			xentloss = self.xent(cat_pred, cat_real)

			mseloss	 = self.mse(con_pred,  con_real)

			layer_loss += xentloss + mseloss

			train_loss[l].append(layer_loss.item())
			train_norm_kl_loss[l].append(mseloss.item())
			train_cat_kl_loss[l].append(xentloss.item())			
	
		return layer_loss, (train_loss[0], train_norm_kl_loss[0], train_cat_kl_loss[0])
			
	def forward(self, zs, acts, eval=False):

		for l in range(self.p['layers']):
			self.lstm[l].reset()
			for t in range(self.p['n_steps']):
				zpred, done, rew = self.lstm[l](zs[:,t], acts[:,t])
				if not t == self.p['n_steps']-1: 
					self.z[l][:,t]		= zs[:,t+1].squeeze(-2)
				self.z_pred[l][:,t] = zpred.squeeze(-2)
				self.d_pred[l][:,t] = done
				self.r_pred[l][:,t] = rew







class ObservationModel(Module):
	def __init__(self, p):
		super(ObservationModel,self).__init__()
		
		self.p = p
		
		self.err_plot_flag = True
		self.plot_errs = []
		self.enc_mode = False
		
		self.mse  = MSELoss(reduction='sum').cuda() if p['gpu'] else MSELoss()
		self.iter_loss = 0.0
		
		#self.xent = CrossEntropyLoss().cuda() if p['gpu'] else CrossEntropyLoss()
		#self.l1 = L1Loss().cuda() if p['gpu'] else L1Loss()
		
		# Initialise Distributions
		self.prior_dist, self.q_dist, self.x_dist, self.cat_dist = mutils.discheck(p)

		p['z_params']	= self.q_dist[0].nparams
		
		# Calc RF masks for dataset 
		masks, p = mutils.calc_rf(p)
		full_mask, reduce_mask = masks

		# Initialise Modules - learnable
		if p['conv']:

			if p['dataset'] == 'mnist' or p['dataset'] == 'moving_mnist':
				self.f_enc = mutils.ListModule(*[MNISTConvEncoder(p,l)	for l in range(p['layers'])])
				self.g_dec = mutils.ListModule(*[MNISTConvDecoder(p,l)	for l in range(p['layers'])])	

			elif p['dataset'] == 'stl10' and p['exp_name'] == 'freq':
				self.f_enc = mutils.ListModule(*[STL10ConvEncoder_freq(p,l) for l in range(p['layers'])])
				self.g_dec = mutils.ListModule(*[STL10ConvDecoder_freq(p,l) for l in range(p['layers'])])
			
			elif p['dataset'] == 'stl10':
				self.f_enc = mutils.ListModule(*[STL10ConvEncoder(p,l) for l in range(p['layers'])])
				self.g_dec = mutils.ListModule(*[STL10ConvDecoder(p,l) for l in range(p['layers'])])
			
			elif p['dataset'] == 'cifar10':
				self.f_enc = mutils.ListModule(*[CIFAR10ConvEncoder(p,l) for l in range(p['layers'])])
				self.g_dec = mutils.ListModule(*[CIFAR10ConvDecoder(p,l) for l in range(p['layers'])])
			
			else:
				self.f_enc = mutils.ListModule(*[ConvEncoder(p,l)  for l in range(p['layers'])])
				self.g_dec = mutils.ListModule(*[ConvDecoder(p,l)  for l in range(p['layers'])])
		else:
			self.f_enc = mutils.ListModule(*[Encoder(p,l)  for l in range(p['layers'])])
			self.g_dec = mutils.ListModule(*[Decoder(p,l)  for l in range(p['layers'])])

		if p['foveate']:
			self.retina = Retina(p['patch_size'], p['num_patches'], p['patch_noise'])
												
		# Initialise Modules - non-learnable
		self.g_obs = mutils.ListModule(*[ObsModel(p,masks,l) for l in range(p['layers'])])
		self.e_err = mutils.ListModule(*[ErrorUnit(p,full_mask,l) for l in range(p['layers'])])
		
		# Initialise Tensors
		self.reset()

		if p['gpu']:
			self.cuda()
		
	def plot(self,i,b, plot_vars=None):
		# pass all tensors to (utils) plot function
		m = {k:v for k, v in self.__dict__.items() if not (k.startswith('__') and k.endswith('__'))}
		mutils.plot(m,self.p,i,b)	
	
	def reset(self):
		# reset / init all model variables 
		# call before each batch
		
		# clears computation graph for next batch
		self.iter_loss = 0
		self.plot_errs = []
		
		# find a better way to handle this condition 
		if self.p['gpu']:
			self.error	  = [zeros(self.p['b'], *ld).cuda() for ld in self.p['ldim']]
			self.targets  = [zeros(self.p['b'], *ld).cuda() for ld in self.p['ldim']]
			self.pred	  = [zeros(self.p['b'], *ld).cuda() for ld in self.p['ldim']]
			self.obs	  = [zeros(self.p['b'], *ld).cuda() for ld in self.p['ldim']]
			self.noise	  = [zeros(self.p['b'], *ld).cuda() for ld in self.p['ldim']]
		else:
			self.error	  = [zeros(self.p['b'], *ld) for ld in self.p['ldim']]
			self.targets  = [zeros(self.p['b'], *ld) for ld in self.p['ldim']]
			self.pred	  = [zeros(self.p['b'], *ld) for ld in self.p['ldim']]
			self.obs	  = [zeros(self.p['b'], *ld) for ld in self.p['ldim']]
			self.noise	  = [zeros(self.p['b'], *ld) for ld in self.p['ldim']]
	
		self.z_pd = [[] for l in range(self.p['layers'])]
		self.z_pc = [[] for l in range(self.p['layers'])]
		
		if self.p['gpu']:
			for l in range(self.p['layers']):
				for alpha in self.p['nz_dis'][l]:
					self.z_pd[l].append(zeros(self.p['b'], 1, alpha).cuda())
				
				self.z_pc[l].append(zeros(self.p['b'], 1, sum(self.p['nz_con'][l:l+2])*2).cuda())
			
			self.z = [zeros(self.p['b'], sum(self.p['z_dim'][l:l+2])).cuda() for l in range(self.p['layers'])]
		else:
			for l in range(self.p['layers']):
				for alpha in self.p['nz_dis'][l]:
					self.z_pd[l].append(zeros(self.p['b'], 1, alpha))
				
				self.z_pc[l].append(zeros(self.p['b'], 1, sum(self.p['nz_con'][l:l+2])*2))
			
			self.z = [zeros(self.p['b'], sum(self.p['z_dim'][l:l+2])) for l in range(self.p['layers'])]
		
		
		if self.p['gpu']:
			if self.p['err_noise']:
				self.noise = [Variable(torch.zeros(self.p['b'], *ld)).cuda() for ld in self.p['ldim']]
		else:
			if self.p['err_noise']:
				self.noise = [Variable(torch.zeros(self.p['b'], *ld)) for ld in self.p['ldim']]
		
		if self.p['use_lstm']:
			for l in range(self.p['layers']):
				self.lstm[l].reset()
			
			if self.p['foveate']:
				self.action	  =	 zeros(self.p['b'], self.p['action_dim'], self.p['n_actions']).cuda()
				if self.p['gpu']:
					self.action = self.action.cuda()

			
			
	def loss(self, curiter, eval=False):

		loss = 0.
		
		train_loss = [[] for i in range(self.p['layers'])] 
		train_norm_kl_loss = [[] for i in range(self.p['layers'])]
		train_cat_kl_loss = [[] for i in range(self.p['layers'])]

		for l in range(self.p['layers']):		
			
			layer_loss = 0.
			
			#err_loss  = torch.abs(self.error[l]).sum() / np.prod(self.p['ldim'][0][1:])# * layer_weights
			
			err_loss = self.mse(self.error[l], self.targets[l])
		
			
			kloss_args	= (self.z_pc[l],   # mu, sig
						   self.p['z_con_capacity'][l], # anealing params
						   curiter)	# data size
						   
			norm_kl_loss = self.q_dist[l].calc_kloss(*kloss_args) #/ self.p['b']
			
			kloss_args	 = (self.z_pd[l],  # alpha
							self.p['z_dis_capacity'][l],  # anneling params 
							self.p['nz_dis'][l], # nclasses per categorical dimension
							curiter)	# data size
						  
			cat_kl_loss = self.cat_dist[l].calc_kloss(*kloss_args) #/ self.p['b']
			
			#err_loss /= (32*32)

			if self.p['elbo_loss']:	
				layer_loss = norm_kl_loss + cat_kl_loss + err_loss
			else:
				layer_loss = err_loss 

			loss += layer_loss / np.prod(self.p['imdim'][1:])
			#loss /= self.p['b']#(32*32)

			train_loss[l].append(err_loss.item())
			train_norm_kl_loss[l].append(norm_kl_loss.item())
			train_cat_kl_loss[l].append(cat_kl_loss.item())
		
		metrics = [m/np.prod(self.p['imdim'][1:]) for m in (train_loss[0], train_norm_kl_loss[0], train_cat_kl_loss[0])]

		#loss /= self.p['b']#np.prod(self.p['imdim'][1:])
		#print(err_loss.item(), norm_kl_loss.item(), cat_kl_loss.item())
		#loss /= 32*32
		#return loss, tuple(x[0]/self.p['b'] for x in metrics)
		return loss, tuple(x[0] for x in metrics)
	
	def decode(self, z, l):
		return self.g_dec[l](z).data
	
	def forward(self, iter, image, actions=None, eval=False, to_matlab=False):
		
		#from torchvision.utils import save_image
		#save_image(image[0], 'lol.png')

		if to_matlab:
			
			self.p['gpu'] = False
			self.reset()
			if not isinstance(image, Tensor) or isinstance(image, FloatTensor):
				image = FloatTensor(np.asarray(image)).unsqueeze(0).unsqueeze(0)
				#from torchvision.utils import save_image
				#save_image(image, 'lol.png')

			if not actions is None:
				actions = FloatTensor(actions).unsqueeze(0).unsqueeze(0)
		
		if self.p['foveate'] and not actions is None:
			
			image, foveated = self.retina.foveate(image, actions)

		#from torchvision.utils import save_image
		#save_image(image[0], 'lol.png')
		#save_image(foveated[0], 'fov.png')

		if len(image.shape) == 4:
			
			image = image.unsqueeze(1)
			image = image.expand(-1, self.p['model_inner'], *image.shape[-3:])
		

		#for t in range(image.shape[1]):
		for t in range(self.p['model_inner']):
			# Top down
			for l_r in reversed(range(self.p['layers'])):
				
				bu = self.error[l_r] if self.p['prednet'] else self.pred[l_r]
				td = self.z[l_r+1] if l_r < self.p['layers']-1 else None
 
				# Encoding - p(z2|x) or p(z1 |x,z2)
				self.z_pc[l_r], self.z_pd[l_r] = self.f_enc[l_r](bu, td)
				
				# Latent Sampling
				latent_sample = []

				# Continuous sampling 
				norm_sample = self.q_dist[l_r].sample_normal(params=self.z_pc[l_r], train=self.training)
				latent_sample.append(norm_sample)

				# Discrete sampling
				for ind, alpha in enumerate(self.z_pd[l_r]):
					cat_sample = self.cat_dist[l_r].sample_gumbel_softmax(alpha, train=self.training)
					latent_sample.append(cat_sample)
					

				self.z[l_r] = torch.cat(latent_sample, dim=-1)

				# Decoding - p(x|z)
				self.pred[l_r] = self.g_dec[l_r](self.z[l_r])

			# Bottom up
			for l in range(self.p['layers']):
				
				self.obs[l] = self.g_obs[l](image[:,t] if l==0 else self.error[l-1])
				self.error[l] = self.e_err[l](self.obs[l], self.pred[l], self.noise[l])
				if self.err_plot_flag:
					self.plot_errs.append(self.error[l][0].data)

			iter_loss, _ = self.loss(iter) 
			self.iter_loss += iter_loss / image.shape[1]
			
		if self.err_plot_flag:
			self.err_plot_flag = False
		
		if to_matlab:

			return self.z[0].detach().numpy(), foveated.detach().numpy()
			
		elif eval:
			return (self.z_pc, self.z_pd), self.pred

		elif self.enc_mode:
			return self.z[0]

		
class ObservationVAE(Module):
	def __init__(self, p):
		super(ObservationVAE,self).__init__()
		
		self.p = p
		
		self.err_plot_flag = True
		self.plot_errs = []
		self.enc_mode = False
		
		self.mse  = MSELoss(reduction='sum').cuda() if p['gpu'] else MSELoss()
		self.iter_loss = 0.0
		
		p = mutils.calc_rf(p)[1]
				
		# Initialise Distributions
		self.prior_dist, self.q_dist, self.x_dist, self.cat_dist = mutils.discheck(p)
		
		p['z_params']	= self.q_dist.nparams

		# Initialise Modules - learnable
		if p['conv']:

			if p['dataset'] == 'mnist' or p['dataset'] == 'moving_mnist':
				self.f_enc = MNISTConvEncoder(p,0)
				self.g_dec = MNISTConvDecoder(p,0)

			elif p['dataset'] == 'stl10' and p['exp_name'] == 'stl10_freq_2' and p['num_freqs'] == 2:
				self.f_enc_low = STL10ConvEncoder_low_freq(p,0)
				self.g_dec_low = STL10ConvDecoder_low_freq(p,0)	

				self.f_enc_hi = STL10ConvEncoder_hi_freq(p,0)
				self.g_dec_hi = STL10ConvDecoder_hi_freq(p,0)	

			elif p['dataset'] == 'stl10' and p['exp_name'] == 'stl10_freq':
				self.f_enc = STL10ConvEncoder_freq(p,0)
				self.g_dec = STL10ConvDecoder_freq(p,0)			
			
			elif p['dataset'] == 'stl10' and p['exp_name'] == 'stl10_patch':
				self.f_enc = STL10_patch_ConvEncoder(p,0)
				self.g_dec = STL10_patch_ConvDecoder(p,0)

			elif p['dataset'] == 'stl10':
				self.f_enc = STL10ConvEncoder(p,0)
				self.g_dec = STL10ConvDecoder(p,0)

			elif p['dataset'] == 'cifar10':
				self.f_enc = CIFAR10ConvEncoder(p,0)
				self.g_dec = CIFAR10ConvDecoder(p,0)
				
			elif p['dataset'] == 'celeba':
				self.f_enc = CelebConvEncoder(p,0)
				self.g_dec = CelebConvDecoder(p,0)			
			
			else:
				self.f_enc = ConvEncoder(p,0)
				self.g_dec = ConvDecoder(p,0)
		else:
			self.f_enc = Encoder(p,0)
			self.g_dec = Decoder(p,0)

		if p['foveate']:
			self.retina = Retina(p['patch_size'], p['num_patches'], p['patch_noise'])
														
		# Initialise Tensors
		self.reset()

		if p['gpu']:
			self.cuda()
		
	def plot(self, i, input_image, plot_vars):
	
		z, pred = plot_vars
		pdir = os.path.join(self.p['plot_dir'], self.p['model_name'])
		matsdir = os.path.join(self.p['plot_dir'], self.p['model_name'], 'mats')
		
		os.makedirs(pdir) if not os.path.exists(pdir) else None
		os.makedirs(matsdir) if not os.path.exists(matsdir) else None
	
		save_image(pred[0].data.cpu(), pdir+'/p{}.png'.format(i))
		save_image(input_image[0].data.cpu(), pdir+'/b{}.png'.format(i))
		
		#if p['vae']:
		#	mu	= m['z_pc'][l].select(-1, 0).data.cpu().numpy()
		#	var = m['z_pc'][l].select(-1, 1).data.cpu().numpy()
		#	sio.savemat(os.path.join(matsdir,'mu_{}.mat'.format(i)), {'r':mu})
		#	sio.savemat(os.path.join(matsdir,'var_{}.mat'.format(i)), {'r':var})

		z	= z.data.cpu().numpy()
		sio.savemat(os.path.join(matsdir,'z_{}.mat'.format(i)), {'r':z})
	
	def reset(self):
		# reset / init all model variables 
		# call before each batch
		
		# clears computation graph for next batch
		self.iter_loss = 0
		self.plot_errs = []
		
		if self.p['use_lstm']:
			for l in range(self.p['layers']):
				self.lstm[l].reset()
			
			if self.p['foveate']:
				self.action	  =	 zeros(self.p['b'], self.p['action_dim'], self.p['n_actions']).cuda()
				if self.p['gpu']:
					self.action = self.action.cuda()
			
	def loss(self, curiter, image, pred, z_pc, z_pd, eval=False):

		loss 			   = 0.		
		train_loss 		   = [] 
		train_norm_kl_loss = []
		train_cat_kl_loss  = [] 
		layer_loss 		   = 0.

		
		err_loss = self.mse(image, pred)			
	
		kloss_args	= (z_pc,   # mu, sig
					   self.p['z_con_capacity'][0], # anealing params
					   curiter)	# data size
					   
		norm_kl_loss = self.q_dist.calc_kloss(*kloss_args) #/ self.p['b']
		
		kloss_args	 = (z_pd,  # alpha
						self.p['z_dis_capacity'][0],  # anneling params 
						self.p['nz_dis'][0], # nclasses per categorical dimension
						curiter)	# data size
					  
		cat_kl_loss = self.cat_dist.calc_kloss(*kloss_args) #/ self.p['b']

		if self.p['elbo_loss']:	
			layer_loss = norm_kl_loss + cat_kl_loss + err_loss
		else:
			layer_loss = err_loss 

		loss += layer_loss

		if self.p['dataset'] == 'mnist':
			loss /= np.prod(self.p['imdim'][1:])
		
		
		metrics = (err_loss.item(), norm_kl_loss.item(), cat_kl_loss.item())

		return loss, metrics
	
	def decode(self, z, l):
		return self.g_dec(z).data
	
	def foveate(self, im, a):
		image, foveated = self.retina.foveate(im, a)
		return foveated.numpy()		
	
	def forward(self, iter, image, actions=None, eval=False, to_matlab=False):
			
		if to_matlab:
			
			self.p['gpu'] = False
			self.reset()
			if not isinstance(image, Tensor) or isinstance(image, FloatTensor):
				image = FloatTensor(np.asarray(image)).unsqueeze(0).unsqueeze(0)
				#from torchvision.utils import save_image
				#save_image(image, 'lol.png')

			if not actions is None:
				actions = FloatTensor(actions).unsqueeze(0).unsqueeze(0)
		
		if self.p['foveate'] and not actions is None:
			image, foveated = self.retina.foveate(image, actions)	

		# Encoding - p(z2|x) or p(z1 |x,z2)
		z_pc, z_pd = self.f_enc(image, None)
		
		# Latent Sampling
		latent_sample = []

		# Continuous sampling 
		norm_sample = self.q_dist.sample_normal(params=z_pc, train=self.training)
		latent_sample.append(norm_sample)

		# Discrete sampling
		for ind, alpha in enumerate(z_pd):
			cat_sample = self.cat_dist.sample_gumbel_softmax(alpha, train=self.training)
			latent_sample.append(cat_sample)
			

		z = torch.cat(latent_sample, dim=-1)

		# Decoding - p(x|z)
		pred = self.g_dec(z)
		
		if self.training:

			iter_loss, _ = self.loss(iter, image, pred, z_pc, z_pd) 
			#self.iter_loss += iter_loss / image.shape[1]
			self.iter_loss += iter_loss

		if self.err_plot_flag:
			self.err_plot_flag = False
		
		if to_matlab:
			return z.detach().numpy(), foveated.detach().numpy()
			
		elif eval:
			return z, pred

		elif self.enc_mode:
			return z


class PrednetWorldModel(Module):
	def __init__(self, p):
		super(PrednetWorldModel,self).__init__()
		
		self.p = p
		
		self.err_plot_flag = True
		self.plot_errs = []
		self.enc_mode = False
		
		self.mse  = MSELoss(reduction='sum').cuda() if p['gpu'] else MSELoss()
		self.iter_loss = 0.0
		
		#self.xent = CrossEntropyLoss().cuda() if p['gpu'] else CrossEntropyLoss()
		#self.l1 = L1Loss().cuda() if p['gpu'] else L1Loss()
		
		self.brain = BrainParameters(brain_name='Learner',
									 camera_resolutions=[{'height': 84, 'width': 84, 'blackAndWhite': False}],
									 num_stacked_vector_observations=self.p['b'],
									 vector_action_descriptions=['', ''],
									 vector_action_space_size=[3, 3],
									 vector_action_space_type=0,  # corresponds to discrete
									 vector_observation_space_size=3)		
		

		
		
		# Initialise Distributions
		self.prior_dist, self.q_dist, self.x_dist, self.cat_dist = mutils.discheck(self.p)

		self.q_dist = self.q_dist[0]
		self.cat_dist = self.cat_dist[0]		
		
		p['z_params']	= self.q_dist.nparams
		# Calc RF masks for dataset 
		masks, p = mutils.calc_rf(p)
		full_mask, reduce_mask = masks

		# Initialise Modules - learnable
		self.f_enc = ConvEncoder(p,0)
		self.g_dec = ConvDecoder(p,0)

		# Putting this in a list precludes it from the parent model's graph 
		self.a_net	= [ActionNet(p,0)]

		# Initialise Modules - non-learnable
		self.g_obs = ObsModel(p,masks,0)
		self.e_err = ErrorUnit(p,full_mask,0)
		
		self.lstm  = DynamicModel(p,0)
		
		# Initialise Tensors
		self.reset()

		if p['gpu']:
			self.cuda()
			self.a_net[0].cuda()
		
	def plot(self, i, input_image, plot_vars):
	
		z, pred = plot_vars
		pdir = os.path.join(self.p['plot_dir'], self.p['model_name'])
		matsdir = os.path.join(self.p['plot_dir'], self.p['model_name'], 'mats')
		
		os.makedirs(pdir) if not os.path.exists(pdir) else None
		os.makedirs(matsdir) if not os.path.exists(matsdir) else None
	
		save_image(pred[0].data.cpu(), pdir+'/p{}.png'.format(i))
		save_image(input_image[0].data.cpu(), pdir+'/b{}.png'.format(i))
	
	def reset(self):
		# reset / init all model variables 
		# call before each batch
		
		# clears computation graph for next batch
		self.iter_loss = 0
		self.plot_errs = []

		# Initialise Distributions
		del self.q_dist, self.cat_dist
		self.prior_dist, self.q_dist, self.x_dist, self.cat_dist = mutils.discheck(self.p)

		self.q_dist = self.q_dist[0]
		self.cat_dist = self.cat_dist[0]
		
		self.lstm.reset()
			
	def loss(self, curiter, error, z_pc, z_pd, eval=False):

		loss = 0.
		
		train_loss = []  
		train_norm_kl_loss = [] 
		train_cat_kl_loss = [] 
		
		#err_loss  = torch.abs(self.error[l]).sum() / np.prod(self.p['ldim'][0][1:])# * layer_weights
		
		err_loss = self.mse(error, torch.zeros_like(error))
	
		
		kloss_args	= (z_pc,   # mu, sig
					   self.p['z_con_capacity'][0], # anealing params
					   curiter)	# data size
					   
		norm_kl_loss = self.q_dist.calc_kloss(*kloss_args) #/ self.p['b']
		
		kloss_args	 = (z_pd,  # alpha
						self.p['z_dis_capacity'][0],  # anneling params 
						self.p['nz_dis'][0], # nclasses per categorical dimension
						curiter)	# data size
					  
		cat_kl_loss = self.cat_dist.calc_kloss(*kloss_args) #/ self.p['b']
		
		#err_loss /= (32*32)

		if self.p['elbo_loss']:	
			layer_loss = norm_kl_loss + cat_kl_loss + err_loss
		else:
			layer_loss = err_loss 

		loss += layer_loss / np.prod(self.p['imdim'][1:])
		#loss /= self.p['b']#(32*32)

		train_loss.append(err_loss.item())
		train_norm_kl_loss.append(norm_kl_loss.item())
		train_cat_kl_loss.append(cat_kl_loss.item())
		
		metrics = [m/np.prod(self.p['imdim'][1:]) for m in (train_loss[0], train_norm_kl_loss[0], train_cat_kl_loss[0])]

		#loss /= self.p['b']#np.prod(self.p['imdim'][1:])
		#print(err_loss.item(), norm_kl_loss.item(), cat_kl_loss.item())
		#loss /= 32*32
		#return loss, tuple(x[0]/self.p['b'] for x in metrics)

		return loss, metrics
	
	def decode(self, z, l):
		return self.g_dec[l](z).data
	
	def forward(self, iter, image, error, actions=None, eval=False, to_matlab=False):
	
		# Encoding - p(z2|x) or p(z1 |x,z2)
		z_pc, z_pd = self.f_enc(error , None)
		
		# Latent Sampling
		latent_sample = []

		# Continuous sampling 
		norm_sample = self.q_dist.sample_normal(params=z_pc, train=self.training)
		latent_sample.append(norm_sample)

		# Discrete sampling
		for ind, alpha in enumerate(z_pd):
			cat_sample = self.cat_dist.sample_gumbel_softmax(alpha, train=self.training)
			latent_sample.append(cat_sample)

		z = torch.cat(latent_sample, dim=-1)

		#self.z[l_r], done = self.lstm[l_r](z, actions[:,t])
		z = self.lstm(z, actions)

		# Decoding - p(x|z)
		pred = self.g_dec(z)

		# Bottom up			
		obs   = self.g_obs(image)
		error = self.e_err(obs, pred, None)
			   # this is not a zeroth layer index (see init)
		#return self.a_net[0](self.z[0].detach(), self.lstm.lstm_h.detach(), actions)
		actions = self.a_net[0](Variable(z.detach()), Variable(self.lstm.lstm_h.detach()), Variable(actions.detach()))
		return actions, error, z_pc, z_pd
