import os
import math
from torch import nn, max
import logging
import urllib.request as request
from torch.utils.data import DataLoader, TensorDataset 
import torchvision.transforms.functional as TTF
from torch import FloatTensor
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import scipy.io as sio
from numbers import Number
import numpy as np
import numpy as np
from itertools import chain, product
from collections.abc import Iterable
from torch.utils.data import TensorDataset
from utils import data_utils as dutils
from utils import model_utils as mutils
from utils import vis

from utils import dist 
import time 
import torch
import glob
#import matlab


def kl_annealing(p, iteration):
	if p['dataset']	  == 'shapes':
		warmup = 7000
	elif p['dataset'] == 'faces' or 'lsun_bedrooms':
		warmup = 2500	
	elif p['dataset'] == 'mnist' or 'mnnist_sequences':
		warmup = 1000
	else:
		print('no such dataset {} :', p['dataset'])
	if p['lambda_annealing']:
		p['lamb'] = np.maximum(0, 0.95 - 1 / warmup * iteration)	 # 1 --> 0
	else:
		p['lamb'] = 0
	if p['beta_annealing']:
		p['beta'] = min(p['beta'], p['beta'] / warmup * iteration)	# 0 --> 1
		
	return p

def calc_rf(p):

	#  Calculate on/off/full receptive fields 
	#  returns a dictionary containing:
	#  the layer wise mask M(xl0, yl0) 
	#  the weight matrix W(xl0*yl0, xl1*yl1)
	#  for each of full / off / on RFs
	#  weird property - square / rect along column
	#  horz / vert rect along rows

	height = [p['imdim'][-1]] * (p['layers']+1)
	width  = [p['imdim'][-2]] * (p['layers']+1)
	chans  = [p['imdim'][-3]] * (p['layers']+1)

	filter_size = [7, 5, 3] # please keep these odd
	stride = [2, 2, 2]

	p['ldim'] = [None] * (p['layers']+1)

	l2l_mask = {'on'	: [None] * (p['layers']+1), 
				'off'  : [None] * (p['layers']+1), 
				'full' : [None] * (p['layers']+1)}
	
	obs_mask = {'on'	: [None] * (p['layers']+1), 
				'off'  : [None] * (p['layers']+1), 
				'full' : [None] * (p['layers']+1)}
				
	

	#l2l_mask = {'full' : [None] * (p['layers']+1)}
	
	#obs_mask = {'full' : [None] * (p['layers']+1)}



	for l in range(p['layers']):

		height[l+1] = ((height[l] - filter_size[l]) // stride[l] ) + 1
		width[l+1]	= ((width[l]  - filter_size[l]) // stride[l] ) + 1
		p['ldim'][l] = [chans[l], height[l], width[l]]

		#a = [i for i in range((filter_size[l]//2)+1)]
		#cntr = a[len(a)//2]
		#on = a[:cntr] ; off = a[cntr:] ; full = a;
        #
		#on_c_rf	  = set(product([*on, *[ -i for i in on]], repeat=2))
		#full_c_rf = set(product([*a,*[-i for i in a]], repeat=2)) 
		#off_c_rf  = full_c_rf.difference(on_c_rf)
		#rfs = {'full' : full_c_rf, 'on' : on_c_rf, 'off' : off_c_rf}
		#
		## indexable im space - disregarding channel dimslatent_dist
		#coords = np.zeros((height[l], width[l]),dtype='i,i,i').tolist()
        #
		#for m in range(height[l]):
		#	for n in range(width[l]):
		#		coords[m][n] = (m,n)
        #
		## chain image coordinates 
		#coords = list(chain.from_iterable(zip(*coords)))
        #
		#for rf_type in l2l_mask.keys():
        #
		#	# apply coordinate transforms to rf	
		#	rf_coords	= [None] * len(coords)
        #
		#	weight_matrix = np.zeros((chans[l]*height[l]*width[l], chans[l+1]*height[l+1]*width[l+1]))
		#	mask_matrix	  = np.zeros((chans[l]*height[l]*width[l], chans[l]*height[l]*width[l]))
		#	# for every image coordinate
		#	for ind, imxy in enumerate(coords): # better plz 
        #
		#		# calculate the receptive field in terms of real indices
		#		imxyrf = np.array([(imxy[0] + rf_[0], imxy[1] + rf_[1]) for rf_ in rfs[rf_type]])
        #
		#		# filter invalid coordinates
		#		validation = lambda y, w, h : y[0]>=0 and y[0]<w and y[1]>=0 and y[1]<h 
		#		imxyrf	 = imxyrf[[validation(rfind, width[l],height[l]) for rfind in imxyrf]]
		#		
		#		# filter unique coordinates
		#		imxyrf	 = list(set(map(tuple, imxyrf)))
		#		rf_coords[ind] = imxyrf
		#		
		#		# mask of ones 
		#		rf_map = np.zeros((chans[l], height[l], width[l])).astype('int')
		#		for rcrd in rf_coords[ind]:
		#			rf_map[:, rcrd[0], rcrd[1]] = 1
        #
		#		# connection map
		#		connect_to = np.zeros((chans[l+1],height[l+1], width[l+1]))
		#		for x in range(height[l+1]):
		#			for y in range(width[l+1]):
		#				connect_to[:,x,y] = rf_map[:, stride[l]*x+filter_size[l+1], stride[l]*y+filter_size[l+1]]
        #
		#		weight_matrix[ind,:] = connect_to.flatten()
		#		mask_matrix[ind,:]	 = rf_map.flatten()
        #
		#	l2l_mask[rf_type][l] = torch.from_numpy(weight_matrix).float()
		#	obs_mask[rf_type][l] = torch.from_numpy(mask_matrix).float()
			

	p['ldim']  = list(filter(None.__ne__, p['ldim']))
	return (obs_mask, l2l_mask), p
	
def discheck(p,hi=0,mu_low=0,sigma_low=1):

	if p['z_dist'] == 'normal':
		prior_dist = [dist.Normal() for _ in range(p['layers'])]
		q_dist	  = [dist.Normal() for _ in range(p['layers'])]

	# set up distribution with empirical prior (from low frequencies)
	elif p['z_dist'] == 'normal' and hi == 1:
		prior_dist = [dist.Normal(mu=mu_low, sigma=sigma_low) for _ in range(p['layers'])]
		q_dist	  = [dist.Normal(mu=mu_low, sigma=sigma_low) for _ in range(p['layers'])]
		
	elif p['z_dist'] == 'laplace':
		prior_dist = [dist.Laplace() for _ in range(p['layers'])]
		q_dist	   = [dist.Laplace() for _ in range(p['layers'])]
	elif p['z_dist'] == 'flow':
		prior_dist = [dist.FactorialNormalizingFlow(dim=sum(p['z_dim'][l:l+2]), nsteps=32) for l in range(p['layers'])]
		q_dist	   = [dist.Normal()	for _ in range(p['layers'])]		
	else:
		print('no z matching dist')
		0/0
	
	cat_dist = [dist.Gumbel(p['nz_dis'][l], p['enc_temp']) for l in range(p['layers'])]
	
	prior_dist = [x.cuda() if p['gpu'] else x for x in prior_dist]
	q_dist	   = [x.cuda() if p['gpu'] else x for x in q_dist]
	cat_dist   = [x.cuda() if p['gpu'] else x for x in cat_dist]
	
	x_dist = None
	if p['x_dist'] == 'normal':
		x_dist = dist.Normal()
	elif p['x_dist'] == 'bernoulli':
		x_dist = dist.Bernoulli()	
	
	if x_dist:
		p['x_dist']= x_dist.cuda() if p['gpu'] else x_dist
	
	if p['prednet']:
		return prior_dist, q_dist, p['x_dist'], cat_dist
	else:
		return prior_dist[0], q_dist[0], p['x_dist'], cat_dist
		# return prior_dist[0], q_dist[0], p['x_dist'], cat_dist[0]

def plot(m, p, iter, bottom_up):

	def _render(_data, pdir, datachar, l, iter, t):
		_dir = os.path.join(pdir, datachar)
		os.makedirs(_dir) if not os.path.exists(_dir) else None
		save_image(_data.data.cpu(), _dir+'/{}_{}_{}.png'.format(l,iter,t))
	
	pdir = os.path.join(p['plot_dir'], p['model_name'])
	matsdir = os.path.join(p['plot_dir'], p['model_name'], 'mats')
	
	os.makedirs(pdir) if not os.path.exists(pdir) else None
	os.makedirs(matsdir) if not os.path.exists(matsdir) else None
	
	for t in range(p['model_inner']-1):
		for l in range(p['layers']):
			
			_render(m['error'][l][0], pdir, 'e',  l, iter, t)
			_render(m['pred'][l][0],  pdir, 'p',  l, iter, t)
			_render(m['z'][l],     pdir, 'z',  l, iter, t)
			_render(bottom_up,  	  pdir, 'bu', 0, iter, t)
		
			if p['vae']:
				mu	= m['z_pc'][l].select(-1, 0).data.cpu().numpy()
				var = m['z_pc'][l].select(-1, 1).data.cpu().numpy()
				sio.savemat(os.path.join(matsdir,'mu_{}_{}_{}.mat'.format(l,iter,t)), {'r':mu})
				sio.savemat(os.path.join(matsdir,'var_{}_{}_{}.mat'.format(l,iter,t)), {'r':var})

			z	= m['z'][l][0].data.cpu().numpy()
			sio.savemat(os.path.join(matsdir,'z_{}_{}_{}.mat'.format(l,iter,t)), {'r':z})	

def isnan(tensor):
	return (tensor != tensor)

def logsumexp(value, dim=None, keepdim=False):
	"""Numerically stable implementation of the operation

	value.exp().sum(dim, keepdim).log()
	"""
	if dim is not None:
		m, _ = torch.max(value, dim=dim, keepdim=True)
		value0 = value - m
		if keepdim is False:
			m = m.squeeze(dim)
		return m + torch.log(torch.sum(torch.exp(value0),
									   dim=dim, keepdim=keepdim))
	else:
		m = torch.max(value)
		sum_exp = torch.sum(torch.exp(value - m))
		if isinstance(sum_exp, Number):
			return m + math.log(sum_exp)
		else:
			return m + torch.log(sum_exp)

def init_weights(m):
	if type(m) == nn.Linear:
		#nn.init.xavier_uniform_(m.weight)
		nn.init.kaiming_normal_(m.weight)
		#m.bias.data.fill_(0.001)
		m.bias.data.fill_(0)
		

def set_requires_grad(module, val):
	# debug only - test clean detachment
	for p in module.parameters():
		p.requires_grad = val


class ListModule(nn.Module):
	def __init__(self, *args):
		
		super(ListModule, self).__init__()
		idx = 0
		for module in args:
			self.add_module(str(idx), module)
			idx += 1

	def __getitem__(self, idx):
		it = iter(self._modules.values())
		for i in range(idx):
			next(it)
		return next(it)

	def __iter__(self):
		return iter(self._modules.values())

	def __len__(self):
		return len(self._modules)
		
def visualise(p, model, e, test_loader):

	import matplotlib.pyplot as plt
	_v = vis.Visualiser(p, model)
	for batch in test_loader:
		#batch = batch[0]
		break
	#if len(batch.shape) == 5:
	#	batch = batch[:,-1]
	batch = dutils.data_check(p, batch)
	# Reconstruct data using Joint-VAE model
	_v.reconstructions(e, batch)

	# Plot samples
	_v.samples(e)

	# Plot all traversals
	_v.all_latent_traversals(e, size=10)

	# Plot a grid of some traversals
	for j in range(len(p['nz_dis'][0])):
		for i in range(p['nz_con'][0]):
			
			_v.latent_traversal_grid(j, i, e, cont_idx=i, cont_axis=1, disc_idx=j, disc_axis=0, size=(p['nz_dis'][0][j], p['nz_dis'][0][j]))

