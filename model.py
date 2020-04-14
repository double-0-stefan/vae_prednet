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
# import torch_xla
# import torch_xla.core.xla_model as xm


class pc_conv_network(nn.Module):
	def __init__(self,p):
		super(pc_conv_network, self).__init__()

		self.err_plot_flag = True
		self.plot_errs = []
		self.enc_mode = False
		
		self.p = p
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

		self.F = None
		self.F_last = None

		self.baseline = None
		
		if p['vae'] == 1:
			self.init_vae(p)
		#self.cuda()

	def init_conv_trans(self, p): # does conv, phi and precision
	
		x = torch.zeros([p['bs'],p['imchan'],self.p['imdim_'],self.p['imdim_']])

		self.p['dim'] = []
		self.conv_trans = []
		phi = []
		Precision = []
		weights = []
		P_chol = []

		# # Image level - needs Precision
		

		## Precision as cholesky factor -> ensure symetric positive semi-definite
		a = torch.eye(p['imchan']*p['imdim_']*p['imdim_'])
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


			## Precision as cholesky factor -> ensure symetric positive semi-definite
			a = torch.eye(p['chan'][j][-1]*x.size(2)*x.size(2))
			#a = torch.mm(a,a.t())
			a = torch.cholesky(a)
			P_chol.append(nn.Parameter(a))

			Precision.append(nn.Bilinear(p['chan'][j][-1]*x.size(2)*x.size(2), p['chan'][j][-1]*x.size(2)*x.size(2), 1, bias=False))
			Precision[j+1].weight = nn.Parameter(a)

			
			

		## APPEND NEW BITS TO MAIN ##
			self.conv_trans.append(nn.ModuleList(conv_trans_block))
			self.p['dim'].append(dim_block)

		# top level phi
		phi.append(nn.Parameter((torch.rand_like(x)).view(self.bs,-1)))

		self.Precision = nn.ModuleList(Precision)
		self.P_chol = nn.ParameterList(P_chol)
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

		print(self)

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

	def loss(self, i, learn):

		# do block
		x = self.phi[i].view(self.bs, self.chan[i][-1], self.dim[i][-1], self.dim[i][-1])
		for j in reversed(range(len(self.p['ks'][i]))):
			
			x = self.conv_trans[i][j](F.relu(x))

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

		## for Cholesky-based precision
		# tril: ensure upper tri doesn't get involved at all!
		if learn == 1:
			P1 = torch.mm(torch.tril(self.P_chol[i+1]), torch.tril(self.P_chol[i+1]).t())
			P0 = torch.mm(torch.tril(self.P_chol[i]), torch.tril(self.P_chol[i]).t())

			self.F +=  0.5*(
				# logdet cov = -logdet precision
				- torch.logdet(P1)

				+ torch.matmul(torch.matmul(PE_1,P1),PE_1.t())

				- torch.logdet(P0)

				+ torch.matmul(torch.matmul(PE_0,P0),PE_0.t())
				)
		else:
			self.F +=  0.5*(
				# logdet cov = -logdet precision
				- torch.logdet(torch.squeeze(self.Precision[i+1].weight))

				+ sum(self.Precision[i+1](PE_1, PE_1))

				- torch.logdet(torch.squeeze(self.Precision[i].weight))

				+ sum(self.Precision[i](PE_0, PE_0))
				)



		# self.F += - 0.5*(
		# 	# logdet cov = -logdet precision
		# 	  torch.logdet(torch.squeeze(self.Precision[i+1].weight))

		# 	- sum(self.Precision[i+1](PE_1, PE_1))

		# 	+ torch.logdet(torch.squeeze(self.Precision[i].weight))

		# 	- sum(self.Precision[i](PE_0, PE_0))
		# 	)
		
	def inference(self):

		for l in range(0, self.nlayers):
			for m in range(len(self.conv_trans[l])):
				self.conv_trans[l][m].requires_grad_(False)
			#self.Precision[l].requires_grad_(False)
			self.P_chol[l].requires_grad_(False)
			self.phi[l].requires_grad_(True)
		#self.optimizer.lr = self.p['lr']

		for i in range(self.iter):
			self.optimizer.zero_grad()
			self.F_old = self.F
			self.F = 0#nn.Parameter(torch.zeros(1))
			#self.phi_old = self.phi
			
			# will need to code reset for phi
			for l in range(0, self.nlayers):
				self.loss(l,learn=0)

			# if i < self.iter-1:
			# 	self.F.backward(retain_graph=True)
			# else:
			self.F.backward()
			#print(i)

			# xm.optimizer_step(self.optimizer)#.step()
			self.optimizer.step()
			print(self.F)
			# end inference if starting to diverge
			# if i > 0:
			# 	if self.F > self.F_old:
			# 		self.F = self.F_old
			# 		self.phi = self.phi_old
			# 		break

			# print(self.F)
			# print(torch.sum(self.images-self.F_old))


	def learn(self):

		# update Precision weights
		# for l in range(0,self.nlayers):
		# 	self.Precision[l].weight = torch.matrix_power(self.weights[l],2)

		#self.weights.requires_grad_(True)
		self.conv_trans.requires_grad_(True)
		# self.Precision.requires_grad_(True)
		for i in range(len(self.P_chol)):
			self.P_chol[i].requires_grad_(True)
		self.phi.requires_grad_(False)
		#self.optimizer.lr = 0.001

		self.optimizer.zero_grad()
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

		self.F.backward()
		# xm.optimizer_step(self.optimizer, barrier=False)
		self.optimizer.step()

		print(self.F)
		print(self.P_chol[0])
		print(self.P_chol[1])
		print(self.P_chol[2])

	def forward(self, iteration, images, learn=1):

		self.optimizer = Adam(self.parameters(), lr=self.p['lr'], weight_decay=1e-5)
		self.iteration = iteration
		self.F_last = self.F

		# images.half()
		if self.p['xla']:
			self.images = images.view(self.bs, -1).to(xm.xla_device())
		else:	
			self.images = images.view(self.bs, -1).cuda()

		# put weights into bilinear for inference and see if faster (no update done)
		for i in range(len(self.P_chol)):
			# reset wactivations
			self.phi[i] = nn.Parameter(torch.rand_like(self.phi[i]))
			self.Precision[i].weight = torch.nn.Parameter(torch.mm(self.P_chol[i],self.P_chol[i].t()).unsqueeze(0))

		self.inference()
		# if learn == 1:
		self.learn()









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

class ObservationVAE_freq(Module):
	def __init__(self, p):
		super(ObservationVAE_freq,self).__init__()
		
		self.p = p
		
		self.err_plot_flag = True
		self.plot_errs = []
		self.enc_mode = False
		
		self.mse  = MSELoss(reduction='sum').cuda() if p['gpu'] else MSELoss()
		self.iter_loss = 0.0
		
		p = mutils.calc_rf(p)[1]
				
		# Initialise Distributions
		self.prior_dist_low, self.q_dist_low, self.x_dist_low, self.cat_dist_low = mutils.discheck(p)
		p['z_params_low']	= self.q_dist.nparams

		# will obv need to have changed priors:
		self.prior_dist_hi, self.q_dist_hi, self.x_dist_hi, self.cat_dist_hi = mutils.discheck(p)
		p['z_params_hi']	= self.q_dist_hi.nparams

		# Initialise Modules - learnable
		if p['conv']:

			if p['dataset'] == 'stl10' and p['exp_name'] == 'stl10_freq_2' and p['num_freqs'] == 2:
				self.f_enc_low = STL10ConvEncoder_low_freq(p,0)
				self.g_dec_low = STL10ConvDecoder_low_freq(p,0)	

				self.f_enc_hi = STL10ConvEncoder_hi_freq(p,0)
				self.g_dec_hi = STL10ConvDecoder_hi_freq(p,0)	

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

		# will need seperate losses for low-pass filtered image and high-pass or full image

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
		z_pc, z_pd = self.f_enc_low(image, None)
		
		# Latent Sampling
		latent_sample_low = []

		# Continuous sampling 
		norm_sample_low, mu_low, sigma_low = self.q_dist_low.sample_normal(params=z_pc, train=self.training)
		latent_sample_low.append(norm_sample_low)

		# Discrete sampling
		for ind, alpha in enumerate(z_pd):
			cat_sample_low = self.cat_dist.sample_gumbel_softmax(alpha, train=self.training)
			latent_sample_low.append(cat_sample_low)
			
		z_low = torch.cat(latent_sample_low, dim=-1)

		# Decoding - p(x|z)
		pred_low = self.g_dec_low(z_low)

		if self.training:
			# need to select correct version of image
			iter_loss, _ = self.loss(iter, image, pred_low, z_pc, z_pd) 
			#self.iter_loss += iter_loss / image.shape[1]
			self.iter_loss += iter_loss

		# high frequency bit comes in
		if iter > self.p['start_hi']:

			# how to set prior on cat distribution?
			self.prior_dist_hi, self.q_dist_hi, self.x_dist_hi, self.cat_dist_hi = mutils.discheck(self.p,hi=1,mu=mu_low,log_sigma=sigma_low)
			self.p['z_params_hi']	= self.q_dist_hi.nparams

			# Encoding - p(z2|x) or p(z1 |x,z2)
			z_pc, z_pd = self.f_enc_hi(image, None)
			
			# Latent Sampling
			latent_sample_hi = []

			# Continuous sampling 
			norm_sample_hi, mu_hi, sigma_hi = self.q_dist_hi.sample_normal(params=z_pc, train=self.training)
			latent_sample_hi.append(norm_sample_hi)

			# Discrete sampling
			for ind, alpha in enumerate(z_pd):
				cat_sample_hi = self.cat_dist.sample_gumbel_softmax(alpha, train=self.training)
				latent_sample.append(cat_sample_hi)
				
			z_hi = torch.cat(latent_sample_hi, dim=-1)

			# Decoding - p(x|z)
			pred_hi  = self.g_dec_hi(z_hi)

			if self.training:
				# need high-frequency version of image here
				iter_loss, _ = self.loss(iter, image, pred_hi, z_pc, z_pd) 
				#self.iter_loss += iter_loss / image.shape[1]
				self.iter_loss += iter_loss

		# combine for export?
		z = torch.cat(z_low,z_hi)
		pred = torch.cat(pred_low, pred_hi) # or sum?
		
		if self.err_plot_flag:
			self.err_plot_flag = False
		
		if to_matlab:
			return z.detach().numpy(), foveated.detach().numpy()
			
		elif eval:
			return z, pred

		elif self.enc_mode:
			return z
		
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




class ObservationVAE_segment(Module):
	def __init__(self, p):
		super(ObservationVAE_segment,self).__init__()
		
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
