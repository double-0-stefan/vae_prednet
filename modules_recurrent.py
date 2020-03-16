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


class SymRCLayer(Module):
    # based on 'Recurrent Convolutional Neural Networks: A Better Model of Biological Object Recognition'
    # Courtney J. Spoerer, Patrick McClure and Nikolaus Kriegeskorte
    # Front. Psychol., 12 September 2017 | https://doi.org/10.3389/fpsyg.2017.01551
        
    def __init__(self, chan1, chan2, kernel, stride, padding):
        super(SymRCLayer, self).__init__()

        self.conv_fwd     = Conv2D(chan1, chan2, kernel, stride=stride, padding=padding)
        self.conv_bac     = ConvTranspose2d(chan2, chan1, kernel, stride=stride, padding=padding) # is this right?
        self.conv_lat_fwd = Conv2D(chan2, chan1, kernel, stride=1, padding=1)
        self.conv_lat_bac = Conv2D(chan1, chan2, kernel, stride=1, padding=1)

    def forward(self, x_fwd, x_bac=None, x_lat_fwd=None, x_lat_bac=None):
        # note: fully symmetric. performs role of encoder and decoder
        # in first two sections lateral, fwd, and bac will have already gone through a relu
        # forward
        if not x_lat_fwd and not x_bac:
            fwd = self.conv_fwd(x_fwd)
        elif not x_bac:
            fwd = self.conv_fwd(x_fwd) + x_lat_fwd
        elif not x_lat_fwd:
            fwd = self.conv_fwd(x_fwd) + x_bac
        else:
            fwd = self.conv_fwd(x_fwd) + x_bac + x_lat_fwd

        # backwards
        if not x_lat_bac and not x_fwd:
            bac = self.conv_bac(x_bac)
        elif not x_bac:
            bac = self.conv_bac(x_bac) + x_lat_bac
        elif not x_lat_bac:
            bac = self.conv_bac(x_bac) + x_fwd
        else:
            bac = self.conv_bac(x_bac) + x_fwd + x_lat_bac

        # lateral - (uses whatever has been worked out above)
        lat_fwd = self.conv_lat_fwd(fwd)
        lat_bac = self.conv_lat_bac(bac)

        return fwd, bac, lat_fwd, lat_bac
        
class RCNN_encoder_decoder(Module):
	def __init__(self, p, l):
		super(RCNN_encoder_decoder, self).__init__()

        self.T = p['iterations']
        self.has_con = p['nz_con'][l] is not None
		self.has_dis = p['nz_dis'][l] is not None	
		self.z_con_dim = 0; self.z_dis_dim = 0
		if self.has_con:
			self.z_con_dim = p['nz_con'][l] * p['z_params']
		if self.has_dis:
			self.z_dis_dim = sum(p['nz_dis'][l])
			self.n_dis_z   = len(p['nz_dis'][l])
		self.z_dim = self.z_con_dim + self.z_dis_dim

        self.conv1  = SymRCLayer(p['ldim'][l][0], 32, 3, stride=1, padding=1)
        self.bn_fwd1 = BatchNorm2d(32)
        self.bn_lat_fwd1 = = BatchNorm2d(p['ldim'][l][0])
        
        self.conv2  = SymRCLayer(32, p['enc_h'][l], 3, stride=1, padding=1)

        self.bn_fwd2 = BatchNorm2d(p['enc_h'][l])
        self.bn_lat_fwd2 = BatchNorm2d(32)
        self.bn_bac2 = BatchNorm2d(32)
        self.bn_lat_bac2 = BatchNorm2d(p['enc_h'][l])

        # linear not recurrent
        if self.has_con:
			# features to continuous latent 
			self.fc_zp = Linear(p['enc_h'][l], self.z_con_dim)
		if self.has_dis:
			# features to categorical latent
			self.fc_alphas = []
			for a_dim in p['nz_dis'][l]:
				self.fc_alphas.append(Linear(p['enc_h'][l],a_dim))
			self.fc_alphas = ModuleList(self.fc_alphas)		

    def forward(self, x): # x here is the image
        
        x1_fwd, x1_bac, x1_lat_fwd, x1_lat_bac, x2_fwd, x2_bac, x2_lat_fwd, x2_lat_bac, x3_bac = None

        for t in range(self.T):

            x1_fwd, x1_bac, x1_lat_fwd, x1_lat_bac = self.conv1(x, x2_bac, x1_lat_fwd, x1_lat_bac)
            x1_fwd = F.relu(self.bn_fwd1(x1_fwd))
            x1_lat_fwd = F.relu(self.bn_lat_fwd1(x1_lat_fwd)  # ??? nb x1_lat_bac doesn't need to get Relu'd at level 1
            
            if t > 0:
                x2_fwd, x2_bac, x2_lat_fwd, x2_lat_bac = self.conv2(x1_fwd, x3_bac, x2_lat_fwd, x2_lat_bac)
                x2_fwd = F.relu(self.bn_fwd2(x2_fwd))
                x2_lat_fwd = F.relu(self.bn_lat_fwd2(x2_lat_fwd))
                x2_bac = F.relu(self.bn_bac2(x2_bac))
                x2_lat_bac = F.relu(self.bn_lat_bac2(x2_lat_bac))

            if t > 1:





class RCNN_encoder_decoder(Module):
	def __init__(self, p, l):
		super(RCNN_encoder_decoder, self).__init__()
		self.output_dim = sum(p['z_dim'][l:l+2]) * p['z_params']
		self.constrained = l < p['layers']-1
		
		self.has_con = p['nz_con'][l] is not None
		self.has_dis = p['nz_dis'][l] is not None	
        			
		
		self.z_con_dim = 0; self.z_dis_dim = 0
		if self.has_con:
			self.z_con_dim = p['nz_con'][l] * p['z_params']
		if self.has_dis:
			self.z_dis_dim = sum(p['nz_dis'][l])
			self.n_dis_z   = len(p['nz_dis'][l])
		
		self.z_dim = self.z_con_dim + self.z_dis_dim

        # Will also need symetric recurrent linear layer

        
        self.bn_fwd       = BatchNorm2d(p['enc_h'])
        self.bn_bac       = BatchNorm2d(p['enc_h'])
        self.conv1        = SymRCLayer(chan1=p['ldim'][l][0], chan2=p['enc_h'], kernel=3, stride=1, padding=1)


		self.conv2 = Conv2d(32, 32, (4,4), stride=2, padding=1)
		self.bn2 = BatchNorm2d(32)
		self.conv3 = Conv2d(32, 64, (4,4), stride=2, padding=1)
		self.bn3 = BatchNorm2d(64)
		self.conv4 = Conv2d(64, 64, (4,4), stride=2, padding=1)
		self.bn4 = BatchNorm2d(64) 
		self.conv5 = Conv2d(64, 128,  (4,4), stride=2, padding=1)
		self.bn5 = BatchNorm2d(128)
		self.fc1 = Linear(128*2*2, p['enc_h'][l])	
		
		self.fc2 = Linear(p['enc_h'][l], p['enc_h'][l]) 
		
				
		if self.has_con:
			# features to continuous latent 
			self.fc_zp = Linear(p['enc_h'][l], self.z_con_dim)
		if self.has_dis:
			# features to categorical latent
			self.fc_alphas = []
			for a_dim in p['nz_dis'][l]:

				self.fc_alphas.append(Linear(p['enc_h'][l],a_dim))
			self.fc_alphas = ModuleList(self.fc_alphas)		

		# setup the non-linearity
		self.dim = p['ldim'][l]
		self.apply(mutils.init_weights)

		# Gaussian smoothing for low frequencies
		self.f = trutils.Gaussian_Smooth()

	def forward(self, x, z_q=None, image=None):
		# needs to deal with both filtered image AND bottom-up input from high frequency encoder
		# as this is the low-frequency encoder, z_q is None and x is bottom-up input/prediction errors from below
		# image needs to be filtered as it is the 'true' input to this encoder
	
		latent_dist = {'con':[], 'dis':[]}

		h = self.f(x) 
		
		h = h.view(-1, *self.dim)
		h = F.relu(self.bn1(self.conv1(h)))
		h = F.relu(self.bn2(self.conv2(h)))
		h = F.relu(self.bn3(self.conv3(h)))
		h = F.relu(self.bn4(self.conv4(h)))
		h = F.relu(self.bn5(self.conv5(h)))
		h = h.view(x.size(0), -1)
		h = F.relu(self.fc1(h))
		h = F.relu((self.fc2(h)))
		if self.constrained:
			h = cat((h, z_q), dim=-1)
		
		# add high-frequency prediction errors if at appropriate stage of development (check self has field for iteration)
		if self.e > self.p['start_hi']-1:
			h = cat((h, x), dim=-1)

		if self.has_con:
			latent_dist['con'] = self.fc_zp(h)

		if self.has_dis:
			latent_dist['dis'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=1))
		
		return latent_dist['con'], latent_dist['dis']



class ConvDecoder(Module):
	# keep filter dims pow(2) please
	# 
	def __init__(self, p, l):
		super(ConvDecoder, self).__init__()
		latents = sum(p['z_dim'][l:l+2])
		
		self.conv1 = ConvTranspose2d(latents, 512,	4, 2)
		self.bn1 = BatchNorm2d(512)
		self.conv2 = ConvTranspose2d(512, 128,	4, 2)
		self.bn2 = BatchNorm2d(128)
		self.conv3 = ConvTranspose2d(128, 64,  4, 2)
		self.bn3 = BatchNorm2d(64)
		self.conv4 = ConvTranspose2d(64, 32, 4, 2, padding=2)
		self.bn4 = BatchNorm2d(32)
		self.conv_final = ConvTranspose2d(32, p['ldim'][l][0],	(4,4), stride=2, padding=1)
		

		# setup the non-linearity
		self.dim = p['ldim'][l]

	def forward(self, z, a=None, vel_obs=None): 
		
		if not a is None and vel_obs is None:
			z = cat((z, a.squeeze(1), vel_obs.squeeze(1)), dim=-1)
		h = z.view(z.size(0), z.size(1), 1, 1)
		h = F.relu(self.bn1(self.conv1(h)))
		h = F.relu(self.bn2(self.conv2(h)))
		h = F.relu(self.bn3(self.conv3(h)))
		h = F.relu(self.bn4(self.conv4(h)))
		mu_img = self.conv_final(h)
		return mu_img