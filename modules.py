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

class MaskedLinear(Module):
	# fully connected layer for receptive field masking
	def __init__(self, indim, outdim, mask):
		super(MaskedLinear, self).__init__()

		def backward_hook(grad):
			out = grad.clone()
			out[self.mask == 0] = 0 
			return out
		
		self.linear = Linear(indim, outdim, bias=None)
		zmask = zeros([outdim, indim]).byte()
		zmask[mask.type(ByteTensor) ] = 1
		self.mask = zmask
		self.linear.weight.data[zmask == 0] = 0			
		self.linear.weight.register_hook(backward_hook)
	
	def forward(self, x):
			return self.linear(x)
					
class CustomizedLinearFunction(Function):
	"""
	autograd function which masks it's weights by 'mask'.
	"""

	# Note that both forward and backward are @staticmethods
	@staticmethod
	# bias, mask is an optional argument
	def forward(ctx, input, weight, bias=None, mask=None):
		if mask is not None:
			# change weight to 0 where mask == 0
			weight = weight * mask
		output = input.mm(weight.t())
		if bias is not None:
			output += bias.unsqueeze(0).expand_as(output)
		ctx.save_for_backward(input, weight, bias, mask)
		return output

	# This function has only a single output, so it gets only one gradient
	@staticmethod
	def backward(ctx, grad_output):
		# This is a pattern that is very convenient - at the top of backward
		# unpack saved_tensors and initialize all gradients w.r.t. inputs to
		# None. Thanks to the fact that additional trailing Nones are
		# ignored, the return statement is simple even when the function has
		# optional inputs.
		input, weight, bias, mask = ctx.saved_tensors
		grad_input = grad_weight = grad_bias = grad_mask = None

		# These needs_input_grad checks are optional and there only to
		# improve efficiency. If you want to make your code simpler, you can
		# skip them. Returning gradients for inputs that don't require it is
		# not an error.
		if ctx.needs_input_grad[0]:
			grad_input = grad_output.mm(weight)
		if ctx.needs_input_grad[1]:
			grad_weight = grad_output.t().mm(input)
			if mask is not None:
				# change grad_weight to 0 where mask == 0
				grad_weight = grad_weight * mask
		#if bias is not None and ctx.needs_input_grad[2]:
		if ctx.needs_input_grad[2]:
			grad_bias = grad_output.sum(0).squeeze(0)

		return grad_input, grad_weight, grad_bias, grad_mask

class CustomizedLinear(Module):
	def __init__(self, mask, bias=True):
		"""
		extended torch.nn module which mask connection.
		Argumens
		------------------
		mask [torch.tensor]:
			the shape is (n_input_feature, n_output_feature).
			the elements are 0 or 1 which declare un-connected or
			connected.
		bias [bool]:
			flg of bias.
		"""
		super(CustomizedLinear, self).__init__()
		self.input_features = mask.shape[0]
		self.output_features = mask.shape[1]
		if isinstance(mask, Tensor):
			self.mask = mask.type(float).t()
		else:
			self.mask = Tensor(mask, dtype=float).t()

		self.mask = Parameter(self.mask, requires_grad=False)

		# nn.Parameter is a special kind of Tensor, that will get
		# automatically registered as Module's parameter once it's assigned
		# as an attribute. Parameters and buffers need to be registered, or
		# they won't appear in .parameters() (doesn't apply to buffers), and
		# won't be converted when e.g. .cuda() is called. You can use
		# .register_buffer() to register buffers.
		# nn.Parameters require gradients by default.
		self.weight = Parameter(Tensor(self.output_features, self.input_features))

		if bias:
			self.bias = Parameter(Tensor(self.output_features))
		else:
			# You should always register all possible parameters, but the
			# optional ones can be None if you want.
			self.register_parameter('bias', None)
		self.reset_parameters()

		# mask weight
		self.weight.data = self.weight.data * self.mask

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input):
		# See the autograd section for explanation of what happens here.
		return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

	def extra_repr(self):
		# (Optional)Set the extra information about this module. You can test
		# it by printing an object of this class.
		return 'input_features={}, output_features={}, bias={}'.format(
			self.input_features, self.output_features, self.bias is not None
		)
			
class ObsModel(Module):
	# 'A' layer from prednet  
	# apply RF mask retaining dimensions
	
	def __init__(self,p,masks,l):
		super(ObsModel, self).__init__()

		# get mask from from calc_rf
		obs_mask, layer_mask = masks
		self.out_full = p['ldim'][l] 
		
		#self.rf_algo = p['rf_algo']
		# self.use_rf = p['use_rf']
		
		# if p['use_rf']:
		# 	if l>0:
		# 		if p['rf_reduce']:
		# 			on_mask	 = layer_mask['on'][int(l)-1]
		# 			off_mask = layer_mask['off'][int(l)-1]
		# 		else:
		# 			on_mask	 = obs_mask['on'][int(l)]
		# 			off_mask = obs_mask['off'][int(l)]					
		# 	else:
		# 		on_mask	 = obs_mask['on'][int(l)]
		# 		off_mask = obs_mask['off'][int(l)]
				
		# 	self.in_flat = on_mask.shape[0]
		# 	self.fc_c = MaskedLinear(self.in_flat, np.prod(self.out_full), on_mask)
		# 	self.fc_s = MaskedLinear(self.in_flat, np.prod(self.out_full), off_mask)
			
		#else:
			
			#self.in_flat  = np.prod(p['ldim'][l])
			#self.out_full = p['ldim'][l]
			#self.fc1  = Linear(self.in_flat, np.prod(self.out_full), bias=None)

		# apply RF
		#self.mask = CustomizedLinear(mask, bias=None)
		#self.fc1  = Linear(self.in_flat, np.prod(self.out_full), bias=None)

	def forward(self, x):
		#x = self.fc1(self.mask(x.view(-1,self.in_flat)))
		
		# if self.use_rf:
		# 	if self.rf_algo == 'independent':
				
		# 		C = F.relu(self.fc_c(x))
		# 		S = F.relu(self.fc_s(x))		
		# 		x = C+S
					
		# 	elif self.rf_algo == 'shared':
			
		# 		C = self.fc_c(x)
		# 		S = self.fc_s(x)
		# 		x = F.relu(C + S)
			
		# 	elif self.rf_algo == 'stacked':
				
		# 		C = F.relu(self.fc_c(x)) 
		# 		S = F.relu(self.fc_s(x))
		# 		x = F.relu(C + S)
			
		#else:
			#x = self.fc1(x.view(-1,self.in_flat))
			
		return x.view(-1,*self.out_full)


class CelebConvEncoder(Module):
	def __init__(self, p, l):
		super(CelebConvEncoder, self).__init__()

		self.conv1 = Conv2d(3,	32, (4,4), stride=2, padding=1)
		self.conv2 = Conv2d(32, 64, (4,4), stride=2, padding=1)
		self.conv3 = Conv2d(64, 128, (4,4), stride=2, padding=1)
		self.conv4 = Conv2d(128, 128, (4,4), stride=2, padding=1)
		self.bs = p['b']
 
		self.has_con = p['nz_con'][l] is not None 
		self.has_dis = p['nz_dis'][l] is not None 
		 
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['ldim'][l]) 
		self.constrained = l < p['layers']-1 
		
		self.fc1 = Linear(128 * 4 * 4, enc_h) 

		
		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
			

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]} 

		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		h = F.relu(self.fc1(x.view(self.bs, -1)))

		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']


class MNISTConvEncoder(Module):
	def __init__(self, p, l):
		super(MNISTConvEncoder, self).__init__()

		self.conv1 = Conv2d(1,	32, (4,4), stride=2, padding=1)
		self.conv2 = Conv2d(32, 64, (4,4), stride=2, padding=1)
		self.conv3 = Conv2d(64, 64, (4,4), stride=2, padding=1)
		self.bs = p['b']
 
		self.has_con = p['nz_con'][l] is not None 
		self.has_dis = p['nz_dis'][l] is not None 
		 
		self.z_con_dim = 0; self.z_dis_dim = 0
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['ldim'][l]) 
		self.constrained = l < p['layers']-1 
		
		self.fc1 = Linear(64 * 4 * 4, enc_h) 

		
		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
			

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]} 

		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		h = F.relu(self.fc1(x.view(self.bs, -1)))

		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']

class STL10ConvEncoder(Module):
	def __init__(self, p, l):
		super(STL10ConvEncoder, self).__init__()	

		self.conv1 = Conv2d(3,  64, 3, padding=1)
		self.bn1  = BatchNorm2d(64)		
		
		self.conv2 = Conv2d(64, 64, 3, padding=1)
		self.bn2  = BatchNorm2d(64)		
		
		self.mp1   = torch.nn.MaxPool2d(2)
		
		self.conv3 = Conv2d(64,  128, 3, padding=1)
		self.bn3  = BatchNorm2d(128)
		
		self.conv4 = Conv2d(128, 128, 3, padding=1)
		self.bn4  = BatchNorm2d(128)
		
		self.mp2   = torch.nn.MaxPool2d(2)

		self.conv5 = Conv2d(128, 256, 3, padding=1)
		self.bn5  = BatchNorm2d(256)
		
		self.conv6 = Conv2d(256, 256, 3, padding=1)
		self.bn6  = BatchNorm2d(256)
		
		self.mp3   = torch.nn.MaxPool2d(2)
		
		
		self.conv7 = Conv2d(256, 256, 3, padding=1)
		self.bn7  = BatchNorm2d(256)
		self.mp4  = torch.nn.MaxPool2d(2)
		
		self.bs = p['b']
 
		self.has_con = p['nz_con'][l] is not None 
		self.has_dis = p['nz_dis'][l] is not None 
		 
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['ldim'][l]) 
		self.constrained = l < p['layers']-1 
		
		self.fc1 = Linear(256 * 6 * 6, enc_h) 

		
		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
			

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]} 

		x = self.bn1(F.elu(self.conv1(x)))
		x = self.bn2(F.elu(self.conv2(x)))
		
		x = self.mp1(x)
		x = self.bn3(F.elu(self.conv3(x)))
		x = self.bn4(F.elu(self.conv4(x)))
		
		x = self.mp2(x)
		x = self.bn5(F.elu(self.conv5(x)))
		x = self.bn6(F.elu(self.conv6(x)))
		
		x = self.mp3(x)		
		x = self.bn7(F.elu(self.conv7(x)))
		
		x = self.mp4(x)		
		x = x.view(x.size(0), -1)
		h = F.elu(self.fc1(x))
		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']

class STL10ConvDecoder(Module):
	
	def __init__(self, p, l):
		super(STL10ConvDecoder, self).__init__()

		# layer configuration 
		latents = sum(p['z_dim'][l:l+2]) 
		hidden	  = p['enc_h'][l] 
		
		self.fc1 = Linear(latents, 256*6*6)
		
		self.dec1 = ConvTranspose2d(256, 256, 2, stride=2)
		self.bn1  = BatchNorm2d(256)
		
		self.dec2 = ConvTranspose2d(256, 256, 2, stride=2)
		self.bn2  = BatchNorm2d(256)
		self.dec3 = ConvTranspose2d(256, 128, 3, padding=1)
		self.bn3  = BatchNorm2d(128)

		self.dec4 = ConvTranspose2d(128, 128, 2, stride=2)
		self.bn4  = BatchNorm2d(128)
		self.dec5 = ConvTranspose2d(128, 64, 3, padding=1)
		self.bn5  = BatchNorm2d(64)
		
		self.dec6 = ConvTranspose2d(64, 64, 2, stride=2)
		self.bn6  = BatchNorm2d(64)
		
		self.dec7 = ConvTranspose2d(64,3,3,padding=1)
		self.bn7  = BatchNorm2d(3)
		
	def forward(self, x):
		
		x = F.relu(self.fc1(x))
		x = x.view(-1,256, 6, 6)
				
		x = self.bn1(F.elu(self.dec1(x)))
		
		x = self.bn2(F.elu(self.dec2(x)))
		x = self.bn3(F.elu(self.dec3(x)))
		
		x = self.bn4(F.elu(self.dec4(x)))
		x = self.bn5(F.elu(self.dec5(x)))
		
		x = self.bn6(F.elu(self.dec6(x)))
		x = self.bn7(F.elu(self.dec7(x)))

		return x
	
class Encoder(Module): 
	# Error > latent Encoder 
	def __init__(self, p, l): 
		super(Encoder, self).__init__() 
		 
		# layer specific config 
		#enc_h = p['enc_h'][l]*2 if l < p['layers']-1 else p['enc_h'][l] 
		 
		self.has_con = p['nz_con'][l] is not None 
		self.has_dis = p['nz_dis'][l] is not None 
		 
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['ldim'][l]) 
		self.constrained = l < p['layers']-1 
		 
		# image to features 
		self.fc1 = Linear(self.imdim, enc_h) 
		#self.fc2 = Linear(enc_h, enc_h) 
		#self.fc3 = Linear(enc_h+p['z_dim'][l+1] if self.constrained else enc_h, enc_h) 

		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
		#self.apply(utils.init_weights) 
			 
	def forward(self, x, z_q=None):			

		latent_dist = {'con':[], 'dis':[]} 
 
		h = F.relu(self.fc1(x.view(-1, self.imdim)))
		#h = F.relu(self.fc2(h))
		if self.constrained: 
			h = cat((h,z_q), dim=-1) 
			h = F.relu(self.fc3(h)) 
		 
		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']

class ConvEncoder(Module):
	def __init__(self, p, l):
		super(ConvEncoder, self).__init__()
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

		self.conv1 = Conv2d(p['ldim'][l][0], 32, (4,4), stride=2, padding=1)	# (1) 42 x 42
		self.bn1 = BatchNorm2d(32)
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

	def forward(self, x, z_q=None):
		latent_dist = {'con':[], 'dis':[]}
		
		h = x.view(-1, *self.dim)
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
		if self.has_con:
			latent_dist['con'] = self.fc_zp(h)

		if self.has_dis:
			latent_dist['dis'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=1))
		

		return latent_dist['con'], latent_dist['dis']

class Link_lo_hi(Module):
	# not needed?
	# still need to use amended image for loss at level 2
	def __init__(self,p,l):
		
		super(Link_lo_hi, self).__init__()

		self.lo_dim = sum(p['nz_dis'][1]) + p['nz_con'][1] * p['z_params']
		self.hi_dim = sum(p['nz_dis'][0]) + p['nz_con'][0] * p['z_params']

		self.fc1 = Linear(self.low_dim, self.hi_dim)
		#self.fc2 = Linear(2*self.hi_dim, self.hi_dim)

		if self.has_con:
			# features to continuous latent 
			self.fc_zp = Linear(self.hi_dim, p['nz_con'][0] * p['z_params'])
		if self.has_dis:
			# features to categorical latent - can have more than one categorical dist
			self.fc_alphas = Linear(self.hi_dim, sum(p['nz_dis'][0]))
			for a_dim in p['nz_dis'][0]:
				self.fc_alphas.append(Linear(self.hi_dim,a_dim))
			self.fc_alphas = ModuleList(self.fc_alphas)		

	def forward(self, x, z_q=None):

		latent_dist = {'con':[], 'dis':[]}

		h = F.relu(self.fc1(x))
		if self.has_con:
			latent_dist['con'] = self.fc_zp(h)

		if self.has_dis:
			latent_dist['dis'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=1))
		
		return latent_dist['con'], latent_dist['dis']

class ConvEncoder_lo_hi(Module):
	def __init__(self, p, l):
		super(ConvEncoder_lo_hi, self).__init__()
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

		self.conv1 = Conv2d(p['ldim'][l][0], 32, (4,4), stride=2, padding=1)	# (1) 42 x 42
		self.bn1 = BatchNorm2d(32)
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

		self.conv1a = Conv2d(p['ldim'][l][0], 32, (4,4), stride=2, padding=1)	# (1) 42 x 42
		self.bn1a = BatchNorm2d(32)
		self.conv2a = Conv2d(32, 32, (4,4), stride=2, padding=1)
		self.bn2a = BatchNorm2d(32)
		self.conv3a = Conv2d(32, 64, (4,4), stride=2, padding=1)
		self.bn3a = BatchNorm2d(64)
		self.conv4a = Conv2d(64, 64, (4,4), stride=2, padding=1)
		self.bn4a = BatchNorm2d(64) 
		self.conv5a = Conv2d(64, 128,  (4,4), stride=2, padding=1)
		self.bn5a = BatchNorm2d(128)
		self.fc1a = Linear(128*2*2, p['enc_h'][l])	
		self.fc2a = Linear(p['enc_h'][l], p['enc_h'][l]) 

		
				
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
	
		latent_dist = {'con_lo':[], 'dis_lo':[], 'con_hi':[], 'dis_hi':[]}

		# low frequency bit
		lo = self.f(x)

		lo = lo.view(-1, *self.dim)
		lo = F.relu(self.bn1(self.conv1(lo)))
		lo = F.relu(self.bn2(self.conv2(lo)))
		lo = F.relu(self.bn3(self.conv3(lo)))
		lo = F.relu(self.bn4(self.conv4(lo)))
		lo = F.relu(self.bn5(self.conv5(lo)))
		lo = lo.view(x.size(0), -1)
		lo = F.relu(self.fc1(lo))
		lo = F.relu((self.fc2(lo)))

		if self.has_con:
			latent_dist['con_lo'] = self.fc_zp(lo)
		if self.has_dis:
			latent_dist['dis_lo'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis_lo'].append(F.softmax(fc_alpha(lo), dim=1))

		# high frequency
		if self.iteration > self.p['start_hi']-1:
			h = x.view(-1, *self.dim)
			h = F.relu(self.bn1a(self.conv1a(h)))
			h = F.relu(self.bn2a(self.conv2a(h)))
			h = F.relu(self.bn3a(self.conv3a(h)))
			h = F.relu(self.bn4a(self.conv4a(h)))
			h = F.relu(self.bn5a(self.conv5a(h)))
			h = h.view(x.size(0), -1)
			h = F.relu(self.fc1a(h))
			h = F.relu((self.fc2a(h)))

			if self.has_con:
				latent_dist['con_hi'] = self.fc_zp(h)
			if self.has_dis:
				latent_dist['dis_hi'] = []
				for fc_alpha in self.fc_alphas:
					latent_dist['dis_hi'].append(F.softmax(fc_alpha(h), dim=1))
		# if self.constrained:
		# 	h = cat((h, z_q), dim=-1)
		
		# # add high-frequency prediction errors if at appropriate stage of development (check self has field for iteration)
		# if self.e > self.p['start_hi']-1:
		# 	h = cat((h, x), dim=-1)

		# if self.has_con:
		# 	latent_dist['con'] = self.fc_zp(h)

		# if self.has_dis:
		# 	latent_dist['dis'] = []
		# 	for fc_alpha in self.fc_alphas:
		# 		latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=1))
		
		return latent_dist['con_lo'], latent_dist['dis_lo'], latent_dist['con_hi'], latent_dist['dis_hi']


class ConvEncoder_hi(Module):
	def __init__(self, p, l):
		super(ConvEncoder_hi, self).__init__()
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

		self.conv1 = Conv2d(p['ldim'][l][0], 32, (4,4), stride=2, padding=1)	# (1) 42 x 42
		self.bn1 = BatchNorm2d(32)
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

	def forward(self, x, z_q=None):
		latent_dist = {'con':[], 'dis':[]}

		x = x(1) # if returned from transform as tuple
		
		h = x.view(-1, *self.dim)
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
		if self.has_con:
			latent_dist['con'] = self.fc_zp(h)

		if self.has_dis:
			latent_dist['dis'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=1))
		

		return latent_dist['con'], latent_dist['dis']


class ConvDilationEncoder(Module):
	def __init__(self, p, l):
		super(ConvEncoder, self).__init__()
		self.output_dim = sum(p['z_dim'][l:l+2]) * p['z_params']
		self.constrained = l < p['layers']-1
		
		self.has_con = p['nz_con'][l] is not None
		self.has_dis = p['nz_dis'][l] is not None				
		
		self.z_con_dim = 0; self.z_dis_dim = 0;
		if self.has_con:
			self.z_con_dim = p['nz_con'][l] * p['z_params']
		if self.has_dis:
			self.z_dis_dim = sum(p['nz_dis'][l])
			self.n_dis_z   = len(p['nz_dis'][l])
		
		self.z_dim = self.z_con_dim + self.z_dis_dim		

		self.conv1 = Conv2d(p['ldim'][l][0], 32, 6, 1, dilation=3)	# (1) 42 x 42
		self.bn1 = BatchNorm2d(32)
		self.conv2 = Conv2d(32, 32, 6, 1, dilation=2)	 # (2) 21 x 21
		self.bn2 = BatchNorm2d(32)
		self.conv3 = Conv2d(32, 64, 6, 2, dilation=3)	 # (3) 10 x 10
		self.bn3 = BatchNorm2d(64)
		self.conv4 = Conv2d(64, 64, 6, 1, dilation=2)		 # (4) 4 x 4
		self.bn4 = BatchNorm2d(64) 
		self.conv5 = Conv2d(64, 128, 6, 2)		 # (5) 2 x 2 
		self.bn5 = BatchNorm2d(128)
		self.conv6 = Conv2d(128, p['enc_h'][l], 4)	 # (6)	512 x 1
		
		self.fc1 = Linear(p['enc_h'][l], p['enc_h'][l]) 
		
				
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

	def forward(self, x, z_q=None):
		latent_dist = {'con':[], 'dis':[]}
		h = x.view(-1, *self.dim)

		h = F.relu(self.bn1(self.conv1(h)))

		h = F.relu(self.bn2(self.conv2(h)))

		h = F.relu(self.bn3(self.conv3(h)))

		h = F.relu(self.bn4(self.conv4(h)))

		h = F.relu(self.bn5(self.conv5(h)))

		h = F.relu((self.conv6(h)))
		h = h.view(x.size(0), -1)
		h = F.relu(self.fc1(h))
		if self.constrained:
			h = cat((h, z_q), dim=-1)


		if self.has_con:
			latent_dist['con'] = self.fc_zp(h)

		if self.has_dis:
			latent_dist['dis'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=1))
		

		return latent_dist['con'], latent_dist['dis']

class MNISTConvDecoder(Module):
	
	def __init__(self, p, l):
		super(MNISTConvDecoder, self).__init__()

		# layer configuration 
		latents = sum(p['z_dim'][l:l+2]) 
		hidden	  = p['enc_h'][l] 
		
		self.fc1 = Linear(latents, hidden)
		self.fc2 = Linear(hidden,  64*4*4)
		
		self.deconv1 = ConvTranspose2d(64, 32, (4,4),stride=2,padding=1)
		self.deconv2 = ConvTranspose2d(32, 32, (4,4),stride=2,padding=1)
		self.deconv3 = ConvTranspose2d(32, 1,  (4,4),stride=2,padding=1)
		
		
	def forward(self, x):
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.deconv1(x.view(-1,64,4,4)))
		x = F.relu(self.deconv2(x))
		x = sigmoid(self.deconv3(x)) 
		return x

class CelebConvDecoder(Module):
	
	def __init__(self, p, l):
		super(CelebConvDecoder, self).__init__()

		# layer configuration 
		latents = sum(p['z_dim'][l:l+2]) 
		hidden	  = p['enc_h'][l] 
		
		self.fc1 = Linear(latents, hidden)
		self.fc2 = Linear(hidden,  64*4*4)
		
		self.deconv1 = ConvTranspose2d(64, 64, (4,4),stride=2,padding=1)
		self.deconv2 = ConvTranspose2d(64, 32, (4,4),stride=2,padding=1)
		self.deconv3 = ConvTranspose2d(32, 32, (4,4),stride=2,padding=1)
		self.deconv4 = ConvTranspose2d(32, 3,  (4,4),stride=2,padding=1)
		
		
	def forward(self, x):
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.deconv1(x.view(-1,64,4,4)))
		x = F.relu(self.deconv2(x))
		x = F.relu(self.deconv3(x))
		x = sigmoid(self.deconv4(x)) 
		return x
		
class Decoder(Module): 
	# Input:  sample from z(l) 
	# Output: prediction at z(l) 
 
	def __init__(self, p,l): 
		super(Decoder, self).__init__() 
		 
		# layer configuration 
		latents = sum(p['z_dim'][l:l+2]) 
		hidden	  = p['enc_h'][l] 
		 
		# nn init 
		self.dim = p['ldim'][l] 
		self.fc1 = Linear(latents, hidden) 
		self.fc2 = Linear(hidden, hidden) 
		#self.fc3 = Linear(hidden, hidden) 
		self.fc4 = Linear(hidden, np.prod(self.dim)) 
		#self.apply(utils.init_weights) 
 
	def forward(self, x):  
		h1 = tanh(self.fc1(x)) 
		h2 = tanh(self.fc2(h1)) 
		#h3 = tanh(self.fc3(h2)) 
		return self.fc4(h1).view(-1, *self.dim)


class ConvDilationDecoder(Module):
	# keep filter dims pow(2) please
	# 
	def __init__(self, p, l):
		super(ConvDecoder, self).__init__()
		latents = sum(p['z_dim'][l:l+2])
		
		self.conv1 = ConvTranspose2d(latents, 512, 4)	# (1) 2 x 2
		self.bn1 = BatchNorm2d(512)
		self.conv2 = ConvTranspose2d(512, 128, 3, 3)  # (2) 4 x 4
		self.bn2 = BatchNorm2d(128)
		self.conv3 = ConvTranspose2d(128, 64, 6, 1, dilation=2)	 # (3) 10 x 10
		self.bn3 = BatchNorm2d(64)
		self.conv4 = ConvTranspose2d(64, 32, 6, 3)	 # (4) 21 x 21 
		self.bn4 = BatchNorm2d(32)
		self.conv5 = ConvTranspose2d(32, 16, 6, 1, dilation=3) # (6) 
		self.bn5 = BatchNorm2d(16)
		
		self.conv_final = ConvTranspose2d(16, p['ldim'][l][0], 1)
		
		# setup the non-linearity
		self.dim = p['ldim'][l]

	def forward(self, z, a=None, vel_obs=None): 
		
		if not a is None and vel_obs is Nonde:
			z = cat((z, a.squeeze(1), vel_obs.squeeze(1)), dim=-1)
		h = z.view(z.size(0), z.size(1), 1, 1)
		h = F.relu(self.bn1(self.conv1(h)))
		h = F.relu(self.bn2(self.conv2(h)))
		h = F.relu(self.bn3(self.conv3(h)))
		h = F.relu(self.bn4(self.conv4(h)))
		h = F.relu(self.bn5(self.conv5(h)))
		mu_img = self.conv_final(h)
		return mu_img

class ConvDecoder64(Module):
	# keep filter dims pow(2) please
	# 
	def __init__(self, p, l):
		super(ConvDecoder64, self).__init__()
		latents = sum(p['z_dim'][l:l+2])
		
		self.conv1 = ConvTranspose2d(latents, 512, (4,4), stride=2, padding=1)
		self.bn1 = BatchNorm2d(512)
		self.conv2 = ConvTranspose2d(512, 128,	(4,4), stride=2, padding=1)
		self.bn2 = BatchNorm2d(128)
		self.conv3 = ConvTranspose2d(128, 64,  (4,4), stride=2, padding=1)
		self.bn3 = BatchNorm2d(64)
		self.conv4 = ConvTranspose2d(64, 32, (4,4), stride=2, padding=1)
		self.bn4 = BatchNorm2d(32)
		self.conv5 = ConvTranspose2d(32,16,	 (4,4), stride=2, padding=1)
		self.bn5 = BatchNorm2d(16)
		self.conv_final = ConvTranspose2d(16, p['ldim'][l][0],	(4,4), stride=2, padding=1)
		

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
		h = F.relu(self.bn5(self.conv5(h)))
		mu_img = self.conv_final(h)
		return mu_img

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

class Cifar10ConvEncoder(Module):
	def __init__(self, p, l):
		super(Cifar10ConvEncoder, self).__init__()	

		self.enc1 = Conv2d(3, 128, 4, stride=2, padding=1)
		self.bn1  = BatchNorm2d(128)

		self.enc2 = Conv2d(128, 256, 4, stride=2, padding=1)
		self.bn2  = BatchNorm2d(256)

		self.enc3 = Conv2d(256, 512, 4, stride=2, padding=1)
		self.bn3  = BatchNorm2d(512)

		self.enc4 = Conv2d(512, 1024, 4, stride=2, padding=1)
		self.bn4  = BatchNorm2d(1024)
		
		self.bs = p['b']
 
		self.has_con = p['nz_con'][l] is not None 
		self.has_dis = p['nz_dis'][l] is not None 
		 
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['ldim'][l]) 
		self.constrained = l < p['layers']-1 
		
		self.fc1 = Linear(1024*2*2, 128) 

		
		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(128, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(128,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
			

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]} 
		x = F.relu(self.bn1(self.enc1(x)))
		x = F.relu(self.bn2(self.enc2(x)))
		x = F.relu(self.bn3(self.enc3(x)))
		x = F.relu(self.bn4(self.enc4(x)))
		x = x.view(self.bs, -1)
		h = self.fc1(x)
		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']



class Cifar10ConvDecoder(Module):
	
	def __init__(self, p, l):
		super(Cifar10ConvDecoder, self).__init__()

		# layer configuration 
		latents = sum(p['z_dim'][l:l+2]) 
		hidden	  = p['enc_h'][l] 
		
		self.fc1 = Linear(latents, 1024 * 4 * 4)
		self.fcbn1  = BatchNorm2d(1024)
		
		self.dec1 = ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
		self.bn1  = BatchNorm2d(512)
		
		self.dec2 = ConvTranspose2d(512, 256, 4, stride=2, padding=1)
		self.bn2  = BatchNorm2d(256)

		self.dec3 = ConvTranspose2d(256, 3, 4, stride=2, padding=1)
		
		
	def forward(self, x):
		
		x = self.fc1(x)
		x = F.relu(self.fcbn1(x.view(-1,1024, 4, 4)))
		x = F.relu(self.bn1(self.dec1(x)))
		x = F.relu(self.bn2(self.dec2(x)))
		x = self.dec3(x)

		return x

class DynamicModel(Module):
	""" x = f(x,v,P)"""
	"""Approximation of the function underlying hidden state dynamics"""
		
	def __init__(self, p, l):

		super(DynamicModel,self).__init__()
		self.hidden_size = p['lstm_h'][l]
		self.n_layers = p['lstm_l']

		self.BS = p['b']
		self.gpu = p['gpu']
		
		self.has_con = p['nz_con'][l] is not None
		self.has_dis = p['nz_dis'][l] is not None				
		
		self.z_con_dim = 0; self.z_dis_dim = 0
		if self.has_con:
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis:
			self.z_dis_dim = sum(p['nz_dis'][l])
			self.n_dis_z   = sum( [x for x in p['nz_dis'][l]] )
		
		
		self.input_size = self.z_con_dim + self.n_dis_z + p['n_actions']

		if self.has_con:
			# features to continuous latent 
			self.fc_zp = Linear(self.hidden_size, self.z_con_dim)
		if self.has_dis:
			# features to categorical latent
			self.fc_alphas = []
			for a_dim in p['nz_dis'][l]:
				self.fc_alphas.append(Linear(self.hidden_size,a_dim))
			self.fc_alphas = ModuleList(self.fc_alphas)		

		self.lstm = LSTM(input_size=self.input_size, num_layers=self.n_layers, hidden_size=self.hidden_size, batch_first=True)
		#self.linear_out = Linear(self.hidden_size, sum(p['z_dim'][l:l+2]) + 1 + 1)
		self.linear_out = Linear(self.hidden_size, sum(p['z_dim'][l:l+2]) )
		self.reset()
		self.apply(mutils.init_weights)
		
	def reset(self):

		self.lstm_h = Variable(next(self.lstm.parameters()).data.new(self.n_layers,	self.BS, self.hidden_size))
		self.lstm_c = Variable(next(self.lstm.parameters()).data.new(self.n_layers, self.BS, self.hidden_size))

		if self.gpu:
			self.lstm_h = self.lstm_h.cuda() ;
			self.lstm_c = self.lstm_c.cuda() ;
			
		self.lstm_h.zero_()
		self.lstm_c.zero_()

	def loss(self, z_real, z_pred):
		"""
		Input:	l	
		Output: loss, (metrics)
		l [int] - layer for which to calculate loss 
		loss[scalar] - loss for current layer
		metrics [tuple] - loss (detached) for discrete & continuous kl
		"""
		
		splitdim = [self.p['nz_con'][l], sum(self.p['nz_dis'][l])]
		
		con_pred, cat_pred = torch.split(pred_z, splitdim, dim=-1) 
		con_target, cat_target = torch.split(target_z, splitdim, dim=-1) 
		
		cat_target = torch.max(cat_target, 1)[1].type(torch.LongTensor).cuda()
		xentloss = self.xent(cat_pred, cat_target)

		mseloss	 = self.mse(con_pred,  con_target)

		layer_loss += xentloss + mseloss
	
		return layer_loss
			
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


	def forward(self, r, a):
		
		latent_dist = {}

		a = a.view(self.BS, 1, -1)
		r = r.view(r.shape[0], 1, -1)
		
		lstm_input = cat((r, a), dim=-1)
		if lstm_input.dim() == 2:
			lstm_input = lstm_dim.unsqueeze(1)
		lstm_out, hidden = self.lstm(lstm_input, (self.lstm_h, self.lstm_c))
		self.lstm_h = hidden[0] ; self.lstm_c = hidden[1]
		
		#linear_out = self.linear_out(lstm_out)
		#done	= linear_out[:,:,0]
		#rew	   = linear_out[:,:,1]
		
		if self.has_con:
			latent_dist['con'] = self.fc_zp(lstm_out)

		if self.has_dis:
			latent_dist['dis'] = []
			for fc_alpha in self.fc_alphas:
				latent_dist['dis'].append(F.softmax(fc_alpha(lstm_out), dim=1))
		
		linear_out = self.linear_out(lstm_out)
		z_pred = linear_out.view(self.BS, -1, 1, 1)
		#done   = linear_out[:,:,-2]
		#rew	   = linear_out[:,:,-1]

		#return latent_dist['con'], cat(latent_dist['dis'], dim=-1)				
		return z_pred#, done, rew

class ActionNet(Module):

	# It takes as inputs both the latent encoding of the current 
	# frame and the hidden state of the MDN-RNN given past latents 
	# and actions and outputs an action.

	def __init__(self, p, l):
		super(ActionNet, self).__init__()
				
		sum(p['z_dim']),sum(p['z_dim']),p['action_dim'], p['lstm_l']
				
		n_actions = 2
		
		latents = sum(p['z_dim'])
		hidden  = sum(p['lstm_h']) 
		action_dim = p['action_dim']
		rnn_l = p['lstm_l']
		self.BS = p['b']
				
		self.fc1 = Linear(latents+hidden*rnn_l+n_actions, hidden)
		self.fc2 = Linear(hidden, hidden)
		self.lr = Linear(hidden, action_dim)
		self.ud = Linear(hidden, action_dim)
		
		self.reward_history = []
		self.policy_history = Variable(torch.Tensor().cuda())
		self.policy_history.requires_grad = True

		self.reward_episode = []
		self.loss_history   = []
		self.gamma = 0.99

	def loss(self, eval=False):

		R = np.asarray([0.] * self.BS)
		rewards = []

		# Discount future rewards back to the present using gamma
		for r in self.reward_episode[::-1]:
			
			r = np.asarray(r)
			R = r + self.gamma * R
			rewards.insert(0,R)
        
		# Scale rewards
		rewards = torch.FloatTensor(rewards)
		rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
		rewards = Variable(rewards)

		rewards.requires_grad = True
		# Calculate loss
		loss1 = Variable(torch.sum(torch.mul(self.policy_history[:,:,0], rewards).mul(-1),-1))
		loss2 = Variable(torch.sum(torch.mul(self.policy_history[:,:,1], rewards).mul(-1),-1))
		loss1.requires_grad = True
		loss2.requires_grad = True
		
		loss = torch.sum(torch.add(loss1,loss2))
		#Save and intialize episode history counters
		
		return loss
		
	def sample_gumbel(self, shape, eps=1e-20):
		U = rand(shape).cuda()
		return -Variable(log(-log(U + eps) + eps))

	def gumbel_softmax_sample(self, logits, temperature):
		y = logits + self.sample_gumbel(logits.size())
		return F.softmax(y / temperature, dim=-1)

	def gumbel_softmax(self, logits, temperature):
		"""
		ST-gumple-softmax
		input: [*, n_class]
		return: flatten --> [*, n_class] an one-hot vector
		"""
		y = self.gumbel_softmax_sample(logits, temperature)
		shape = y.size()
		_, ind = y.max(dim=-1)
		y_hard = zeros_like(y).view(-1, shape[-1])
		y_hard.scatter_(1, ind.view(-1, 1), 1)
		y_hard = y_hard.view(*shape)
		y_hard = (y_hard - y).detach() + y
		return y_hard.view(-1,3)		

	def forward(self, cur_z, prev_lstmh, prev_a): 
			
		cur_z      = cur_z.view(cur_z.shape[0],-1)
		prev_lstmh = prev_lstmh.view(cur_z.shape[0],-1)
		prev_a     = prev_a.view(cur_z.shape[0], -1)
				
		x = cat((cur_z, prev_lstmh, prev_a), dim=-1)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		lr = self.lr(x)
		ud = self.ud(x)
		lr_action = self.gumbel_softmax(lr,temperature=1.0)
		ud_action = self.gumbel_softmax(ud,temperature=1.0)
		action = stack([lr_action, ud_action], -2).squeeze(0)
		
		return action

class ErrorUnit(Module):
	""" Calculate layer-wise error """
	# Input: prediction, observation at l
	# Output: Error at l

	def __init__(self, p, mask, l):
		super(ErrorUnit,self).__init__()
		self.in_flat = np.prod(p['ldim'][l])
		self.out_full = p['ldim'][l] 
		
		self.err_noise = p['err_noise']
		
		if p['use_rf']:
			# apply RF
			self.mask = CustomizedLinear(mask['full'][l], bias=None)
			self.fc1 = Linear(self.flat, self.flat, bias=None)

		self.f = trutils.Gaussian_Smooth()

	def forward(self, bottom_up, top_down, noise=None, freq_filter=None):

		if freq_filter:
			bottom_up = self.f(bottom_up)
		
		e_up   = F.relu(bottom_up - top_down)
		e_down = F.relu(top_down - bottom_up)
		error  = add(e_up,e_down).view(-1,*self.out_full)

		if noise:
			noise = noise.data.normal_(0, 0.01)
			error =	 error + noise		

		return	error


class Retina(Module):
	# Foveate input at l0 
	# Input:  image data x
	# Output: image data x foveated at (x,y)

	def __init__(self, g, k, patch_noise=None):
		super(Retina, self).__init__()
		self.g = g
		self.k = k
		self.patch_noise = patch_noise

	def foveate(self, x, l):

		phi = []
		size = self.g
		full, fov = self.extract_patch(x, l, size)
		return full, fov

	def extract_patch(self, x, l, size):
		
		# single-batch matlab stuff
		if len(x.shape) == 2:
			x = FloatTensor(x)
			x = x.unsqueeze(0).unsqueeze(0)

		B, C, H, W = x.shape
		
		full = zeros_like(x)
		
		if self.patch_noise:
			full = full.uniform_(0, self.patch_noise)

		
		#l = l * 100
		
		patch = zeros(B,C,self.g, self.g)
		l = torchmax(l, -ones_like(l))
		l = torchmin(l, ones_like(l))
		coords = self.denormalize(H, l).view(B,2)

		try:
			patch_x = coords[:, 0] - (size // 2)
			patch_y = coords[:, 1] - (size // 2)
		except:
			patch_x = coords[0] - (size // 2)
			patch_y = coords[1] - (size // 2)
		

		for i in range(B):
			im = x[i].unsqueeze(dim=0)
			T = im.shape[-1]
			
			try:
				from_x, to_x = patch_x[i], patch_x[i] + size
				from_y, to_y = patch_y[i], patch_y[i] + size
			
			except:
				from_x, to_x = patch_x, patch_x + size
				from_y, to_y = patch_y, patch_y + size
			
			# cast to ints
			from_x, to_x = from_x.item(), to_x.item()
			from_y, to_y = from_y.item(), to_y.item()
			
			x_range = [x for x in range(from_x, to_x)]
			x_lim = range(0, 32)
			
			y_range = [y for y in range(from_y, to_y)]
			y_lim = range(0, 32)

			# todo replace with to/from = max(28, to/from) 
			for xi in range(self.g):
				for yi in range(self.g):
					
					if x_range[xi] in x_lim and y_range[yi] in y_lim:
						patch[i,:,xi,yi] = x[:,:,x_range[xi], y_range[yi]]
						full[i,:,x_range[xi], y_range[yi]] = x[:,:,x_range[xi], y_range[yi]]
					else:
						patch[i,:,xi,yi] = 0 
		
		if x.is_cuda:
			return full.cuda(), patch.cuda()
		else:
			return full, patch

	def denormalize(self, t_size, coords):
		
		return (0.5 * ((coords + 1.0) * t_size)).long()
		#return ((coords + 1.0) * t_size).long()

	def exceeds(self, from_x, to_x, from_y, to_y, T):

		if (
			(from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T)
		):
			return True
		return False


class STL10_patch_ConvEncoder(Module):
	def __init__(self, p, l):
		super(STL10_patch_ConvEncoder, self).__init__()	

		self.conv1 = Conv2d(1,  32, 3, padding=0) # 7 to 5
		self.bn1  = BatchNorm2d(32)		
		
		self.conv2 = Conv2d(32, 64, 3, padding=0) # 5 to 3
		self.bn2  = BatchNorm2d(64)

		self.conv3 = Conv2d(64, 128, 3, padding=0) # 3 to 1
		self.bn3  = BatchNorm2d(128)
				
		self.bs = p['b']
 
		self.has_con = p['nz_con'][l] is not None 
		self.has_dis = p['nz_dis'][l] is not None 
		 
		self.z_con_dim = 0; self.z_dis_dim = 0
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['ldim'][l]) 
		self.constrained = l < p['layers']-1 
		
		self.fc1 = Linear(128, enc_h) 

		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
			

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]} 

		x = self.bn1(F.elu(self.conv1(x)))
		x = self.bn2(F.elu(self.conv2(x)))
		x = self.bn3(F.elu(self.conv3(x)))
		x = x.view(x.size(0), -1)
		h = F.elu(self.fc1(x))
		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']

class STL10_patch_ConvDecoder(Module):
	
	def __init__(self, p, l):
		super(STL10_patch_ConvDecoder, self).__init__()

		self.bs = 200

		# layer configuration 
		self.latents = 75
		self.hidden	 = 100 
		
		self.fc1 = Linear(self.latents, self.hidden)
		self.fc2 = Linear(self.hidden, 128)
		
		self.bn1  = BatchNorm2d(128)
		
		self.dec1 = ConvTranspose2d(128, 64, 3)
		self.bn2  = BatchNorm2d(64)

		self.dec2 = ConvTranspose2d(64, 32, 3)
		self.bn3  = BatchNorm2d(32)

		self.dec3 = ConvTranspose2d(32, 1, 3)
		
	def forward(self, x):
		# (B x C x H x W)
		#x = x.view(200,-1)
		x = F.elu(self.fc1(x))
		#x = x.view(-1,1,1,20)
		x = F.elu(self.fc2(x))

		x = x.view(-1,128,1,1)
		x = self.bn1(x)
				
		x = self.bn2(F.elu(self.dec1(x)))
		
		x = self.bn3(F.elu(self.dec2(x)))
		x = self.dec3(x)
		
		return x

# class STL10ConvEncoder_freq(Module):
# 	def __init__(self, p, l):
# 		super(STL10ConvEncoder_freq, self).__init__()	

# 		self.mag_conv1 = Conv2d(1,  64, 35,stride=3, padding=1)
		
# 		self.conv1 = Conv2d(1,  64, 3, padding=1)
# 		self.bn1  = BatchNorm2d(64)		
		
# 		self.conv2 = Conv2d(64, 64, 3, padding=1)
# 		self.bn2  = BatchNorm2d(64)		
		
# 		self.mp1   = torch.nn.MaxPool2d(2)
		
# 		self.conv3 = Conv2d(64,  128, 3, padding=1)
# 		self.bn3  = BatchNorm2d(128)
		
# 		self.conv4 = Conv2d(128, 128, 3, padding=1)
# 		self.bn4  = BatchNorm2d(128)
		
# 		self.mp2   = torch.nn.MaxPool2d(2)

# 		self.conv5 = Conv2d(128, 256, 3, padding=1)
# 		self.bn5  = BatchNorm2d(256)
		
# 		self.conv6 = Conv2d(256, 256, 3, padding=1)
# 		self.bn6  = BatchNorm2d(256)
		
# 		self.mp3   = torch.nn.MaxPool2d(2)
		
		
# 		self.conv7 = Conv2d(256, 256, 3, padding=1)
# 		self.bn7  = BatchNorm2d(256)
# 		self.mp4  = torch.nn.MaxPool2d(2)
		
# 		self.bs = p['b']
 
# 		self.has_con = p['nz_con'][l] is not None 
# 		self.has_dis = p['nz_dis'][l] is not None 
		 
# 		self.z_con_dim = 0; self.z_dis_dim = 0; 
# 		if self.has_con: 
# 			self.z_con_dim = p['nz_con'][l] 
# 		if self.has_dis: 
# 			self.z_dis_dim = sum(p['nz_dis'][l]) 
# 			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
# 		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
# 		enc_h = p['enc_h'][l] 
# 		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
# 		self.imdim = np.prod(p['ldim'][l]) 
# 		self.constrained = l < p['layers']-1 
		
# 		self.fc1 = Linear(256*4 , enc_h) 
# #		self.fc1 = Linear(256 * 5 * 5, enc_h) 

		
# 		if self.has_con: 
# 			# features to continuous latent	 
# 			self.fc_zp = Linear(enc_h, out_dim) 
# 		if self.has_dis: 
# 			# features to categorical latent 
# 			self.fc_alphas = [] 
# 			for a_dim in p['nz_dis'][l]: 
# 				self.fc_alphas.append(Linear(enc_h,a_dim)) 
# 			self.fc_alphas = ModuleList(self.fc_alphas) 
			

# 	def forward(self, x, z_q=None):
		
# 		latent_dist = {'con':[], 'dis':[]} 

# 		x = self.bn1(F.elu(self.mag_conv1(x)))
# 		x = self.bn2(F.elu(self.conv2(x)))
		
# 		x = self.mp1(x)
# 		x = self.bn3(F.elu(self.conv3(x)))
# 		x = self.bn4(F.elu(self.conv4(x)))
		
# 		x = self.mp2(x)
# 		x = self.bn5(F.elu(self.conv5(x)))
# 		x = self.bn6(F.elu(self.conv6(x)))
		
# 		x = self.mp3(x)		
# 		x = self.bn7(F.elu(self.conv7(x)))
		
# 		#x = self.mp4(x)		
# 		x = x.view(x.size(0), -1)
# 		h = F.elu(self.fc1(x))
# 		if self.has_con: 
# 			latent_dist['con'] = self.fc_zp(h) 
 
# 		if self.has_dis: 
# 			latent_dist['dis'] = [] 
# 			for fc_alpha in self.fc_alphas: 
# 				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
# 		return latent_dist['con'], latent_dist['dis']

# class STL10ConvDecoder_freq(Module):
	
# 	def __init__(self, p, l):
# 		super(STL10ConvDecoder_freq, self).__init__()

# 		# layer configuration 
# 		self.latents = sum(p['z_dim'][l:l+2]) 
# 		self.hidden	  = p['enc_h'][l] 

# 		self.fc1 = Linear(self.latents, self.hidden)
# 		self.fc2 = Linear(self.hidden, 256*4)

# 		self.dec1 = ConvTranspose2d(256, 256, 3)#, dilation=2,padding=2)
# 		self.bn1  = BatchNorm2d(256)
		
# 		self.dec2 = ConvTranspose2d(256, 256, 3,dilation=2)
# 		self.bn2  = BatchNorm2d(256)
# 		self.dec3 = ConvTranspose2d(256, 128, 3,dilation=2)
# 		self.bn3  = BatchNorm2d(128)

# 		self.dec4 = ConvTranspose2d(128, 128, 3,dilation=1)
# 		self.bn4  = BatchNorm2d(128)
# 		self.dec5 = ConvTranspose2d(128, 64, 3,dilation=3)
# 		self.bn5  = BatchNorm2d(64)
		
# 		self.dec6 = ConvTranspose2d(64, 64, 3)#,dilation=1)
# 		self.bn6  = BatchNorm2d(64)
		
# 		self.dec7 = ConvTranspose2d(64,1,35,stride=3,padding=1)
# 		self.bn7  = BatchNorm2d(1)
		
# 	def forward(self, x, z_q=None):

# 		x = F.relu(self.fc1(x))
# 		x = F.relu(self.fc2(x))

# 		x = x.view(-1, 256, 2, 2)
				
# 		x = self.bn1(F.elu(self.dec1(x)))
		
# 		x = self.bn2(F.elu(self.dec2(x)))
# 		x = self.bn3(F.elu(self.dec3(x)))
		
# 		x = self.bn4(F.elu(self.dec4(x)))
# 		x = self.bn5(F.elu(self.dec5(x)))
		
# 		x = self.bn6(F.elu(self.dec6(x)))
# 		x = self.bn7(F.elu(self.dec7(x)))

# 		return x


class STL10ConvEncoder_freq(Module):
	def __init__(self, p, l):
		super(STL10ConvEncoder_freq, self).__init__()	

		self.conv1 = Conv2d(3,  64, 3, padding=1)
		self.bn1  = BatchNorm2d(64)		
		
		self.conv2 = Conv2d(64, 64, 3, padding=1)
		self.bn2  = BatchNorm2d(64)		
		
		self.mp1   = torch.nn.MaxPool2d(2)
		
		self.conv3 = Conv2d(64,  128, 3, padding=1)
		self.bn3  = BatchNorm2d(128)
		
		self.conv4 = Conv2d(128, 128, 3, padding=1)
		self.bn4  = BatchNorm2d(128)
		
		self.mp2   = torch.nn.MaxPool2d(2)

		self.conv5 = Conv2d(128, 256, 3, padding=1)
		self.bn5  = BatchNorm2d(256)
		
		self.conv6 = Conv2d(256, 256, 3, padding=1)
		self.bn6  = BatchNorm2d(256)
		
		self.mp3   = torch.nn.MaxPool2d(2)
		
		
		self.conv7 = Conv2d(256, 256, 3, padding=1)
		self.bn7  = BatchNorm2d(256)
		self.mp4  = torch.nn.MaxPool2d(2)
		
		self.bs = p['b']
 
		self.has_con = p['nz_con'][l] is not None 
		self.has_dis = p['nz_dis'][l] is not None 
		 
		self.z_con_dim = 0; self.z_dis_dim = 0; 
		if self.has_con: 
			self.z_con_dim = p['nz_con'][l] 
		if self.has_dis: 
			self.z_dis_dim = sum(p['nz_dis'][l]) 
			self.n_dis_z   = len(p['nz_dis'][l]) 
			 
		self.z_dim = self.z_con_dim + self.z_dis_dim 
			 
		enc_h = p['enc_h'][l] 
		out_dim = sum(p['nz_con'][l:l+2]) * p['z_params'] 
		self.imdim = np.prod(p['ldim'][l]) 
		self.constrained = l < p['layers']-1 
		
		self.fc1 = Linear(256 * 6 * 6, enc_h) 

		
		if self.has_con: 
			# features to continuous latent	 
			self.fc_zp = Linear(enc_h, out_dim) 
		if self.has_dis: 
			# features to categorical latent 
			self.fc_alphas = [] 
			for a_dim in p['nz_dis'][l]: 
				self.fc_alphas.append(Linear(enc_h,a_dim)) 
			self.fc_alphas = ModuleList(self.fc_alphas) 
			

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]} 

		x = self.bn1(F.elu(self.conv1(x)))
		x = self.bn2(F.elu(self.conv2(x)))
		
		x = self.mp1(x)
		x = self.bn3(F.elu(self.conv3(x)))
		x = self.bn4(F.elu(self.conv4(x)))
		
		x = self.mp2(x)
		x = self.bn5(F.elu(self.conv5(x)))
		x = self.bn6(F.elu(self.conv6(x)))
		
		x = self.mp3(x)		
		x = self.bn7(F.elu(self.conv7(x)))
		
		x = self.mp4(x)		
		x = x.view(x.size(0), -1)
		h = F.elu(self.fc1(x))
		if self.has_con: 
			latent_dist['con'] = self.fc_zp(h) 
 
		if self.has_dis: 
			latent_dist['dis'] = [] 
			for fc_alpha in self.fc_alphas: 
				latent_dist['dis'].append(F.softmax(fc_alpha(h), dim=-1)) 
		 
		return latent_dist['con'], latent_dist['dis']

class STL10ConvDecoder_freq(Module):
	
	def __init__(self, p, l):
		super(STL10ConvDecoder_freq, self).__init__()

		# layer configuration 
		latents = sum(p['z_dim'][l:l+2]) 
		hidden	  = p['enc_h'][l] 
		
		self.fc1 = Linear(latents, 256*6*6)
		
		self.dec1 = ConvTranspose2d(256, 256, 2, stride=2)
		self.bn1  = BatchNorm2d(256)
		
		self.dec2 = ConvTranspose2d(256, 256, 2, stride=2)
		self.bn2  = BatchNorm2d(256)
		self.dec3 = ConvTranspose2d(256, 128, 3, padding=1)
		self.bn3  = BatchNorm2d(128)

		self.dec4 = ConvTranspose2d(128, 128, 2, stride=2)
		self.bn4  = BatchNorm2d(128)
		self.dec5 = ConvTranspose2d(128, 64, 3, padding=1)
		self.bn5  = BatchNorm2d(64)
		
		self.dec6 = ConvTranspose2d(64, 64, 2, stride=2)
		self.bn6  = BatchNorm2d(64)
		
		self.dec7 = ConvTranspose2d(64,3,3,padding=1)
		self.bn7  = BatchNorm2d(3)
		
	def forward(self, x):
		
		x = F.relu(self.fc1(x))
		x = x.view(-1,256, 6, 6)
				
		x = self.bn1(F.elu(self.dec1(x)))
		
		x = self.bn2(F.elu(self.dec2(x)))
		x = self.bn3(F.elu(self.dec3(x)))
		
		x = self.bn4(F.elu(self.dec4(x)))
		x = self.bn5(F.elu(self.dec5(x)))
		
		x = self.bn6(F.elu(self.dec6(x)))
		x = self.bn7(F.elu(self.dec7(x)))

		return x