import os
import math
from torch import nn, max
import logging
import urllib.request as request
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import torchvision.transforms.functional as TTF
from torch import FloatTensor
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import scipy.io as sio
from numbers import Number
import numpy as np
from itertools import chain, product
from collections.abc import Iterable
from torch.utils.data import TensorDataset
from utils import dist 
#from utils import blur_utils
import time 
import torch
import glob
import torch
import kornia.filters
from utils import transform_utils as trutils

#from animalai.envs import UnityEnvironment
#from animalai.envs.arena_config import ArenaConfig
#from data.MovingMNIST import MovingMNIST
from torchvision.utils import save_image

def generate_env_data(p):

	vals_per_action = 3

	env_name = 'env/AnimalAI'
	env = UnityEnvironment( n_arenas=p['n_arenas'],
							file_name=env_name)

	files = glob.glob("examples/configs/env_configs/*.yaml") # list of all .yaml files in a directory 


	for file in files:
		
		img_tensor = torch.zeros(p['n_arenas'], p['n_trials'], p['n_steps'], 3, 84, 84)
		act_tensor = torch.zeros(p['n_arenas'], p['n_trials'], p['n_steps'], 2, 1)

		config = ArenaConfig(file)

		iteration = 0
		for t_o in range(p['n_trials']):

			obs = env.reset(arenas_configurations=config, train_mode=True)['Learner']
			action = {}

			# First action in new environment - don't do anything 
			rand_action = torch.randint(0,3,(p['n_actions']*p['n_arenas'],1))
			rand_action = rand_action.cuda() if p['gpu'] else rand_action
			action['Learner'] = rand_action
			
			for t_i in range(p['n_steps']):
					
				# run batch
				info 	 =  env.step(vector_action=action)['Learner']
				vis_obs  = torch.FloatTensor(info.visual_observations[0]).cuda().permute(0,-1,1,2)
				vel_obs  = torch.tensor(info.vector_observations).cuda()
				text_obs = info.text_observations
				reward 	 = info.rewards				
				
				img_tensor[:, t_o, t_i] = vis_obs
				act_tensor[:, t_o, t_i] = rand_action.view(p['n_arenas'], p['n_actions'], 1)

			
		torch.save(img_tensor, 'imgs'+file.split('.')[0].split('\\')[1]+'.pt') 
		torch.save(act_tensor, 'acts'+file.split('.')[0].split('\\')[1]+'.pt') 
	env.close()


def roll(x, shift, dim, fill_pad=None):
	
    if 0 == shift:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift).cuda())
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim))), gap], dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim)).cuda())
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([gap, x.index_select(dim, torch.arange(shift).cuda())], dim=dim)
		

def train_val_split(p, dataset):

		shuffle_dataset = True 
		random_seed = 34532
		validation_split = 0
		if hasattr(dataset, 'data'):
			dataset_size = len(dataset.data)
		else:
			dataset_size = len(dataset)
	
		indices = list(range(dataset_size))
		split = int(np.floor(validation_split * dataset_size))
		if shuffle_dataset :
			#np.random.seed(random_seed)
			np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]

		# Creating PT data samplers and loaders:
		train_sampler = SubsetRandomSampler(train_indices)
		valid_sampler = SubsetRandomSampler(val_indices)

		train_loader = torch.utils.data.DataLoader(dataset, batch_size=p['b'], #pin_memory=True,
												   sampler=train_sampler, num_workers=0)
		validation_loader = torch.utils.data.DataLoader(dataset, batch_size=p['b'], 
														sampler=train_sampler)
		
		return train_loader, validation_loader


def get_dataset(p, split='train', transform=None, static=False, exp=None,
				target_transform=None, download=True, path='data', from_matlab=False):
	
	# exp = integer corresponding to animalai environment configuration
	
	name = p['dataset']
	batch_size = p['b']
	train = (split == 'train')
	root = os.path.join(os.path.realpath(path), name)
	
	if name == 'animalai':

		# p['n_arenas'] = 20
		# p['n_trials'] = 5
		# p['n_steps']  = 20
		# p['n_actions'] = 2
		# model_inner = 4 

		exps = glob.glob('data/animalai/imgs*.pt') # list of all .yaml files in a directory 
		exps = [x.split('imgs')[1] for x in exps]

		im_data  = [] ;  act_data = []

		_imshape  = (-1, 3, 84, 84) if static else (-1, p['n_steps'],3, 84, 84)
		_actshape = (-1, 2, 3) if static else (-1, p['n_steps'], 2, 3)

		for x in exps:
			im_file  = glob.glob('data/animalai/imgs{}'.format(x)) 
			act_file = glob.glob('data/animalai/acts{}'.format(x)) 
			im_data.append(torch.load(im_file[0]).view(*_imshape).cpu())
			act_data.append(torch.load(act_file[0]).view(*_actshape).cpu())

		# sort validation data
		im_tensors  = torch.cat(im_data,  dim=0)
		act_tensors = torch.cat(act_data, dim=0)
		#if static:
			# repeat single image n_model (meta) times
		#	im_tensors = im_tensors.unsqueeze(1).expand(-1,p['model_inner'],*_imshape[1:])

		dataset = TensorDataset(*[im_tensors, act_tensors])
		
		return train_val_split(p, dataset)
			
	elif name == 'voc':
		data = datasets.VOCSegmentation(root=root, year='2012', image_set='train',
				download=True, transform=transforms.Compose([
					#transforms.Grayscale(num_output_channels=1),
					transforms.ToTensor(),
					Gaussian_Smooth()
				])
		)

		train =  DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
		test  =  DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)

		return train, test
		

	elif name == 'stl10' and p['exp_name'] == 'stl10_patch':
		
		data =	datasets.STL10(root=root,
							  split=split,
							  transform=transforms.Compose([
								transforms.RandomCrop(p['sb_patch_size'], padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
								transforms.Grayscale(num_output_channels=1),
								transforms.ToTensor(),
								#transforms.Normalize((0.5),(0.5)),
															
								]),
							  download=True)
		
		train =  DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
		test  =  DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)

		return train, test

	elif name == 'stl10' and p['exp_name'] == 'stl10_freq':

		# custom transform - will need to split image into low and high frequency components
		# or perhaps not - perhaps just have two streams within encoder that have kernels of different sizes
		# will ?want to initialise thise at different times
		# so start the low frequency stuff earlier in training?
		
	#	gauss = get_gaussian_kernel();#kornia.filters.GaussianBlur((15, 15), (10, 10))

		data =	datasets.STL10(root=root,
							  split=split,
							  transform=transforms.Compose([

								#transforms.Grayscale(num_output_channels=1),
								transforms.ToTensor(),
								#trutils.Gaussian_Smooth(),
								
								#transforms.Normalize((0.5),(0.5)),
															
								]),
							  download=True)

		
		train =  DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
		test  =  DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)

		return train, test

	elif name == 'stl10':
		
		data =	datasets.STL10(root=root,
							  split=split,
							  transform=transforms.Compose([
								#transforms.Grayscale(num_output_channels=1),
								transforms.ToTensor(),
								#transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
															
								]),
							  download=True)
		
		train =  DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
		test  =  DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)

		return train, test

	

	elif name == 'cifar10':
		
		data =	datasets.CIFAR10(root=root,
							  train=split,
							  transform=transforms.Compose([
								#transforms.Grayscale(num_output_channels=3),
								transforms.ToTensor(),
								transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
															
								]),
							  download=True)
		
		train =  DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
		test  =  DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)

		return train, test
							  

	elif name == 'mnist':
		
		root = os.path.join(root, 'processed')
		
		class MNISTransform(object):
			def __init__(self, b,t,dim):
				self.b = b
				self.t = t
				self.dim = dim
				self.norm = torchvision.transforms.Normalize(0.1307, 0.3081, inplace=False)

			def __call__(self, image):
				
				image = TTF.to_tensor(image) # (batch, c, h, w)
				image = self.norm(image)
				new_im = torch.zeros(image.shape[0],32,32)
				new_im[:,2:30,2:30] = image
				image = new_im.unsqueeze(1)	 # (batch, 1, c, h, w)
				#print(self.dim)
				
				image = image.expand(self.t,*self.dim) # (batch, t, c, h, w)

				return image
		if static:
			transform = MNISTransform(p['b'],1,p['imdim'])
		else:
			transform = MNISTransform(p['b'],p['n_steps'],p['imdim'])

		data = datasets.MNIST(root=root, 
							 train=True, download=True,
							 transform=transform)
							 
		if from_matlab:
			return data.data[:batch_size].numpy()
		
		return train_val_split(p, data)
		
	elif name == 'moving_mnist':
		
		#root = os.path.join(root, 'processed')		
        #
		#
		#train = MovingMNIST(root='.data/mnist', transform=transforms.ToTensor(), train=True,  download=True)
		#test  = MovingMNIST(root='.data/mnist', transform=transforms.ToTensor(), train=False, download=True)
		#
		#train = torch.utils.data.DataLoader(train, batch_size=p['b'], num_workers=0)
		#test = torch.utils.data.DataLoader(test, batch_size=p['b'], num_workers=0)
		#
		#return train, test
		
		
		x = np.load("./data/movingmnistdata.npz", encoding="bytes")
		

		dset = np.load("./data/movingmnistdata.npz", encoding="bytes")["arr_0"]
		
		if from_matlab:
			#return matlab.double(dset[:p['b']])
			return dset[:batch_size]
		imgs = torch.from_numpy(dset).float() / 255.
		imgs = imgs.view(-1,10, 1, 32, 32)
		
		
		data = TensorDataset(imgs, torch.zeros(imgs.shape[0]))

		if from_matlab:
			#return matlab.double(data.data[:p['b']])
			return data.data[:batch_size].numpy()
		
		
		return train_val_split(p, data)



	elif name == 'mnist_sequences':
		root = os.path.join(root, 'processed')
		data = datasets.MNIST(root=root, train=True, download=True,
		transform= transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]))
		
		images = data.data
		labels = data.targets
		new_ims = []
		
		# lossy method
		all_seqs = [[0,1,2,3], [3,4,5,6], [6,7,8,9]]

		# hacky way to ensure image-label alignment
		lim = min([min([images[labels==i].size(0) for i in s])] for s in all_seqs)[0]
		new_ims     = [torch.stack([images[labels == i][:lim,:,:] for i in s], dim=1) for s in all_seqs]	
		new_targets = [torch.stack([labels[labels == i][:lim] for i in s], dim=1) for s in all_seqs]
		new_ims = torch.cat(new_ims, dim=0) ; new_targets = torch.cat(new_targets, dim=0)
		
		# prevent batch underflow for divsors of 100
		def roundown(x): # could replace 100 with p['b']
			return int(math.floor(x / 100.0)) * 100
		data.data = np.expand_dims(new_ims[:roundown(new_ims.size(0))], axis=2)
		#data.data = new_ims[:roundown(new_ims.size(0))]
		data.targets = new_targets[:roundown(new_ims.size(0))].numpy()

		return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
	
	elif name == 'lsun_bedroom':
		data = datasets.LSUN(root=root, 
							 train=True, 
							classes=['bedroom_train'],
							transform=transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5),
													 (0.5, 0.5, 0.5)),
							]), download=True)
							 
	elif name == 'dsprites':

		url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
		
		if not os.path.exists('data/dsprites'):
			os.makedirs('data/dsprites')

		try:
			dset = np.load("./data/dsprites/dsprites.npz", encoding="bytes")["imgs"]
		except:
			request.urlretrieve(url, "./data/dsprites/dsprites.npz")
			dset = np.load("./data/dsprites/dsprites.npz", encoding="bytes")["imgs"]	
		
		if from_matlab:
			#return matlab.double(dset[:p['b']])
			return dset[:batch_size]
		imgs = torch.from_numpy(dset).float()
		data = TensorDataset(imgs)
		if from_matlab:
			#return matlab.double(data.data[:p['b']])
			return data.data[:batch_size].numpy()
		
		
		return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
	
	elif name == 'celeba':
		transform = transforms.Compose([transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		data = ImageFolder('data/celeba', transform)
	
		return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)

	elif name == 'bsds500':
		url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
		if not os.path.exists('data/bsds500'):
			os.makedirs('data/dsprites')
		

	else:
		print('No such dataset : '.format(name))



def rotate_mnist(t, data):
	""" fully rotate an mnist character in nt """
	bg_value = -0.5 
	new_imgs = []
	thresh = Variable(torch.Tensor([0.1])) # threshold
	image = np.reshape(data, (-1, 28))
	for i in range(t):
		angle = -(360 // 12) * i
		new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)
		new_img = torch.tensor(new_img).view((-1, 1, 28,28))
		new_img = (new_img < thresh).float() * 1
		new_img = new_img.add(-1).abs()
		new_imgs.append(new_img)
	return torch.stack(new_imgs, dim=1)
	

def data_check(p, data):
	
	if isinstance(data, list):
		data = data[0]
	data = Variable(data.cuda() if p['gpu'] else data)
	if p['rotating']:
		data = rotate_mnist(p['t'], data)
	#assert not max(data) > 1.
	return data
