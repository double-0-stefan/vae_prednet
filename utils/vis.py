import os
import numpy as np
import torch
import numpy as np
from scipy import stats
from utils import data_utils as dutils
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image


class Visualiser():
	def __init__(self, p, model):
		"""
		Visualizer is used to generate images of samples, reconstructions,
		latent traversals and so on of the trained model.
		Parameters
		----------
		"""
		self.model = model
		self.latent_traverser = LatentTraverser(p)
		self.plot_dir = os.path.join(p['plot_dir'], p['model_name'], 'traversals')
		if not os.path.exists(self.plot_dir):
			os.mkdir(self.plot_dir)
		

		self.save_images = True	 # If false, each method returns a tensor
								 # instead of saving image.

	def reconstructions(self, iteration, data, size=(8, 8), filename='recon.png'):
		"""
		Generates reconstructions of data through the model.
		Parameters
		----------
		data : torch.Tensor
			Data to be reconstructed. Shape (N, C, H, W)
		size : tuple of ints
			Size of grid on which reconstructions will be plotted. The number
			of rows should be even, so that upper half contains true data and
			bottom half contains reconstructions
		"""
		# Plot reconstructions in test mode, i.e. without sampling from latent
		filename = 'recon_{}.png'.format(iteration)

		self.model.eval()
		# Pass data through VAE to obtain reconstruction
		#input_data = Variable(data)

		if isinstance(data,list):
			data = data[0]
			_, recon_data = self.model(0, data, actions=data[1], eval=True)
		else:
			_, recon_data = self.model(0, data, eval=True)
		
		# [0] here is layer 0 
		#recon_data = recon_data[0]
		self.model.train()
		
		# Upper half of plot will contain data, bottom half will contain
		# reconstructions
		num_images = int(size[0] * size[1] // 2)
		#originals = input_data[:num_images,-1].cpu()
		originals = data[:num_images].cpu()
		reconstructions = recon_data.view(-1, *self.model.p['ldim'][0])[:num_images].cpu()
		# If there are fewer examples given than spaces available in grid,
		# augment with blank images
		num_examples = originals.size()[0]

		if self.model.p['pc_cnn']:
			originals = torch.squeeze(originals,1)
			reconstructions = torch.squeeze(reconstructions,1)

		if num_images > num_examples:
			blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
			originals = torch.cat([originals, blank_images])
			reconstructions = torch.cat([reconstructions, blank_images])
		# Concatenate images and reconstructions
		comparison = torch.cat([originals, reconstructions])
		
		if self.save_images:
			save_image(comparison.data, os.path.join(self.plot_dir, filename), nrow=size[0])
		else:
			return make_grid(comparison.data, nrow=size[0])

	def samples(self, e, size=(8, 8), filename='samples.png'):
		"""
		Generates samples from learned distribution by sampling prior and
		decoding.
		size : tuple of ints
		"""
		filename = 'samples_{}.png'.format(e)

		# Get prior samples from latent distribution
		cached_sample_prior = self.latent_traverser.sample_prior
		self.latent_traverser.sample_prior = True
		prior_samples = self.latent_traverser.traverse_grid(size=size)
		self.latent_traverser.sample_prior = cached_sample_prior

		# Map samples through decoder
		generated = self._decode_latents(prior_samples)

		if self.save_images:
			save_image(generated.data, self.plot_dir+'/' +filename, nrow=size[1])
		else:
			return make_grid(generated.data, nrow=size[1])

	def latent_traversal_line(self, e, cont_idx=None, disc_idx=None, size=8,
							  filename='traversal_line.png'):
		"""
		Generates an image traversal through a latent dimension.
		Parameters
		----------
		See viz.latent_traversals.LatentTraverser.traverse_line for parameter
		documentation.
		"""
		
		filename = 'traversal_line_{}.png'.format(e)

		# Generate latent traversal
		latent_samples = self.latent_traverser.traverse_line(cont_idx=cont_idx,
															 disc_idx=disc_idx,
															 size=size)

		# Map samples through decoder
		generated = self._decode_latents(latent_samples)

		if self.save_images:
			save_image(generated.data, self.plot_dir+'/' +filename, nrow=size)
		else:
			return make_grid(generated.data, nrow=size)

	def latent_traversal_grid(self, i, j, e, cont_idx=None, cont_axis=None,
							  disc_idx=None, disc_axis=None, size=(5, 5),
							  filename='traversal_grid.png'):
		"""
		Generates a grid of image traversals through two latent dimensions.
		Parameters
		----------
		See viz.latent_traversals.LatentTraverser.traverse_grid for parameter
		documentation.
		"""
		filename = 'traversal_grid_d{}_c{}_e{}.png'.format(i,j,e)

		# print(cont_idx)
		# print(cont_axis)
		# print(disc_idx)
		# print(disc_axis)
		# print(size)

		# Generate latent traversal
		latent_samples = self.latent_traverser.traverse_grid(cont_idx=cont_idx,
															 cont_axis=cont_axis,
															 disc_idx=disc_idx,
															 disc_axis=disc_axis,
															 size=size)

		# Map samples through decoder
		generated = self._decode_latents(latent_samples)

		if self.save_images:

			save_image(generated.data, self.plot_dir+'/' +filename, nrow=size[1])
		else:
			return make_grid(generated.data, nrow=size[1])

	def all_latent_traversals(self, e, size=8, filename='all_traversals.png'):
		"""
		Traverses all latent dimensions one by one and plots a grid of images
		where each row corresponds to a latent traversal of one latent
		dimension.
		Parameters
		----------
		size : int
			Number of samples for each latent traversal.
		"""
		
		filename = 'all_traversals_{}.png'.format(e)
		latent_samples = []
		# Perform line traversal of every continuous and discrete latent
		for cont_idx in range(self.model.p['nz_con'][0]):
			latent_samples.append(self.latent_traverser.traverse_line(cont_idx=cont_idx,
																	  disc_idx=None,
																	  size=size))

		# for disc_idx in range(len(self.model.p['nz_dis'][0])):
		# 	latent_samples.append(self.latent_traverser.traverse_line(cont_idx=None,
		# 															  disc_idx=disc_idx,
		# 															  size=size))
		# # Decode samples
		generated = self._decode_latents(torch.cat(latent_samples, dim=0))

		if self.save_images:
			save_image(generated.data, self.plot_dir+'/' +filename, nrow=size)
		else:
			return make_grid(generated.data, nrow=size)

	def _decode_latents(self, latent_samples):
		"""
		Decodes latent samples into images.
		Parameters
		----------
		latent_samples : torch.autograd.Variable
			Samples from latent distribution. Shape (N, L) where L is dimension
			of latent distribution.
		"""
		latent_samples = Variable(latent_samples)
		if self.model.p['gpu']:
			latent_samples = latent_samples.cuda()
		return self.model.decode(latent_samples, 0).cpu()


def reorder_img(orig_img, reorder, by_row=True, img_size=(3, 32, 32), padding=2):
	"""
	Reorders rows or columns of an image grid.
	Parameters
	----------
	orig_img : torch.Tensor
		Original image. Shape (channels, width, height)
	reorder : list of ints
		List corresponding to desired permutation of rows or columns
	by_row : bool
		If True reorders rows, otherwise reorders columns
	img_size : tuple of ints
		Image size following pytorch convention
	padding : int
		Number of pixels used to pad in torchvision.utils.make_grid
	"""
	reordered_img = torch.zeros(orig_img.size())
	_, height, width = img_size

	for new_idx, old_idx in enumerate(reorder):
		if by_row:
			start_pix_new = new_idx * (padding + height) + padding
			start_pix_old = old_idx * (padding + height) + padding
			reordered_img[:, start_pix_new:start_pix_new + height, :] = orig_img[:, start_pix_old:start_pix_old + height, :]
		else:
			start_pix_new = new_idx * (padding + width) + padding
			start_pix_old = old_idx * (padding + width) + padding
			reordered_img[:, :, start_pix_new:start_pix_new + width] = orig_img[:, :, start_pix_old:start_pix_old + width]

	return reordered_img

class LatentTraverser():
	def __init__(self, p):
		"""
		LatentTraverser is used to generate traversals of the latent space.
		Parameters
		----------
		latent_spec : dict
			See jointvae.models.VAE for parameter definition.
		"""

		self.sample_prior = False  # If False fixes samples in untraversed
								   # latent dimensions. If True samples
								   # untraversed latent dimensions from prior.
		self.is_continuous = p['nz_con'] is not None 
		# self.is_discrete = p['nz_dis'] is not None
		
		self.cont_dim = p['nz_con'][0] if self.is_continuous else None
		# self.disc_dims = p['nz_dis'][0] if self.is_discrete else None
		# if not isinstance(self.disc_dims, list):
		# 	self.disc_dims = [self.disc_dims]

	def traverse_line(self, cont_idx=None, disc_idx=None, size=5):
		"""
		Returns a (size, D) latent sample, corresponding to a traversal of the
		latent variable indicated by cont_idx or disc_idx.
		Parameters
		----------
		cont_idx : int or None
			Index of continuous dimension to traverse. If the continuous latent
			vector is 10 dimensional and cont_idx = 7, then the 7th dimension
			will be traversed while all others will either be fixed or randomly
			sampled. If None, no latent is traversed and all latent
			dimensions are randomly sampled or kept fixed.
		disc_idx : int or None
			Index of discrete latent dimension to traverse. If there are 5
			discrete latent variables and disc_idx = 3, then only the 3rd
			discrete latent will be traversed while others will be fixed or
			randomly sampled. If None, no latent is traversed and all latent
			dimensions are randomly sampled or kept fixed.
		size : int
			Number of samples to generate.
		"""
		samples = []

		if self.is_continuous:
			samples.append(self._traverse_continuous_line(idx=cont_idx,
														  size=size))
		# if self.is_discrete:
		# 	for i, disc_dim in enumerate(self.disc_dims):
		# 		if i == disc_idx:
		# 			samples.append(self._traverse_discrete_line(dim=disc_dim,
		# 														traverse=True,
		# 														size=size))
		# 		else:
		# 			samples.append(self._traverse_discrete_line(dim=disc_dim,
		# 														traverse=False,
		# 														size=size))

		return torch.cat(samples, dim=1)

	def _traverse_continuous_line(self, idx, size):
		"""
		Returns a (size, cont_dim) latent sample, corresponding to a traversal
		of a continuous latent variable indicated by idx.
		Parameters
		----------
		idx : int or None
			Index of continuous latent dimension to traverse. If None, no
			latent is traversed and all latent dimensions are randomly sampled
			or kept fixed.
		size : int
			Number of samples to generate.
		"""
		if self.sample_prior:
			samples = np.random.normal(size=(size, self.cont_dim))
		else:
			samples = np.zeros(shape=(size, self.cont_dim))

		if idx is not None:
			# Sweep over linearly spaced coordinates transformed through the
			# inverse CDF (ppf) of a gaussian since the prior of the latent
			# space is gaussian
			cdf_traversal = np.linspace(0.05, 0.95, size)
			cont_traversal = stats.norm.ppf(cdf_traversal)

			for i in range(size):
				samples[i, idx] = cont_traversal[i]

		return torch.Tensor(samples)

	def _traverse_discrete_line(self, dim, traverse, size):
		"""
		Returns a (size, dim) latent sample, corresponding to a traversal of a
		discrete latent variable.
		Parameters
		----------
		dim : int
			Number of categories of discrete latent variable.
		traverse : bool
			If True, traverse the categorical variable otherwise keep it fixed
			or randomly sample.
		size : int
			Number of samples to generate.
		"""
		samples = np.zeros((size, dim))

		if traverse:
			for i in range(size):
				samples[i, i % dim] = 1.
		else:
			# Randomly select discrete variable (i.e. sample from uniform prior)
			if self.sample_prior:
				samples[np.arange(size), np.random.randint(0, dim, size)] = 1.
			else:
				samples[:, 0] = 1.

		return torch.Tensor(samples)

	def traverse_grid(self, cont_idx=None, cont_axis=None, disc_idx=None,
					  disc_axis=None, size=(5, 5)):
		"""
		Returns a (size[0] * size[1], D) latent sample, corresponding to a
		two dimensional traversal of the latent space.
		Parameters
		----------
		cont_idx : int or None
			Index of continuous dimension to traverse. If the continuous latent
			vector is 10 dimensional and cont_idx = 7, then the 7th dimension
			will be traversed while all others will either be fixed or randomly
			sampled. If None, no latent is traversed and all latent
			dimensions are randomly sampled or kept fixed.
		cont_axis : int or None
			Either 0 for traversal across the rows or 1 for traversal across
			the columns. If None and disc_axis not None will default to axis
			which disc_axis is not. Otherwise will default to 0.
		disc_idx : int or None
			Index of discrete latent dimension to traverse. If there are 5
			discrete latent variables and disc_idx = 3, then only the 3rd
			discrete latent will be traversed while others will be fixed or
			randomly sampled. If None, no latent is traversed and all latent
			dimensions are randomly sampled or kept fixed.
		disc_axis : int or None
			Either 0 for traversal across the rows or 1 for traversal across
			the columns. If None and cont_axis not None will default to axis
			which cont_axis is not. Otherwise will default to 1.
		size : tuple of ints
			Shape of grid to generate. E.g. (6, 4).
		"""
		if cont_axis is None and disc_axis is None:
			cont_axis = 0
			disc_axis = 0
		elif cont_axis is None:
			cont_axis = int(not disc_axis)
		elif disc_axis is None:
			disc_axis = int(not cont_axis)

		samples = []

		if self.is_continuous:
			samples.append(self._traverse_continuous_grid(idx=cont_idx,
														  axis=cont_axis,
														  size=size))
		# if self.is_discrete 
		# 	for i, disc_dim in enumerate(self.disc_dims):
		# 		if i == disc_idx:
		# 			samples.append(self._traverse_discrete_grid(dim=disc_dim,
		# 														axis=disc_axis,
		# 														traverse=True,
		# 														size=size))
		# 		else:
		# 			samples.append(self._traverse_discrete_grid(dim=disc_dim,
		# 														axis=disc_axis,
		# 														traverse=False,
		# 														size=size))

		return torch.cat(samples, dim=1)

	def _traverse_continuous_grid(self, idx, axis, size):
		"""
		Returns a (size[0] * size[1], cont_dim) latent sample, corresponding to
		a two dimensional traversal of the continuous latent space.
		Parameters
		----------
		idx : int or None
			Index of continuous latent dimension to traverse. If None, no
			latent is traversed and all latent dimensions are randomly sampled
			or kept fixed.
		axis : int
			Either 0 for traversal across the rows or 1 for traversal across
			the columns.
		size : tuple of ints
			Shape of grid to generate. E.g. (6, 4).
		"""
		num_samples = size[0] * size[1]

		if self.sample_prior:
			samples = np.random.normal(size=(num_samples, self.cont_dim))
		else:
			samples = np.zeros(shape=(num_samples, self.cont_dim))

		if idx is not None:
			# Sweep over linearly spaced coordinates transformed through the
			# inverse CDF (ppf) of a gaussian since the prior of the latent
			# space is gaussian
			cdf_traversal = np.linspace(0.05, 0.95, size[axis])
			cont_traversal = stats.norm.ppf(cdf_traversal)

			for i in range(size[0]):
				for j in range(size[1]):
					if axis == 0:
						samples[i * size[1] + j, idx] = cont_traversal[i]
					else:
						samples[i * size[1] + j, idx] = cont_traversal[j]

		return torch.Tensor(samples)

	def _traverse_discrete_grid(self, dim, axis, traverse, size):
		"""
		Returns a (size[0] * size[1], dim) latent sample, corresponding to a
		two dimensional traversal of a discrete latent variable, where the
		dimension of the traversal is determined by axis.
		Parameters
		----------
		idx : int or None
			Index of continuous latent dimension to traverse. If None, no
			latent is traversed and all latent dimensions are randomly sampled
			or kept fixed.
		axis : int
			Either 0 for traversal across the rows or 1 for traversal across
			the columns.
		traverse : bool
			If True, traverse the categorical variable otherwise keep it fixed
			or randomly sample.
		size : tuple of ints
			Shape of grid to generate. E.g. (6, 4).
		"""
		num_samples = size[0] * size[1]
		samples = np.zeros((num_samples, dim))

		if traverse:
			disc_traversal = [i % dim for i in range(size[axis])]
			for i in range(size[0]):
				for j in range(size[1]):
					if axis == 0:
						samples[i * size[1] + j, disc_traversal[i]] = 1.
					else:
						samples[i * size[1] + j, disc_traversal[j]] = 1.
		else:
			# Randomly select discrete variable (i.e. sample from uniform prior)
			if self.sample_prior:
				samples[np.arange(num_samples), np.random.randint(0, dim, num_samples)] = 1.
			else:
				samples[:, 0] = 1.

		return torch.Tensor(samples)