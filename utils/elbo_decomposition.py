import os
import math
from utils import dist
import torch
from torch import cuda
from numbers import Number
from torch.autograd import Variable

def estimate_entropies(qz_samples, qz_params, q_dist):
	"""Computes the term:
		E_{p(x)} E_{q(z|x)} [-log q(z)]
	and
		E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
	where q(z) = 1/N sum_n=1^N q(z|x_n).
	Assumes samples are from q(z|x) for *all* x in the dataset.
	Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

	Computes numerically stable NLL:
		- log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

	Inputs:
	-------
		qz_samples (K, S) Variable
		qz_params  (N, K, nparams) Variable
	"""

	# Only take a sample subset of the samples
	#qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:10000]).cuda())
	qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:100]))


	K, S = qz_samples.size()
	N, _, nparams = qz_params.size()

	assert(nparams == q_dist.nparams)
	assert(K == qz_params.size(1))

	#marginal_entropies = torch.zeros(K).cuda()
	marginal_entropies = torch.zeros(K)
	#joint_entropy = torch.zeros(1).cuda()
	joint_entropy = torch.zeros(1)

	pbar = tqdm(total=S)
	k = 0
	while k < S:
		batch_size = min(10, S - k)
		logqz_i = q_dist.log_density(
			qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
			qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])

		k += batch_size

		# computes - log q(z_i) summed over minibatch
		marginal_entropies += (math.log(N) - logsumexp(logqz_i, dim=0, keepdim=False).data).sum(1)

		# computes - log q(z) summed over minibatch
		logqz = logqz_i.sum(1)	# (N, S)
		
		joint_entropy += (math.log(N) - logsumexp(logqz, dim=0, keepdim=False).data).sum(0)

		
		pbar.update(batch_size)
	pbar.close()

	marginal_entropies /= S
	joint_entropy /= S

	return marginal_entropies, joint_entropy


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


def analytical_NLL(qz_params, q_dist, prior_dist, qz_samples=None):
	"""Computes the quantities
		1/N sum_n=1^N E_{q(z|x)} [ - log q(z|x) ]
	and
		1/N sum_n=1^N E_{q(z_j|x)} [ - log p(z_j) ]
	Inputs:
	-------
		qz_params  (N, K, nparams) Variable
	Returns:
	--------
		nlogqz_condx (K,) Variable
		nlogpz (K,) Variable
	"""
	pz_params = Variable(torch.zeros(1).type_as(qz_params.data).expand(qz_params.size()), volatile=True)

	nlogqz_condx = q_dist.NLL(qz_params).mean(0)
	nlogpz = prior_dist.NLL(pz_params, qz_params).mean(0)
	return nlogqz_condx, nlogpz


def elbo_decomposition(vae, dataset_loader, layers, imdims):
	
	N = len(dataset_loader.dataset)	 # number of data samples
	logpxs=[]; dependences=[]; informations=[]; dimwise_kls=[]
	analytical_cond_kls=[]; marginal_entropiess=[]; joint_entropys=[]
	
	for l in range(layers): # this loop can be replaced 
		imdim = imdims[l]
		K = sum(vae.p['z_dim'][l:l+2])					# number of latent variables
		S = 1							 # number of latent variable samples
		nparams = vae.q_dist[l].nparams

		print('Computing q(z|x) distributions.')

		# compute the marginal q(z_j|x_n) distributions
		qz_params = torch.Tensor(N, K, nparams)
		n = 0
		logpx = 0
		for xs in dataset_loader:
			xs = xs[0]
			#xs = Variable(xs.cuda())
			xs = Variable(xs)

			batch_size = xs.size(0)
			vae.p['b'] = batch_size
			xs = xs.type(torch.FloatTensor)
			if vae.p['rotating']:
				xs = utils.rotate_mnist(p['t'], xs)
			else:
				if len(xs.shape) == 3:
					xs = xs.unsqueeze(1)
				xs = xs.unsqueeze(2 if vae.p['sequences'] else 1) # see calc args
				xs = xs.expand(vae.p['b'], vae.p['t'],*vae.p['imdim'])									
				
			#if len(xs.shape) > 4:
			#	xs = xs[:,0]
			#xs = [Variable(xs.view(batch_size, -1, *imdim).cuda()), Variable(z if l < layers-1 else None)]			
			#xs = Variable(xs.view(batch_size, -1, *imdim).cuda())
			
			vae.reset()
			#xs = xs.cuda()
			#print(xs.shape)
			#print(xs.is_cuda)
			vae.p['gpu'] = False
			z_params, x_params, _ = vae(xs, None, True)
			z = None
			z_params = z_params[l].view(batch_size, K, nparams)
			qz_params[n:n + batch_size] = z_params.data
			n += batch_size			
			logpx += vae.x_dist.log_density(xs[:,1], params=x_params[l]).view(batch_size, -1).data.sum()
		
		vae.del_vars()
		cuda.empty_cache()
		# Reconstruction term
		logpx = logpx / (N * S)
		with torch.no_grad():
			qz_params = Variable(qz_params)
		# sample S times from each marginal q(z_j|x_n)
		qz_params_expanded = qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)
		qz_samples = vae.q_dist[l].sample(params=qz_params_expanded)
		qz_samples = qz_samples.transpose(0, 1).contiguous().view(K, N * S)
		
		print('Estimating entropies.')
		marginal_entropies, joint_entropy = estimate_entropies(qz_samples, qz_params, vae.q_dist[l])

		if hasattr(vae.q_dist[l], 'NLL'):
			nlogqz_condx = vae.q_dist[l].NLL(qz_params).mean(0)
		else:
			nlogqz_condx = - vae.q_dist[l].log_density(qz_samples,
				qz_params_expanded.transpose(0, 1).contiguous().view(K, N * S)).mean(1)

		if hasattr(vae.prior_dist[l], 'NLL'):
			pz_params = vae._get_prior_params(N * K, l).contiguous().view(N, K, -1)
			nlogpz = vae.prior_dist[l].NLL(pz_params, qz_params).mean(0)
		else:
			nlogpz = - vae.prior_dist[l].log_density(qz_samples.transpose(0, 1)).mean(0)

		# nlogqz_condx, nlogpz = analytical_NLL(qz_params, vae.q_dist, vae.prior_dist)
		nlogqz_condx = nlogqz_condx.data
		nlogpz = nlogpz.data

		# Independence term
		# KL(q(z)||prod_j q(z_j)) = log q(z) - sum_j log q(z_j)
		dependence = (- joint_entropy + marginal_entropies.sum())[0]

		# Information term
		# KL(q(z|x)||q(z)) = log q(z|x) - log q(z)
		information = (- nlogqz_condx.sum() + joint_entropy)[0]

		# Dimension-wise KL term
		# sum_j KL(q(z_j)||p(z_j)) = sum_j (log q(z_j) - log p(z_j))
		dimwise_kl = (- marginal_entropies + nlogpz).sum()

		# Compute sum of terms analytically
		# KL(q(z|x)||p(z)) = log q(z|x) - log p(z)
		analytical_cond_kl = (- nlogqz_condx + nlogpz).sum()

		print('Dependence: {}'.format(dependence))
		print('Information: {}'.format(information))
		print('Dimension-wise KL: {}'.format(dimwise_kl))
		print('Analytical E_p(x)[ KL(q(z|x)||p(z)) ]: {}'.format(analytical_cond_kl))
		print('Estimated  ELBO: {}'.format(logpx - analytical_cond_kl))

		logpxs.append(logpx.item());dependences.append(dependence.item());informations.append(information.item()) 
		dimwise_kls.append(dimwise_kl.item());analytical_cond_kls.append(analytical_cond_kl.item())
		marginal_entropiess.append([x.item() for x in marginal_entropies]); joint_entropys.append(joint_entropy.item())
	return logpxs, dependences, informations, dimwise_kls, analytical_cond_kls, marginal_entropiess, joint_entropys
