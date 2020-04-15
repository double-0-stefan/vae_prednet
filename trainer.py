import numpy as np
from os.path import join 
from torch.optim import Adam
from scipy.io import savemat
from torch import cuda, no_grad, save, load, cat, zeros
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from logging import getLogger
from utils import train_utils as tutils
from utils import model_utils as mutils
from utils import data_utils as dutils
from model import ObservationModel, ObservationVAE, TransitionModel
import matplotlib.pyplot as plt
from torchvision.utils import save_image
#from animalai.envs import UnityEnvironment
#from animalai.envs.arena_config import ArenaConfig
import glob

from torch import zeros, FloatTensor, tensor, argmax, zeros_like
from torch.autograd import Variable
import torch

# import torch_xla
# import torch_xla.core.xla_model as xm
# import torch_xla.debug.metrics as met
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.distributed.xla_multiprocessing as xmp
# import torch_xla.utils.utils as xu

class InteractiveTrainer(object):	

	def __init__(self, p, model):
		
		self.p = p
		self.model = model

		logger = getLogger('train')	

		env_name = 'env/AnimalAI'
		self.env = UnityEnvironment( n_arenas=p['b'],
								file_name=env_name)

		default_brain = self.env.brain_names[0]
		self.brain = self.env.brains[default_brain]
				
		self.iteration = 0 
		self.plot_iter = 0
		self.logger = getLogger('train')
		self.methandle = tutils.MetricsHandler() 
		
		self.optimizer = Adam(self.model.parameters(), lr=p['lr'])

		# 0 != a layer index. listing precludes a_net params from prednet graph
		self.act_opt   = Adam(self.model.a_net[0].parameters(), lr=p['lr'])
		
		
	def train(self):
		
		self.model.train() 

		self.model.p = tutils.set_paths(self.p, 'interactive_model')

		self.logger.info('\n Training Observation Model \n ')
		self.logger.info('Model Overview: \n {} \n'.format(self.model.parameters))
		trainp  = sum(_p.numel() for _p in self.model.parameters() if _p.requires_grad)
		ntrainp = sum(_p.numel() for _p in self.model.parameters() if not _p.requires_grad)
		self.logger.info('Trainable Params {} \n'.format(tutils.group(trainp)))
		self.logger.info('Non-Trainable Params {} \n'.format(tutils.group(ntrainp)))

		#while self.iteration < self.model.p['n_iter']:
		for e in range(self.p['e']):
		#while self.iteration < 1:
			self.logger.info(' Training Epoch {} of {} '.format(e+1,self.p['e']))
			self.model.train()
			self.train_epoch()
			with no_grad():
				self.model.eval()
				self.eval_batch()
		#self.eval_batch(force_write=True)
		self.plot_loss()
		
	
	def train_epoch(self):
		""" All items from dataloader are passed to forward.
			Handle input data in the model's forward function, 
			or overwrite this function in a subordinate class. """
		epoch_loss 	    = 0
		epoch_reward    = 0
		epoch_anet_loss = 0

		all_files = glob.glob("examples/configs/env_configs/*.yaml") # list of all .yaml files in a directory 
		files = []
		for file in all_files:
			if any(x in file for x in ['1', '2', '4']):
				files.append(file)
				
		for file in files:

			config = ArenaConfig(file)
			
			obs = self.env.reset(arenas_configurations=config, train_mode=True)['Learner']
			action = {}
				
			init_action = zeros(self.p['n_actions']*self.p['b'],1)
			init_action = init_action.cuda() if self.p['gpu'] else init_action 
			action['Learner'] = init_action
			done = False
			
			self.model.reset()
			errors = zeros_like(FloatTensor(obs.visual_observations[0])).cuda()
			self.model.a_net[0].policy_history = torch.zeros(1000, self.p['b'], 2)
			
			i = 0
			while  i < 1000:

				# run batch
				#print(action['Learner'].shape)
				info 	 =  self.env.step(vector_action=action)['Learner']
				
				vis_obs  = FloatTensor(info.visual_observations[0]).cuda().permute(0,-1,1,2)
				#plt.figure()
				#print(vis_obs.shape)
				#plt.imshow(vis_obs[0].permute(1,2,0).cpu()) 
				#plt.show() 
				vel_obs  = tensor(info.vector_observations).cuda()
				text_obs = info.text_observations
				reward 	 = info.rewards		
				epoch_reward += np.mean(reward)
				self.model.a_net[0].reward_episode.append(reward)
				
				self.model.zero_grad()
				actions, errors, z, z_pc, z_pd = self.model(self.iteration, vis_obs, errors, tensor(action['Learner']).cuda())
				acts = argmax(actions.detach(), dim=-1)
				
				self.model.a_net[0].policy_history[i,:] = acts
				
				action['Learner'] = acts.view(self.p['n_actions']*self.p['b'],1)
				#print(action['Learner'].shape)
				# backward
				
				self.vae_opt.zero_grad()			
				vae_loss, metrics = self.model.loss(self.iteration, errors, z_pc, z_pd)
				#loss.backward(retain_graph=True)
				vae_loss.backward()
				self.optimizer.step()		
				vae_epoch_loss += vae_loss.item()
				
				self.rnn_opt.zero_grad()			
				vae_loss = self.model.lstm[0].loss(z)
				#loss.backward(retain_graph=True)
				vae_loss.backward()
				self.optimizer.step()		
				vae_epoch_loss += vae_loss.item()
				
				
				
				self.iteration += 1
				i += 1
				self.model.reset()
				errors = errors.detach()

			self.act_opt.zero_grad()
			a_loss = self.model.a_net[0].loss()
			a_loss.backward()
			self.act_opt.step()	
			epoch_anet_loss = a_loss.item()
			self.model.a_net[0].loss_history.append(loss.item())
			self.model.a_net[0].policy_history = Variable(torch.Tensor())
			self.model.a_net[0].reward_episode = []				

		epoch_loss      = epoch_loss / 1000 / 4
		epoch_anet_loss = epoch_anet_loss / 1000 / 4

		self.logger.info('Mean Epoch Loss  {}'.format(epoch_loss))
		self.logger.info('Mean Epoch Action Loss  {}'.format(epoch_anet_loss))
		self.logger.info('Mean Epoch Reward  {}'.format(epoch_reward))
				
	def eval_batch(self, force_write=None):
		
		self.model.reset()
		tutils.save_checkpoint({'model': self.model, 
								'state_dict': self.model.state_dict(),
								'args': self.model.p}, 
								 self.model.p['model_dir'],  
								 self.model.p['model_name'], 0)	
		
		#self.model.err_plot_flag = True
		#for image_data, act_data in self.test_loader:
		#	plot_vars = self.model(self.iteration, image_data.cuda(), act_data.cuda(), eval=True)
		#	break 
        #
		#if (self.iteration % self.p['plot_iter'] == 0) or force_write:
		#	self.model.plot(self.iteration, image_data, plot_vars)
		#	self.vis(self.iteration)
		#	#save_image(self.model.plot_errs, 'Fig_pe_signals_over_time_{}.png'.format(self.iteration))
		#	self.plot_iter += 1

								
		#log_py_pth  = join(self.p['metrics_dir'], '{}.pth'.format(self.model.p['model_name']))
		#log_mat_pth = join(self.p['metrics_dir'], '{}.mat'.format(self.model.p['model_name']))
		#save(metrics, log_py_pth)
		#savemat(log_mat_pth, metrics)
		
		
		#self.methandle.extend(metrics)
		
		#if self.model.p['calc_disentanglement']:
		#	logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, \
		#	joint_entropy = elbo_decomposition(model, test_loader, p['layers'], p['ldim'])
		#		
		#	metrics = {'logpx': logpx,	'dependence': dependence,'information': information,
		#			   'dimwise_kl': dimwise_kl, 'analytical_cond_kl': analytical_cond_kl,
		#			   'marginal_entropies': marginal_entropies,'joint_entropy': joint_entropy}
		#		
		#	log_py_pth  = join(self.p['metrics_dir'], 'disentanglement', self.p['name']+'.pth')
		#	log_mat_pth = join(self.p['metrics_dir'], 'disentanglement', self.p['name']+'.mat')
		#	save(metrics, log_py_pth)
		#	savemat(log_mat_pth, metrics)
		#
			
	def vis(self, e):
		mutils.visualise(self.model.p, self.model, e, self.test_loader)

	def plot_loss(self):
		
		err_loc = join(self.p['metrics_dir'], 'err_loss.png')
		plt.plot(self.methandle.metrics['loss'])
		plt.savefig(err_loc)   # save the figure to file
		plt.close()

		cat_loc = join(self.p['metrics_dir'], 'cat_loss.png')
		plt.plot(self.methandle.metrics['disc_kl'])
		plt.savefig(cat_loc)   # save the figure to file
		plt.close()

		kl_loc = join(self.p['metrics_dir'], 'cont_loss.png')
		plt.plot(self.methandle.metrics['cont_kl'])
		plt.savefig(kl_loc)   # save the figure to file
		plt.close()

class Trainer(object):	

	def __init__(self, p, dataloader):
		
		self.p = p
		self.train_loader = dataloader[0]
		self.test_loader   = dataloader[1]		
		
		self.iteration = 0 
		self.plot_iter = 0
		self.logger = getLogger('train')
		self.methandle = tutils.MetricsHandler() 
	
	def train_epoch(self):
		""" All items from dataloader are passed to forward.
			Handle input data in the model's forward function, 
			or overwrite this function in a subordinate class. """
		epoch_loss = 0

		for data in self.train_loader:
	
			data = [x.cuda() for x in data]	

			# prepare model for training
			self.model.reset()

			# forward 
			self.model(self.iteration, *data)
			
			# backward
			self.optimizer.zero_grad()			
			self.model.iter_loss.backward(retain_graph=True)
			self.optimizer.step()
			
			epoch_loss += self.model.iter_loss.item()
			
			self.iteration += 1

		epoch_loss = epoch_loss / len(self.train_loader.dataset)
		self.logger.info('Mean Epoch Loss - {}'.format(epoch_loss))

	def train_epoch_pc_cnn(self):
		""" All items from dataloader are passed to forward.
			Handle input data in the model's forward function, 
			or overwrite this function in a subordinate class. """
		epoch_loss = 0

		for data in self.train_loader:
	
			if self.p['xla']:
				data = [x.to(xm.xla_device()) for x in data]
			else:
				data = [x.cuda() for x in data]	

			# prepare model for training
			self.model.reset()

			# forward 
			self.model(self.iteration, *data)
			
			# backward
			# self.optimizer.zero_grad()			
			# self.model.iter_loss.backward(retain_graph=True)
			# self.optimizer.step()
			
			# epoch_loss += self.model.iter_loss.item()
			epoch_loss += self.model.F
			self.iteration += 1

		#epoch_loss = epoch_loss / len(self.train_loader.dataset)
		self.logger.info('Mean Epoch Loss - {}'.format(epoch_loss))
		self.epoch_loss = epoch_loss
				
	def eval_batch(self, e, force_write=None):
		
		self.model.reset()
		
		self.model.err_plot_flag = True
		for image_data, act_data in self.test_loader:
			plot_vars = self.model(self.iteration, image_data.cuda(), act_data.cuda(), eval=True)
			break 

		if (e % self.p['plot_iter'] == 0) or force_write:

			self.model.plot(self.iteration, image_data, plot_vars)
			self.vis(self.iteration)
			#save_image(self.model.plot_errs, 'Fig_pe_signals_over_time_{}.png'.format(self.iteration))
			self.plot_iter += 1


		#_, metrics = self.model.loss(self.iteration)		
		
		#disc_min, disc_max, disc_num_iters, disc_gamma = self.model.p['z_dis_capacity'][0]
		#disc_cap_current = (disc_max - disc_min) * self.iteration / float(disc_num_iters) + disc_min
		#disc_cap_current = min(disc_cap_current, disc_max)
		#disc_theoretical_max = sum([float(np.log(dim)) for dim in self.model.p['nz_dis'][0]])
		#disc_cap_current = min(disc_cap_current, disc_theoretical_max)
		#		
		#self.logger.info(' \n Current Iteration : {}  | Current Discrete Capacity {} \n'.format(self.iteration, disc_cap_current))
		#self.logger.info(' \n Err : {} | Norm KL : {} | Cat KL : {} \n'.format(metrics[0], metrics[1], metrics[2]))

		#metrics = {'loss':metrics[0],'cont_kl':metrics[1],'disc_kl':metrics[2]}
		
		#print(model.state_dict())
		tutils.save_checkpoint({'model': self.model, 
								'state_dict': self.model.state_dict(),
								'args': self.model.p}, 
								 self.model.p['model_dir'],  
								 self.model.p['model_name'], 0)	
								
		#log_py_pth  = join(self.p['metrics_dir'], '{}.pth'.format(self.model.p['model_name']))
		#log_mat_pth = join(self.p['metrics_dir'], '{}.mat'.format(self.model.p['model_name']))
		#save(metrics, log_py_pth)
		#savemat(log_mat_pth, metrics)
		
		
		#self.methandle.extend(metrics)
		
		#if self.model.p['calc_disentanglement']:
		#	logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, \
		#	joint_entropy = elbo_decomposition(model, test_loader, p['layers'], p['ldim'])
		#		
		#	metrics = {'logpx': logpx,	'dependence': dependence,'information': information,
		#			   'dimwise_kl': dimwise_kl, 'analytical_cond_kl': analytical_cond_kl,
		#			   'marginal_entropies': marginal_entropies,'joint_entropy': joint_entropy}
		#		
		#	log_py_pth  = join(self.p['metrics_dir'], 'disentanglement', self.p['name']+'.pth')
		#	log_mat_pth = join(self.p['metrics_dir'], 'disentanglement', self.p['name']+'.mat')
		#	save(metrics, log_py_pth)
		#	savemat(log_mat_pth, metrics)
		#
			
	def vis(self, e):
		mutils.visualise(self.model.p, self.model, e, self.test_loader)

	def plot_loss(self):
		
		err_loc = join(self.p['metrics_dir'], 'err_loss.png')
		plt.plot(self.methandle.metrics['loss'])
		plt.savefig(err_loc)   # save the figure to file
		plt.close()

		cat_loc = join(self.p['metrics_dir'], 'cat_loss.png')
		plt.plot(self.methandle.metrics['disc_kl'])
		plt.savefig(cat_loc)   # save the figure to file
		plt.close()

		kl_loc = join(self.p['metrics_dir'], 'cont_loss.png')
		plt.plot(self.methandle.metrics['cont_kl'])
		plt.savefig(kl_loc)   # save the figure to file
		plt.close()
	
class TransitionTrainer(Trainer):	
	def __init__(self, p, dataloader):
		super(TransitionTrainer, self).__init__(p, dataloader)
		
		self.obs_model_path = join(p['model_dir'], p['model_name'])
		self.obs_model  = None	
	
	def prep_latent_dataloaders(self, obs_model):
		with no_grad():
			latent_data  = []
			action_data  = []
			_actshape = (self.p['b'], self.p['n_steps'], self.p['n_actions'], self.p['action_dim'])
						
			for b, data in enumerate(self.train_loader):
				
				z   = zeros(self.p['b'], self.p['n_steps'], self.p['z_dim'][0])
				act = zeros(*_actshape) 
				
				data[0] = data[0].cuda()
				data[1] = data[1].cuda()
				for t in range(self.p['n_steps']):

					z[:,t] = obs_model(data[0][:,t].squeeze(1))
					act[:,t] = data[1][:,t]

				latent_data.append(z)
				action_data.append(act)		
		
			# roll images into future
			z_tensor   = cat(latent_data,  dim=0).cuda()
			z_tensor   = dutils.roll(z_tensor, 1, -1, fill_pad=0).unsqueeze(-2)
			act_tensor = cat(action_data, dim=0)

			dataset = TensorDataset(*[z_tensor, act_tensor])
		
		return dutils.train_val_split(self.p,dataset)
			
	def _load_obs_model(self):
		with no_grad():
			obs_model = ObservationModel(self.p).cuda()
			obs_model.load_state_dict(load(self.obs_model_path+'.pth')['state_dict'])
			obs_model.eval()
		return obs_model
			
	def _prep_transition_model(self):
		
		# load vision model 
		obs_model  = self._load_obs_model()
		obs_params = obs_model.named_parameters()
		obs_model.enc_mode = True
		_train, _test = self.prep_latent_dataloaders(obs_model)
		self.train_loader = _train
		self.test_loader  = _test
		# initialise transition model 
		self.p['use_lstm'] = True ; 
		rnn = TransitionModel(self.p)
		rnn_params = dict(rnn.state_dict())
		
		# copy and freeze modules from vision model
		for name_v, param_v in obs_params:
			if name_v in rnn_params:
				rnn_params[name_v].data.copy_(param_v.data)
				rnn_params[name_v].requires_grad = False
		
		rnn.load_state_dict(rnn_params)
		
		self.model = rnn

		self.model.p['datasize']  = len(self.train_loader.dataset)
		self.model.p['n_batches'] = len(self.train_loader)
		self.model.p['n_iter']    = self.p['datasize'] * self.p['e']

		cuda.empty_cache()			

	def train(self):

		self._prep_transition_model()
		self.model.train() 
		self.model.p = tutils.set_paths(self.p, 'trans_model')
		
		self.optimizer = Adam(self.model.parameters(), lr=self.model.p['lr'], weight_decay=1e-5)

		self.logger.info('\n Training Transition Model \n ')
		self.logger.info('Model Overview: \n {} \n'.format(self.model.parameters))
		trainp  = sum(_p.numel() for _p in self.model.parameters() if _p.requires_grad)
		ntrainp = sum(_p.numel() for _p in self.model.parameters() if not _p.requires_grad)
		self.logger.info('Trainable Params {} \n'.format(tutils.group(trainp)))
		self.logger.info('Non-Trainable Params {} \n'.format(tutils.group(ntrainp)))
				
		
		#while self.iteration < self.model.p['n_iter']:
		#while self.iteration < 1:
		for e in range(self.p['e']):
			self.logger.info(' Training Epoch {} of {} '.format(e+1,self.p['e']))
			self.train_epoch()
			with no_grad():
				self.eval_batch()
		self.plot_loss()
				
class SaccadeTrainer(Trainer):
	def __init__(self, p, dataloader):
		super(SaccadeTrainer, self).__init__(p, dataloader)
		
		self.obs_model_path = join(p['model_dir'], p['model_name'])
		self.obs_model  = None	
	
	def prep_latent_dataloaders(self, obs_model):
		with no_grad():
			latent_data  = []
			action_data  = []
			_actshape = (-1, self.p['n_steps'], 1, 2)
			indices = list(range(self.p['n_actions']))

			for b, data in enumerate(self.train_loader):

				z   = zeros(self.p['b'], self.p['n_steps'], self.p['z_dim'][0]).cuda()
				act = zeros(self.p['b'], self.p['n_steps'], self.p['n_actions'], self.p['action_dim']).cuda()

				data[0] = data[0].cuda()

				for t in range(self.p['n_steps']):
					
					lr = np.random.choice(self.p['action_dim'])
					ud = np.random.choice(self.p['action_dim'])
					act[:,t, 0, lr] = 1
					act[:,t, 1, ud] = 1
					z[:,t] = obs_model(data[0][:,t], actions=act[:,t])

				latent_data.append(z)
				action_data.append(act)				
				cuda.empty_cache()	

			# roll images into future
			z_tensor   = cat(latent_data,  dim=0).cuda()
			#z_tensor   = dutils.roll(z_tensor, 1, -1, fill_pad=0).unsqueeze(-2)
			act_tensor = cat(action_data, dim=0)

			dataset = TensorDataset(*[z_tensor, act_tensor])

		return dutils.train_val_split(self.p, dataset)
		
	def _load_obs_model(self):
		with no_grad():
			obs_model = ObservationModel(self.p).cuda()
			obs_model.load_state_dict(load(self.obs_model_path+'.pth')['state_dict'])
			obs_model.eval()
		return obs_model
			
	def _prep_saccade_model(self):
		
		# load vision model 
		self.p['foveate'] = True
		obs_model  = self._load_obs_model()
		obs_params = obs_model.named_parameters()
		obs_model.enc_mode = True
		self.train_loader, self.test_loader = self.prep_latent_dataloaders(obs_model)
		del obs_model 
		cuda.empty_cache()			
		
		# initialise transition model 
		self.p['use_lstm'] = True ; 
		rnn = TransitionModel(self.p)
		rnn_params = dict(rnn.state_dict())
		
		# copy and freeze modules from vision model
		for name_v, param_v in obs_params:
			if name_v in rnn_params:
				rnn_params[name_v].data.copy_(param_v.data)
				rnn_params[name_v].requires_grad = False
		
		rnn.load_state_dict(rnn_params)
		
		self.model = rnn
				

		self.model.p['datasize']  = len(self.train_loader.dataset)
		self.model.p['n_batches'] = len(self.train_loader)
		self.model.p['n_iter']    = self.p['datasize'] * self.p['e']

	def train(self):
		
		self.p['foveate'] = True
		self._prep_saccade_model()
		self.model.train() 
		self.model.p = tutils.set_paths(self.p, 'saccade_model')
		
		self.optimizer = Adam(self.model.parameters(), lr=self.model.p['lr'], weight_decay=1e-5)

		self.logger.info('\n Training Saccade Model \n ')
		self.logger.info('Model Overview: \n {} \n'.format(self.model.parameters))
		trainp  = sum(_p.numel() for _p in self.model.parameters() if _p.requires_grad)
		ntrainp = sum(_p.numel() for _p in self.model.parameters() if not _p.requires_grad)
		self.logger.info('Trainable Params {} \n'.format(tutils.group(trainp)))
		self.logger.info('Non-Trainable Params {} \n'.format(tutils.group(ntrainp)))


		#while self.iteration < self.model.p['n_iter']:
		for e in range(self.p['e']):
			self.logger.info(' Training Epoch {} of {} '.format(e+1,self.p['e']))
			self.train_epoch()
		self.plot_loss()

class ObservationTrainer(Trainer):
	def __init__(self, p, dataloader, model):
		super(ObservationTrainer, self).__init__(p, dataloader)
		
		self.model = model
		try:
			self.model.p['datasize']  = len(dataloader[0].dataset)
		except:
			self.model.p['datasize']  = 50000
		self.model.p['n_batches'] = len(dataloader[0])
		self.model.p['n_iter']    = self.model.p['datasize'] * p['e']

		self.optimizer = Adam(self.model.parameters(), lr=p['lr'])		
		
	def train(self):
		
		self.model.train() 

		self.model.p = tutils.set_paths(self.p, 'obs_model')


		self.logger.info('\n Training Observation Model \n ')
		self.logger.info('Model Overview: \n {} \n'.format(self.model.parameters))
		trainp  = sum(_p.numel() for _p in self.model.parameters() if _p.requires_grad)
		ntrainp = sum(_p.numel() for _p in self.model.parameters() if not _p.requires_grad)
		self.logger.info('Trainable Params {} \n'.format(tutils.group(trainp)))
		self.logger.info('Non-Trainable Params {} \n'.format(tutils.group(ntrainp)))

		#while self.iteration < self.model.p['n_iter']:
		for e in range(self.p['e']):
		#while self.iteration < 1:
			self.e = e # add iteration number to self
			self.logger.info(' Training Epoch {} of {} '.format(e+1,self.p['e']))
			self.model.train()
			self.train_epoch()
			with no_grad():
				self.model.eval()
				self.eval_batch(e)
		self.eval_batch(e, force_write=True)
		self.plot_loss()


class pc_cnn_Trainer(Trainer):
	def __init__(self, p, dataloader, model):
		super(pc_cnn_Trainer, self).__init__(p, dataloader)
		
		self.model = model
		try:
			self.model.p['datasize']  = len(dataloader[0].dataset)
		except:
			self.model.p['datasize']  = 50000
		self.model.p['n_batches'] = len(dataloader[0])
		self.model.p['n_iter']    = self.model.p['datasize'] * p['e']

		#self.optimizer = Adam(self.model.parameters(), lr=p['lr'])		
		
	def train(self):
		
		self.model.train() 

		self.model.p = tutils.set_paths(self.p, 'obs_model')


		self.logger.info('\n Training Observation Model \n ')
		# self.logger.info('Model Overview: \n {} \n'.format(self.model.parameters))
		# trainp  = sum(_p.numel() for _p in self.model.parameters() if _p.requires_grad)
		# ntrainp = sum(_p.numel() for _p in self.model.parameters() if not _p.requires_grad)
		# self.logger.info('Trainable Params {} \n'.format(tutils.group(trainp)))
		# self.logger.info('Non-Trainable Params {} \n'.format(tutils.group(ntrainp)))

		#while self.iteration < self.model.p['n_iter']:
		for e in range(self.p['e']):
		#while self.iteration < 1:
			self.e = e # add iteration number to self
			self.logger.info(' Training Epoch {} of {} '.format(e+1,self.p['e']))
			self.model.train()
			
			self.train_epoch_pc_cnn()

			self.model.scheduler.step(self.epoch_loss)

			if e % self.model.p['plot_iter'] == 0:
				tutils.save_checkpoint({'model': self.model, 
								'state_dict': self.model.state_dict(),
								'args': self.model.p}, 
								 self.model.p['model_dir'],  
								 self.model.p['model_name'], 0)	
		# 	with no_grad():
		# 		self.model.eval()
		# 		self.eval_batch(e)
		# self.eval_batch(e, force_write=True)
		# self.plot_loss()
