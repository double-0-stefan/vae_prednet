from os.path import join  
import yaml 
import numpy as np

from utils import train_utils as tutils 
from utils import model_utils as mutils 
from utils import data_utils  as dutils

from model import *
from trainer import * 
 
from pprint import pprint, pformat
from logging import shutdown
from torch.nn import MSELoss, Module, CrossEntropyLoss
from utils import elbo_decomposition 
from torch import cuda, no_grad, isnan, load
from torch.autograd import Variable, set_detect_anomaly, detect_anomaly
from argparse import ArgumentParser
import glob 

def main(args):	
	
	largs = yaml.load(open(args.config), Loader=yaml.SafeLoader)	
	logger, logf = tutils.logs(largs['exp_name'])
	logger.info(args)
	
	if largs['generate_data']:
		dutils.generate_env_data()

	#set_detect_anomaly(True)
	#logger.warning('Anomaly detection on - disable for performance')
	
	argv = tutils.product_dict(largs)
	nrun = len([x for x in tutils.product_dict(largs)])
	f = 0 ; flist = []

	for perm, mvars in enumerate(argv):
		try:
			logger.info('Training')
			logger.info('Reading permutation : {} of {}'.format(perm, nrun))

			# Initialise parameters
			mvars = tutils.arg_check(mvars, perm)
			
			# Initialise Model
			pprint(mvars)
			
			if mvars['interactive']:
				model = PrednetWorldModel(mvars)
				trainer = InteractiveTrainer(mvars, model)
				trainer.train()
				del trainer, model 
			
			else:
				
				if mvars['prednet']:
					obs_model = GenerativeModel(mvars).cuda()
				elif mvars['pc_cnn']:
					if mvars['xla']:
						obs_model = pc_conv_network(mvars).to(xm.xla_device())
					else:
						obs_model = pc_conv_network(mvars).cuda()
				else:
					obs_model = ObservationVAE(mvars).cuda()
				
				# Initialise Dataloader(s)
				if largs['train_gx']:
					if largs['pc_cnn']:
						logger.info('Training Observation Model - pc_cnn')
						data = dutils.get_dataset(mvars, split='train',static= not mvars['dynamic'])
						gx_trainer  = pc_cnn_Trainer(mvars, data, obs_model)
						gx_trainer.train()
						del gx_trainer, obs_model
					else:
						logger.info('Training Observation Model')
						data = dutils.get_dataset(mvars, split='train',static= not mvars['dynamic'])
						gx_trainer  = ObservationTrainer(mvars, data, obs_model)
						gx_trainer.train()
						del gx_trainer, obs_model
					

				if largs['train_fx']:
					logger.info('Training Transition Model')
					data = dutils.get_dataset(mvars, split='train', static=True)
					if mvars['dataset'] == 'mnist':
						fx_trainer  = SaccadeTrainer(mvars, data)
					else:
						fx_trainer = TransitionTrainer(mvars, data)
					fx_trainer.train()
					del fx_trainer
						
		except Exception as e:     # most generic exception you can catch
			
			logf.error('Error - config {}'.format(perm))
			logger.exception(e)
			f += 1 ; flist.append(perm)
			0/0


	logf.info('\n Total Evaluation Failures : {} / {} '.format(f, perm+1))	
	logf.info(pprint(flist))
	shutdown() # close logs

if __name__ == '__main__':
	ap = ArgumentParser()
	ap.add_argument("-m", '--config', required=False, default="parameters.yaml", help="Path to model config", dest='config')
	main(ap.parse_args())