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
			latent_sample.append(cat_sample_low)
			
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