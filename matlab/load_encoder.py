def get_model(path):
	from model import ObservationModel
	import torch


	modelvars = torch.load(path, map_location='cpu')
	
	modelvars['args']['gpu'] = False
	modelvars['args']['b']	 = 1
	modelvars['args']['foveate'] = True
	modelvars['args']['matlab'] = True
	
	
	model = ObservationModel(modelvars['args'])
	model.load_state_dict(modelvars['state_dict'])	
	model.eval()
	model_dict = {'model':model, 'params': modelvars['args']}

	return model_dict


def decode(model, z):
	from torch import FloatTensor
	return model['model'].decode(FloatTensor(z),0).cpu().numpy()