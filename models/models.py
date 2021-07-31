from .HDRCNN_model import HDRCNN
from .HDRCNN_model_Genr import HDRCNN_Genr
from .HDRCNN_2scale_share_model import HDRCC_2_share
def create_model(opt):
	model = None
	if opt.model == 'test':
		assert (opt.dataset_mode == 'single')
		from .test_model import TestModel
		model = TestModel()
	elif opt.model == 'HDRCNN':
		model = HDRCNN()
	elif opt.model == 'HDRCNN_GEN':
		model = HDRCNN_Genr()
	elif opt.model == 'HDRCNN_2SCALE_SHARE':
		model = HDRCC_2_share()

	model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model
