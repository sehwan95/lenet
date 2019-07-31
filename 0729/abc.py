caffe_root = '/home/socmgr/caffe/python/'

import sys
import json
sys.path.insert(0,caffe_root + 'python')
import caffe

prototxt = '/home/socmgr/caffe/examples/mnist/lenet.prototxt'
caffemodel = '/home/socmgr/caffe/examples/mnist/lenet_iter_5000.caffemodel'

dict_params = {}
net = caffe.Net(prototxt,caffemodel,caffe.TEST)
for param in net.params:
	dict_params[param] = {'data': net.params[param][0].data.tolist(),
				'shape': net.params[param][0].data.shape}

with open('lenet_weights.json','w')as outfile:
	json.dump(dict_params,outfile)
