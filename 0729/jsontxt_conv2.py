import json
from pprint import pprint
import sys

with open('lenet_weights.json')as data_file:
	data = json.load(data_file)
sys.stdout = open('lenet_conv2_weight.txt','w')

#pprint(data)
for i in range (50) :
	for j in range (20) :
		for k in range (5) :
			for l in range (5) :
				print(data["conv2"]["data"][i][j][k][l])

