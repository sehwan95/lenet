import json
from pprint import pprint
import sys

with open('lenet_weights.json')as data_file:
	data = json.load(data_file)
sys.stdout = open('lenet_conv1_weight.txt','w')

#pprint(data)
for i in range (20) :
	for j in range (1) :
		for k in range (5) :
			for l in range (5) :
				print(data["conv1"]["data"][i][j][k][l])

