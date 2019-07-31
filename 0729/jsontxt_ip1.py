import json
from pprint import pprint
import sys

with open('lenet_weights.json')as data_file:
	data = json.load(data_file)
sys.stdout = open('lenet_ip1_weight.txt','w')

#pprint(data)
for i in range (500) :
	for j in range (800) :
		#print i
		#print j
		print(data["ip1"]["data"][i][j])

#with open('lente_weights_ip2.txt','w')as outfile:
#	txt.dump(data_ip2,outfile)

