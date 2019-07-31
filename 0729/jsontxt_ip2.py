import json
from pprint import pprint
import sys

with open('lenet_weights.json')as data_file:
	data = json.load(data_file)
sys.stdout = open('lenet_ip2_weight.txt','w')

#pprint(data)
for i in range (10) :
	for j in range (500) :
		#print i
		#print j
		print(data["ip2"]["data"][i][j])

#with open('lente_weights_ip2.txt','w')as outfile:
#	txt.dump(data_ip2,outfile)

