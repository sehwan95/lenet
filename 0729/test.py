import json
from pprint import pprint
from decimal import Decimal

with open('weights.json')as data_file:
	data = json.load(data_file)

json_shape1 = data["fire2/expand1x1"]["shape"][0]
json_shape2 = data["fire2/expand1x1"]["shape"][1]

for j in range(json_shape1) :
	for i in range(json_shape2) :
		json_data = data["fire2/expand1x1"]["data"][0][i][0][0]
		a=Decimal(json_data)*(2**15)
		b=int(Decimal(a))
		c=float(Decimal(b)/(2**15))
		print(json_data)
		print(c)
