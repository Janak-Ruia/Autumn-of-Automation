import numpy as np
import math
n=int(input())
numbers=list(input("Enter numbers: ").split(' '))
p=np.empty((n, 1), int)
for i in range(n):
	p[i]=numbers[i]
fin_list=dict()
for i in range(n):
	maxi=0
	max_index=-1
	for j in range(i, n):
		diff=p[j]-p[i]
		if diff>maxi:
			maxi=diff
			max_index=i
	fin_list[max_index]=maxi

max_index_fin=max(fin_list, key=fin_list.get)
print(int(fin_list[max_index_fin]), '\n ', max_index_fin+1)

