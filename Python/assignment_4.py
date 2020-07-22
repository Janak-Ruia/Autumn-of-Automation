import numpy as np

def calc():
	y=np.random.randint(5, 500, size=(20, 1), dtype=np.int32)
	x=np.random.normal(size=(20,20))
	xt=x.transpose()
	xm=np.dot(x,xt)
	xm=np.linalg.inv(xm)
	x_final=np.dot(xm, xt)
	return np.dot(x_final, y)

print(calc())



