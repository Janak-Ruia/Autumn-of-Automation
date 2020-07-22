import math

def is_prime(n):
	if n<=1:
		return False
	elif n%2==0:
		return False
	else:
		for i in range(3, int(math.sqrt(n)+1), 2):
			if n%i==0:
				return False
		return True

def twin_prime(n1):
	n2=n1+2
	if is_prime(n1):
		if is_prime(n2):
			return True
	else:
		return False

def list_twins(l, u):
	twins=[]
	for i in range(l+1, u):
		if twin_prime(i) and i+2<u:
			twins.append((i,i+2))
	return twins

def find_twins(d):
	l=10**(d-1)
	u=l*10
	return list_twins(l, u)


d=int(input("Enter no. of digits: "))
twins=find_twins(d)
f=open("myFirstFile.txt", "w+")
for i in range(len(twins)):
	f.write(f"{twins[i]}  \n")

