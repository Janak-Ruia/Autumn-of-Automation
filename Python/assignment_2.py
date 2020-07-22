import math

#image function is needed to handle the case when there are 9 s in the middle of palindromes. 
def image(n):
	length=int(math.log10(n))+1
	if length%2==0:
		q=10**(length/2)
		n=int(n/q)
		og=n
		while og>0:
			n= int(n*10 + og%10)
			og//=10
		return n
	else:
		q=10**(length//2)
		n=int(n/q)
		og=int(n/10)
		while og>0:
			n = int(n*10 + og%10)
			og//=10
		return n



def find(n):
	#assuming n is palindrome
	if n==0:
		return 1
	else:
		length=int(math.log10(n))+1 #no. of digits
		if n==10**length-1:
			return n+2
		elif length%2==0:
			n_dash=int(n+(1.1*10**(length/2)))
			return image(n_dash)

		else:
			n_dash=int(n+10**(length/2))
			return image(n_dash)
