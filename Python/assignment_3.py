from math import sqrt

class Complex:
	def __init__(self, x, y):
		self.real=x
		self.imag=y

	def display(self):
		if self.imag>0:
			ans=str(self.real)+" +"+str(self.imag)+"i"
		else:
			ans=str(self.real)+" "+str(self.imag)+"i"
		print(ans)
	
	def add(self, a):
		c=self
		c.real+=a.real
		c.imag+=a.imag
		return c

	def subtract(self, a):
		c=self
		c.real-=a.real
		c.imag-=a.imag
		return c

	def modulus(self):
		mod=sqrt(self.real**2+self.imag**2)
		return mod

	def conjugate(self):
		c=self
		c.imag=-self.imag
		return c

	def multiplication(self, a):
		r=self.real*a.real - self.imag*a.imag
		i=self.real*a.imag + self.imag*a.real
		c=Complex(r,i)
		return c

	def inverse(self):
		print(self.modulus())
		mod=self.modulus()
		c=Complex(self.real/mod, -self.imag/mod)
		return c
