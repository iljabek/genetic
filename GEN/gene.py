#!/usr/bin/env python
"GAH: Gene"

#import os,sys
import random
import decimal


def main():
	pass
	test_gene()

class gene(object):
	"""
Gene. 
	has val from max..min
	setter truncates input outside range
	mutate sets val from a uniform distribution
	"""

	def __init__(self, inval):
		self._min = 0.
		self._max = 1.
		self.val = inval
#		self._val = decimal.Decimal(0).quantize(decimal.Decimal('1.E-7'), rounding=decimal.ROUND_DOWN)
	
	@property
	def max(self):
		"Getter for max value"
		return self._max
	@max.setter
	def max(self, inmax):
		"max setter, must be >= min"
		self._max = max(inmax, self.min)

	@property
	def min(self):
		"Getter for min value"
		return self._min
	@min.setter
	def min(self, inmin):
		"min setter, must be <= max"
		self._min = min(inmin, self.max)

	@property
	def val(self):
		"Getter for current value"
		return self._val
	@val.setter
	def val(self, inval):
		"""
		Setter puts inval into min.....val......max range
		"""
		tmpval = min( max(inval, self.min) , self.max)
#		self._val = tmpval
		self._val = decimal.Decimal(tmpval).quantize(decimal.Decimal('1.E-8'), rounding=decimal.ROUND_DOWN)
		if tmpval != inval:
			msg = "* gene input " + str(inval) + " outside range " 
			msg += str(self.min) + ".." + str(self.max) 
			msg += ", set to " +str(self.val)
			print msg
	
	def __str__(self):
		return str(self.val)+ " in (" +str(self.min) + ".." + str(self.max)+ ")"

	def copy(self):
		c = gene(self.val)
		c.min = self.min
		c.max = self.max
		return c
		
	def __mul__(self , other):
		"A*B: mean of each gene"
		c = self.copy()
		c.val = (self.val + other.val)/decimal.Decimal(2.)
		return c
	
	def __add__(self , other):
		c = self.copy()
		rnd = random.random() >= 0.5
		if rnd: c.val = self.val
		else: c.val = other.val
		return c
	
	def mutate(self): # shuffle
		"new value within range"
		self.val = random.uniform(self.min , self.max)

	def perturb(self): # shuffle
		"move by small amount around the current value"
		self.val = random.gauss( float(self.val) , (self.max-self.min)/100. )

		pass


def test_gene():
	cog = gene(0.)
	print cog
	print cog.val
	cog.val = 0.2
	print cog
	cog.val = 1.2
	print cog
	cog.val = -0.2
	print cog
	cog.mutate()
	print "mutate: ", cog
	cog.mutate()
	print "mutate: ", cog
	sproc = gene(0.2)
	print "2nd:", sproc
	print "A+B",cog+sproc
	print "A+B",cog+sproc
	print "A+B",cog+sproc
	print "A*B",cog*sproc

	print "pre: ", cog
	cog.perturb()
	print "pert:", cog




if __name__ == '__main__':
	main()
