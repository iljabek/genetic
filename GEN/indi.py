#!/usr/bin/env python
"GAH: Indi"

#import os,sys
#import math
import random
import copy
#import numpy
import hashlib

from gene import gene


def main():
	pass
	test_indi()

class indi(object):
	"""
Individual.
	contains genes, defining uniqe hash
	can be iterated over genes
	can mutate all or some genes
	"""
	def __init__(self,genes): #,invals=[]):
		self._genes  = copy.deepcopy(genes) # list of genes
		self.fitness = -1.e-12 # bad for integral
		self._i = 0 # positoin for iter and next
		"""
		if len(invals) == len(genes):
			for i in range(len(self._genes)): 
				self._genes[i].vals = invals[i] 
		"""

	"""
	@property
	def genes(self):
		"Getter for genes"
		return self._genes
	@genes.setter
	def genes(self, ingenes):
		""
		if isinstance(ingenes, list):
			if len(ingenes) == len(self):
				if all( [isinstance(g, gene) for g in ingenes] ):
					for i in range(len(self)):
						self[i].val = ingenes[i].val
				else:
					for i in range(len(self)):
						self[i].val = ingenes[i]
		else:
			for i in range(len(self)):
				self[i].val = ingenes
	"""

	def __getitem__(self, item):
		if isinstance(item, slice):
			return indi(self._genes[item])
		else:
			return self._genes[item]

	def __setitem__(self, item, val):
		if isinstance(item, slice):
			for i in range(item.start,item.stop):
				self[i] = val[i]
		else:
			if isinstance(val, gene):
				self._genes[item] = val
			else:
				self._genes[item].val = val

	def __iter__(self):
		if self._i >= len(self):
			self._i = 0
		return self

	def next(self):
		if self._i >= len(self):
			self._i = 0
			raise StopIteration()
		else:
			self._i += 1 
			return self._genes[self._i-1] 
	
	def __len__(self):
		return len(self._genes)

	def __cmp__(self, other): # 
		if self.fitness > other.fitness:
			return 1
		elif self.fitness < other.fitness:
			return -1
		else:
			return 0

	"""	
	def __lt__(self, other): # <
		return self.fitness < other.fitness
	def __gt__(self, other): # >
		return self.fitness > other.fitness
	def __le__(self, other): # <=
		return self.fitness <= other.fitness
	def __ge__(self, other): # >=
		return self.fitness >= other.fitness
	def __eq__(self, other): # ==
		return self.fitness == other.fitness
	def __ne__(self, other): # !=
		return self.fitness != other.fitness
	"""

	def getall(self):
		return [g.val for g in self._genes]

	@property
	def hash(self):
		"Hashing using all genes' values, makes inidis uniqe"
		c = ";".join([str(i) for i in self.getall()])
		cm = str( hashlib.md5(  c  ).hexdigest().encode('utf-8') )
		icm = cm[0:7]
#		icm = "{0:09d}".format(int(  cm[0:7]   ,16))
		return icm

	def __str__(self):
		OUT="I'm "+self.hash+"! Fitness: " + str(self.fitness) +  "\n "
		for g in self:
			OUT += str(g) + " " 
		return OUT

### GA relevant functions below

	def mitosis(self):
		clone = indi(self._genes)
		clone.fitness = self.fitness
		return clone 


#	def mutateI(self,I):
#		self[I].mutate()
	
	def mutateAll(self):
		for i in range(len(self)):
			self[i].mutate()
	
	def perturbateAll(self):
		for i in range(len(self)):
			self[i].perturbate()
	
	def mutateAnyN(self,N):
		mutees = []
		for i in range(N):
			#choose random gene from all
			mutee = random.randint(0,len(self)-1)
			#don't mutate same gene twice
			if (mutee in mutees):
				tries=0
				while (mutee in mutees) or tries<len(self):
					#rechoose until new
					mutee = random.randint(0,len(self)-1)
					tries+=1
			self[mutee].mutate()
#			self.mutateI(mutee)
			mutees.append(mutee)

	def __add__(self , other):
		"A+B: create offspring with random genes from both parents"
		child = self.mitosis()
		for i in range(len(self)):
			child[i] = self[i] + other[i]
		return child

	def __mul__(self , other):
		"A*B: create offspring with mean of each gene from both parents"
		child = self.mitosis()
		for i in range(len(self)):
			child[i] = self[i] * other[i]
		return child

def evalFit(ind):
	print "overload me!"
	fit = sum(ind.getall())
	ind.fitness = fit
	return fit


def getCar():
	car = []
	car.append( gene(0) )
	car.append( gene(0) )
	car.append( gene(0) )
	car.append( gene(0) )
	car.append( gene(0) )
	return car

def test_indi():
	jeep = indi(getCar())
	print "First gene: ", jeep[0]
	for i in jeep:
		print i
	print "New Indi: "
	print jeep
	for i in jeep:
		i.mutate()
	print "All Single Gene Mutated Indi: "
	print jeep
	print "Single Gene Mutated Indi: "
	jeep[1].mutate()
	evalFit(jeep)
	print "Single Gene Mutated Indi: "
	print jeep
	jeep.mutateAll()
	evalFit(jeep)
	print "Mutate All: "
	print jeep	
	jeep.perturbateAll()
	evalFit(jeep)
	print "Perturbate All: "
	print jeep
	print "Mutate Any 2: "
	jeep.mutateAnyN(2)
	evalFit(jeep)
	print "first", jeep
	jag = jeep.mitosis()
	jag.mutateAll()
	evalFit(jeep)
	print "first", jeep	
	for g in jeep:
		print "  ", id(g)
	print "second", jag	
	for g in jag:
		print "  ", id(g)

	print "first+second : ", jeep+jag
	print "first*second : ", jeep*jag
	jeep[0:2] = jag[0:2]
	print "first", jeep	
	for g in jeep:
		print "  ", id(g)
	print "second", jag	
	for g in jag:
		print "  ", id(g)

if __name__ == '__main__':
	main()
