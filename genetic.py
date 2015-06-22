#!/usr/bin/env python
import os,sys
import math
import random
import copy
import numpy
import hashlib

def main():
	pass
	#test_gene()
	#test_indi()
	test_world()

class gene(object):
	def __init__(self,name,descr):
		self._name = name
		self._val  = 0
		self._typ  = 0 # 0=tuple, 1=int range, 2=float
		self._tupl = ()

		if type(descr)==tuple:
			self._typ   = 0
			self._tupl  = descr
		elif type(descr)==list:
			if len(descr)==3:
				self._typ   = 1
				#self._tupl  = numpy.arange(descr[0],descr[1],descr[2]).tolist()
				self._tupl  = (descr[0],descr[1],descr[2])
			elif len(descr)==2:
				self._typ   = 2
				self._tupl  = (descr[0],descr[1])

	@staticmethod
	def randrange_float(start, stop, step):
		return random.randint(0, int(round((stop - start) / step))) * step + start

	def mutate(self):
		if self._typ == 0:
			#self._val = self._tupl[random.randrange(0, len(self._tupl))]
			self._val = random.choice(self._tupl)
		elif self._typ == 1:
			#self._val = self._tupl[random.randrange(0, len(self._tupl))]
			self._val = gene.randrange_float(self._tupl[0],self._tupl[1],self._tupl[2])
			#self._val = int((self._val * 10000) + 0.5) / 10000.0
		else:
			self._val = random.uniform(self._tupl[0],self._tupl[1])
	
	def set(self,val):
		if self._typ == 0:
			if val in self._tupl:
				self._val = val
			else:
				self._val = self._tupl[int(val)%len(self._tupl)]
		elif self._typ == 1: 
			if val in numpy.arange(self._tupl[0],self._tupl[1],self._tupl[2]).tolist():
				self._val = val
		else:
			if self._tupl[0]<=val and val<=self._tupl[1]:
				self._val = val

	def get(self):
		return self._val
	
	def __str__(self):
		return str(self.get())

class indi(object):
	def __init__(self,genes):
		self._genes   = copy.deepcopy(genes)
		self._fitness = -1
		#self._name    = name
		self._name    = self.hash()

#@classmethod
#def crossover(cls,name,ind1,ind2):
#return cls(name,ind._genes)

	@classmethod
	def mitosis(cls,ind):
		return cls(ind._genes)

	def getall(self):
		OUT=[]
		for g in self._genes:
			OUT.append(g.get())
		return OUT

	def findGene(self,genename):
		pos=-1
		for g in range(len(self._genes)):
			if self._genes[g]._name == genename:
				pos=g
				break
		return pos
	
	def getGene(self,genename):
		return self._genes[self.findGene(genename)].get()
	
	def hash(self):
		return hashlib.md5(";".join(str(self.getall())).encode('utf-8')).hexdigest()[0:6]

	def mutateAll(self):
		for g in self._genes:
			g.mutate()
		self._name = self.hash()
	
	def mutate(self,N):
		cnt=0
		while cnt < N:
			mutant=random.randint(0,len(self._genes)-1)
			#print "Mutating: ",self._genes[mutant]._name, " = ", self._genes[mutant].get()
			self._genes[mutant].mutate()
			#print "Result: ",self._genes[mutant]._name, " = ", self._genes[mutant].get()
			cnt+=1
		self._name = self.hash()


	def __str__(self):
		OUT="I'm "+self._name+"! Fitness: " + str(self._fitness) +  "\n "
		for g in self._genes:
			OUT += g._name + ": " + str(g.get()) + "\n " 
		return OUT[:-1]

class GEN(object):
	def __init__(self,indis):
		self._num = 0
		self._indis = []
		for i in indis:
			self._indis.append(copy.deepcopy(i))

	@classmethod
	def clone(cls,proto,N):
		folk = []
		for i in range(N):
			folk.append(indi.mitosis(proto))
		return cls(folk)

	def getWinner(self):
		return self._indis[0]

	def getFitSumN(self,N):
		return sum([i._fitness for i in self._indis[:N]])

	def getFitSum(self):
		return self.getFitSumN(self.getNindis())
	
	def getNindis(self):
		return len(self._indis)
	
	def __str__(self):
		OUT="GEN Nr."+str(self._num)+ " , Fitnesses(hash): "
		", ".join([str(i._fitness) for i in self._indis]) +'\n'
		for i in self._indis:
			OUT += str(i.hash()) +"("+ str(i._fitness)+") , "
		#for i in self._indis:
			#OUT += str(i)
		return OUT

	"Changing Methods"

	def levelUp(self):
		self._num += 1

	def mutateAll(self):
		for i in self._indis:
			i.mutateAll()		
		#self._num += 1

	def mutateAllN(self,N):
		for i in self._indis:
			i.mutate(N)
			hashes = [a.hash() for a in self._indis]
			for tries in range(3):
				if i.hash() in hashes:
					i.mutate(N)
			if i.hash() in hashes:
				i.mutateAll()
		#self._num += 1

	def mutateWorstKbyN(self,K,N):
		for i in self._indis[len(self._indis)-K:]:
			i.mutate(N)
			hashes = [a.hash() for a in self._indis]
			for tries in range(3):
				if i.hash() in hashes:
					i.mutate(N)
			if i.hash() in hashes:
				i.mutateAll()
		#self._num += 1

	def mutateRandomKbyN(self,K,N):
		mutants=[ random.randint(0,len(self._indis)-1) for pick in range(K) ]
		print " to be mutated: ", mutants
		for ind in mutants:
			i = self._indis[ind]
			#print i
			i.mutate(N)
			hashes = [a.hash() for a in self._indis]
			for tries in range(3):
				if i.hash() in hashes:
					i.mutate(N)
			if i.hash() in hashes:
				i.mutateAll()
		#self._num += 1

	def crossover(self):
		for g in range(len(self.getWinner()._genes)):
			genpool=[[ind._genes[g]._val] for ind in self._indis]
			random.shuffle(genpool)
			for ind in range(len(self._indis)):
				self._indis[ind]._genes[g].set(genpool[ind][0])
		#self._num += 1

	def evalFitAll(self):
		for i in self._indis:
			GEN.evalFit(i)
	
	def sortFittest(self):
		winners     = zip(  [i._fitness for i in self._indis]  ,  self._indis  )
		self._indis = list(reversed(zip(*sorted( winners ))[1]))

	def reproNFittest(self,N):
		cntInidis = len(self._indis)
		fitSumN   = self.getFitSumN(N)
		#print fitSumN
		offspringN = []
		for i in self._indis[:N]:
			"calculate portion of children normed to fitness sum, at least one"
			NoOfWinner = max(1, int(1.*cntInidis*(i._fitness/fitSumN)))
			#print int(1.*cntInidis*(i._fitness/fitSumN)), cntInidis, (i._fitness/fitSumN) 
			offspringN.append(NoOfWinner)
		#print cntInidis, offspringN, sum(offspringN)
		"append if list too short, i.e. there are too few"
		if sum(offspringN) < cntInidis :
			for c in range( cntInidis-sum(offspringN) ):
				offspringN[c%len(offspringN)] += 1
		#print cntInidis, offspringN, sum(offspringN)
		"reduce if list too long, i.e. there are too many"
		if sum(offspringN) > cntInidis :
			#print range( N,N-(sum(offspringN)-cntInidis),-1 )
			for c in range( N,N-(sum(offspringN)-cntInidis),-1 ):
				#print c
				offspringN[c%len(offspringN)] -= 1
		print " Fitness dispositions: ",[ ind._fitness for ind in self._indis[:N]], 
		print cntInidis, offspringN, sum(offspringN)
		offspringN = [x-1 for x in offspringN if x > 0]
		Nactual = cntInidis - sum(offspringN)
		looser=Nactual
		for winner in range(Nactual):
			#print offspringN[winner]
			for n in range(offspringN[winner]):
				#print looser, winner,"," ,
				self._indis[looser] = indi.mitosis(self._indis[winner])
				looser+=1
		#self._num += 1
				
	
	@classmethod
	def evalFit(cls,ind):
		fit = ind.getall()[2] + ind.getall()[4]*ind.getall()[0]
		ind._fitness = fit
	


def getBeam():
	beam = []
	beam.append( gene("Energy",  [     1,   100, 0.5]        ) )
	beam.append( gene("Power",   [   0.2,     5, 0.1]        ) )
	beam.append( gene("TunLen",  [    10,  1000,   5]        ) )
	beam.append( gene("TunRad",  [    50,   500,  50]        ) )
	beam.append( gene("TarLen",  [    20,   200,  10]        ) )
	beam.append( gene("TarRad",  [     1,    10, 0.5]        ) )
	beam.append( gene("TarPos",  [ 13963, 14063,  10]        ) )
	beam.append( gene("Cur1",    [   100,   350,  10]        ) )
	beam.append( gene("Cur2",    [   100,   350,  10]        ) )
	beam.append( gene("PosFocT", [     1,     9,   1]        ) )
	beam.append( gene("Mat",     ("myGraphite","myBeryllium","Water","quartz","Tungsten")) )
	return beam

def test_world():
	proto = indi( getBeam())
	Gen0 = GEN.clone(proto,10)
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0
	print 
	print "Selection, Reproduction"
	print 
	Gen0.reproNFittest(3)
	print 
	print "Crossover"
	print 
	Gen0.crossover()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	Gen0.levelUp()
	print Gen0
	print 
	print "Mutation"
	print 
	Gen0.mutateRandomKbyN(5,5)
	Gen0.evalFitAll()
	Gen0.sortFittest()
	Gen0.levelUp()
	print Gen0

def test_mutation():
	proto = indi( getBeam())
	Gen0 = GEN.clone(proto,100)
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0
	print 
	print "Selection, Reproduction"
	print 
	Gen0.reproNFittest(10)
	#Gen0.evalFitAll()
	#Gen0.sortFittest()
	#print Gen0
	print 
	print "Mutation"
	print 
	Gen0.mutateAllN(5)
	Gen0.evalFitAll()
	Gen0.sortFittest()
	Gen0.levelUp()
	print Gen0
	print 
	print "Selection, Reproduction"
	print 
	Gen0.reproNFittest(10)
	#Gen0.evalFitAll()
	#Gen0.sortFittest()
	#print Gen0
	print 
	print "Mutation"
	print 
	Gen0.mutateAllN(5)
	Gen0.evalFitAll()
	Gen0.sortFittest()
	Gen0.levelUp()
	print Gen0
	print 
	print "Selection, Reproduction"
	print 
	Gen0.reproNFittest(10)
	#Gen0.evalFitAll()
	#Gen0.sortFittest()
	#print Gen0
	print 
	print "Mutation"
	print 
	Gen0.mutateAllN(5)
	Gen0.evalFitAll()
	Gen0.sortFittest()
	Gen0.levelUp()
	print Gen0


	#Gen1 = GEN(1,Gen0._indis)
	#Gen1.mutateAll()
	#Gen1.evalFitAll()
	#Gen1.sortFittest()
	#print Gen1
	
	
def test_GEN():
	print "*** Test Generation"
	proto = indi( getBeam())
	proto.mutateAll()
	Gen0 = GEN.clone(proto,10)
	#print Gen0
	print "**** Test Generation all mutated"
	Gen0.mutateAll()
	Gen0.evalFitAll()
	print Gen0
	print Gen0.getFitSum()
	Gen0.sortFittest()
	print Gen0

	
def test_indi2():
	print "*** Test indi, single mutation and mock generation with mutation"
	proto = indi( getBeam())
	print proto
	proto.mutateAll()
	print proto
	gen=[]
	for i in range(2):
		gen.append(indi.mitosis(proto))
		gen[i].mutate(2)	
	for i in range(2):
		print gen[i]


def test_indi():
	c = indi(getBeam())
	print c
	c.mutateAll()
	print c
	c.mutateAll()
	print c
	c.mutate(1)
	print c
	c.mutate(5)
	print c
	
	d = indi.mitosis(c)
	print d


def test_gene():
	TunRad = gene("TunRad",[50,500])
	print TunRad
	TunRad.mutate()
	print TunRad
	TunRad.mutate()
	print TunRad
	TunRad.set(20)
	print TunRad.get()
	TunRad = gene("TunRad",[50,500,10])
	print TunRad
	TunRad.mutate()
	print TunRad
	TunRad.mutate()
	print TunRad
	TunRad.set(70)
	print TunRad.get()
	TunRad = gene("TunRad",("C","W","Be"))
	print TunRad
	TunRad.mutate()
	print TunRad
	TunRad.mutate()
	print TunRad
	TunRad.set(20)
	print TunRad.get()

"""
('a','b','c')
[0, 10, 1]
[0., 1.]

r = gene("TarRad",(1.,10.))
r.mutate()
	> 2.

"""

if __name__ == '__main__':
	main()


