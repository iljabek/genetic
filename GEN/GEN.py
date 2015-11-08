#!/usr/bin/env python
"GAH: Generation"

#import os,sys
#import math
import random
import copy
#import numpy
import hashlib

from indi import indi
from gene import gene


def main():
	pass
	test_GEN()
#	test_IO()

class GEN(object):
	"""
Generation.
	contains individuals
	defines global operators
	"""
	def __init__(self,indis):
		"""
		takes a list of indis, or another GEN for copying
		"""
		self._num = 0
		self._memory = {}
		self._i = 0 # positoin for iter and next
		self._indis  = []  # list of indis
#		self._indis  = copy.deepcopy(indis) # list of indis
		if isinstance(indis,list):
			for i in indis:
				self._indis.append(copy.deepcopy(i))
		if isinstance(indis,GEN):
			for i in indis:
				self._indis.append(copy.deepcopy(i))
			self._num = indis._num
			self._memory = copy.deepcopy(indis._memory)
		print self._indis


	def __getitem__(self, item):
		return self._indis[item]
#		if isinstance(item, slice):
#			return indi(self._indis[item])
#		else:
#			return self._indis[item]

	def __setitem__(self, item, inindi):
		"""
		sets slice of or a single indi
		"""
		if isinstance(item, slice):
			for i in range(item.start,item.stop):
				self[i] = val[i]
		else:
			if isinstance(inindi, indi):
				self._indis[item] = inindi

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
			return self._indis[self._i-1] 
	
	def __len__(self):
		return len(self._indis)


	def __str__(self):
		OUT = "GEN Nr."+str(self._num)+ ": "
		for i in self:
			print i
		OUT += ", ".join([str(i.fitness) for i in self]) +'\n'
		for i in self:
			OUT += str(i.hash) +" ("+ str(i.fitness)+" fit) , "
		return OUT



### GA relevant functions below

	@classmethod
	def clone(cls,proto,N):
		"""
			clone from proto-individuum N-times => populate
			use constructor in the end
		"""
		folk = []
		for i in range(N):
			folk.append(proto.mitosis())
		return cls(folk)

	def getWinner(self):
		return self[0]
	
	def getFitN(self,N):
		"get list of fitnesses up to N"
		return [i.fitness for i in self[0:N]]

	def getFitSumN(self,N):
		return sum(self.getFitN(N))	
	
	def getHashes(self):
		return [i.hash for i in self]
	
	def getUniqueGenes(self):
		u=[]
		# Magic follows, don't touch!
		## set = only unique 
		for g in range(len(self.getWinner())):
			u.append(len(list(set([ind[g] for ind in self]))))
		return u

	
	def printFitness(self,N):
#		subGen = self[0:N]
		print " Fitness dispositions: ",[ '{0:.3g}'.format(f) for f in self.getFitN(N)], " Sum:", Gen.getFitSumN(N)

#### Changing Methods

	def levelUp(self):
		self._num += 1

	def fillMemory(self,ind):
		h=ind.hash
		if not h in self._memory:
			self._memory[h] = ind.fitness
		#else:
			#pass
			#print "  * found "+h+" in the GenMem!"

	def updateMemory(self):
		for i in self:
			h=i.hash
			if not h in self._memory:
				self._memory[h] = i.fitness
			#else:
				#print "  * found "+h+" in the GenMem!"

	def getMemory(self,ind):
		h=ind.hash
		if not h in self._memory:
			return -1
		else:
			return self._memory[h]


	def evalFit(self,ind):
		print "overload me!"
		fit = sum(ind.getall()[:5])
		ind.fitness = fit
	
	def evalFitAll(self):
		for i in self:
			self.evalFit(i)
	
	def sortFittest(self):
		winners     = zip(  [i.fitness for i in self]  ,  self  )
		print winners
#		self.i = 0
		sortedindis = list(reversed(zip(*sorted( winners ))[1]))
		for i in range(len(sortedindis)):
			self[i] = sortedindis[i].mitosis()


### Selection + Reproduction


	def getRankProb(self, pos):
		"""
			linear probability by position=rank
			for Rank Selection + Roulette  Wheel
		"""
		Nall = 1.
#		i0 = len(self)
		N = len(self)
		if pos > N:
			return 0
		else:
#			return 2.*Nall/i0 * (1. - pos/(i0-1))
			return 2.*(N-pos)/N/(N+1) 

	def Selection_RouletteWheel(self):
		"""
		
		"""
		fitSumN = 1.
		offspring = []
		for i in range(len(self)):
			r = random.uniform(0.,1.)
			sumRankFit = 0.
			winner = 0
			nTries = 0
			while r > sumRankFit and nTries < 10*len(self):
				winner = random.randint(0,len(self))
				sumRankFit += self.getRankProb(winner)
				nTries += 1
#				print " ", winner, sumRankFit, self.getRankProb(winner),  nTries
			offspring.append(winner)
		print "Sorted Offspring: ", sorted(offspring)
		newindis = []
		for i in sorted(offspring):
			newindis.append(self[i].mitosis())
		for i in range(len(self)):
			self[i] = newindis[i]


	def Selection_NFittest(self,N):
		cntInidis = len(self)
		fitSumN   = self.getFitSumN(N)
		#print fitSumN
		offspringN = []
		for i in self[0:N]:
			#"calculate portion of children normed to fitness sum, at least one"
			NoOfWinner = max(1, int(1.*cntInidis*(i._fitness/fitSumN)))
			#print int(1.*cntInidis*(i._fitness/fitSumN)), cntInidis, (i._fitness/fitSumN) 
			offspringN.append(NoOfWinner)
		#print cntInidis, offspringN, sum(offspringN)

		#"append if list too short, i.e. there are too few"
		if sum(offspringN) < cntInidis :
			for c in range( cntInidis-sum(offspringN) ):
				offspringN[c%len(offspringN)] += 1
		#print cntInidis, offspringN, sum(offspringN)
		
		#"reduce if list too long, i.e. there are too many"
		if sum(offspringN) > cntInidis :
			#print range( N,N-(sum(offspringN)-cntInidis),-1 )
			for c in range( N,N-(sum(offspringN)-cntInidis),-1 ):
				#print c
				offspringN[c%len(offspringN)] -= 1
		#print " Fitness dispositions: ",[ ind._fitness for ind in self[:N]], 
		print offspringN, sum(offspringN)
		# offspringN has for each position=indi no. of copies
		# reduce by the already "done" N first indis
		offspringN = [x-1 for x in offspringN if x > 0]
		Nactual = cntInidis - sum(offspringN)
		# start with the first position to be replaced = N+1
		looser=Nactual
		for winner in range(Nactual):
			#print offspringN[winner]
			for n in range(offspringN[winner]):
				#print looser, winner,"," ,
				self[looser] = indi.mitosis(self[winner])
				looser+=1
		#self._num += 1	


### Mutation


	def mutateAll(self):
		for i in self:
			i.mutateAll()		
		#self._num += 1

	def mutateClonesByN(self,N):
		clones2mutate = []       # list of indeces of indis to be mutated
		allHashes = self.getHashes()       # list of hashes
		#get indeces which have earlier accurances
		#by going from start, checking all following
		#print allHashes 
		for i in range(len(allHashes)-1):   # last one never needs to be checked 
			if not i in clones2mutate:    #else it's a registered clone
				j=i+1    #start checking from next
				while allHashes[i] in allHashes[j:]:
					#print "   looking at ",j, "at ", allHashes.index(allHashes[i],j),
					j = allHashes.index(allHashes[i],j)
					#print " found ",allHashes[i], " at ",j, " total ", len(clones2mutate)
					clones2mutate.append(j)
					j+=1
					#print "   to check", allHashes[j:]
					#if len(clones2mutate)>=K:
						#break
				#if len(clones2mutate)>=K:
					#break
		print "* found clones:" , len(clones2mutate)
#		print "* ", clones2mutate
		for i in clones2mutate:
			self[i].mutateAnyN(N)


def writeStateToFile(Gen, path,joined=True,gziped=True):
	import pickle
	import gzip
	import os
	
	fullpath=""
	if joined:
		fullpath = path+"/GenAll.pkl"
	else:
		num = '{0:04d}'.format(Gen._num)
		fullpath = path+"/GenN"+num+".pkl"
	
	if gziped:
		fullpath += ".gz"
	
	if joined and os.path.isfile(fullpath):
		#"Write to the beginning of the .pkl file if not first, gzip if needed"
		#"use temporary file, append the old"

		if gziped:
			f = gzip.open(fullpath,"rb")
			newf = gzip.open("./Gen____.tmp", "wb")
		else:
			f = open(fullpath,"rb")
			newf = open("./Gen____.tmp", "wb")
		pickle.dump( Gen , newf )
		for chunk in iter(lambda: f.read(1024), b""):
			newf.write(chunk)
		newf.close()
		f.close()
		os.remove(fullpath)
		os.rename("./Gen____.tmp",fullpath)
	else:
		#"if not joined or first joined file"
		if gziped:
			f = gzip.open(fullpath,"wb")
		else:
			f = open(fullpath,"wb")
		pickle.dump( Gen , f )
		f.close()
				
	#pickle.dump( Gen0, open( evolpath+"/Gen0.pkl", "wb" ) )
	#pickle.dump( Gen0, open( evolpath+"/GenAll.pkl", "wb" ) )
	#pickle.dump( Gen, gzip.open( evolpath+"/GenAll.pkl.gz", "wb" ) )

def readStateFromFile(path,joined=True,gziped=True,N=0):
	import glob
	import pickle
	import gzip
	import os
	
	fullpath=""
	
	if joined:
		fullpath = path+"/GenAll.pkl"
		if gziped: 
			fullpath += ".gz"
	else:
		maxGen=0
		if gziped: 
			ext=".pkl.gz"
		else: 
			ext=".pkl"
		for fn in glob.glob(path+"/GenN*"+ext):
			fnGen = int(fn[fn.index("GenN")+len("GenN"):fn.index(".pkl")]) # *GenNXXXX.pkl*
			if fnGen >= maxGen: 
				maxGen = fnGen
				fullpath = fn
				
	print fullpath
	if gziped: 
		f = gzip.open(fullpath,"rb")
	else:
		f = open(fullpath,"rb")
		
	for i in range(N+1): GenMax = pickle.load(f)
	f.close()
	return GenMax


def evalFitOverload(ind):
	print "I'm monkey-patched"
	fit = sum(ind.getall())
	ind.fitness = fit

def getCar():
	car = []
	car.append( gene(0) )
	car.append( gene(0) )
	car.append( gene(0) )
	car.append( gene(0) )
	car.append( gene(0) )
	car.append( gene(0) )
	car.append( gene(0) )
	return car

def test_IO():
	proto = indi( getCar())
	Gen0 = GEN.clone(proto,10)
	
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	#print Gen0.getWinner()
	writeStateToFile(Gen0,"./evol0/",True,True) #joined,gziped
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0.getWinner()
	writeStateToFile(Gen0,"./evol0/",True,True) #joined,gziped
	del Gen0
	Gen0 = 	readStateFromFile("./evol0/",True,True) #joined,gziped
	print Gen0.getWinner()

	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	#print Gen0.getWinner()
	writeStateToFile(Gen0,"./evol0/",False,True) #joined,gziped
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0.getWinner()
	writeStateToFile(Gen0,"./evol0/",False,True) #joined,gziped
	del Gen0
	Gen0 = 	readStateFromFile("./evol0/",False,True) #joined,gziped
	print Gen0.getWinner()

	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	#print Gen0.getWinner()
	writeStateToFile(Gen0,"./evol0/",True,False) #joined,gziped
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0.getWinner()
	writeStateToFile(Gen0,"./evol0/",True,False) #joined,gziped
	del Gen0
	Gen0 = 	readStateFromFile("./evol0/",True,False) #joined,gziped
	print Gen0.getWinner()
	
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	#print Gen0.getWinner()
	writeStateToFile(Gen0,"./evol0/",False,False) #joined,gziped
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0.getWinner()
	writeStateToFile(Gen0,"./evol0/",False,False) #joined,gziped
	del Gen0
	Gen0 = 	readStateFromFile("./evol0/",False,False) #joined,gziped
	print Gen0.getWinner()



def test_GEN():
	print "*** Test Generation"
	proto = indi( getCar())
	proto.mutateAll()
	Gen0 = GEN([proto])
	print "minimal Gen: " , Gen0
	print "----"

	Gen0 = GEN.clone(proto,10)
	print Gen0
	Gen0.mutateClonesByN(2)
	print Gen0
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0	
	Gen0.evalFit = evalFitOverload
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0	


if __name__ == '__main__':
	main()
