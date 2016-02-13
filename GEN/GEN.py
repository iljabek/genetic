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
#	test_Memory()
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
		#GA parameters shared by methods
		" Probability to mutate a gene "
		self.MutateGeneProb = 0.6      #float 0..1 => (0..1)*NoGenes
		" Probability to mutate an individual "
		self.MutateIndiProb = 0.       #float 0..1 => (0..1)*NoIndis
		" Share (Prob.) of all individuals to reproduce"
		self.WinToReporduceFrac = 0.5  #float 0..1 => (0..1)*NoIndis
		" Elitism, share of individuals which are not modified"
		self.ElitistFrac    = 0.1      #float 0..1 => (0..1)*NoIndis
		# Weights: 0/3=fitness, 1/3=uniform, 2/3=linear, (3=exp)
		self.WeightMode     = 0
#		self.__WeightModes  = 2
		self.GeneCousins    = 0.6			 #float 0..1=> (0..1)*NoGenes

		if isinstance(indis,list):
			for i in indis:
				self._indis.append(copy.deepcopy(i))
		if isinstance(indis,GEN):
			for i in indis:
				self._indis.append(copy.deepcopy(i))
			self._num = indis._num
			self._memory = copy.deepcopy(indis._memory)
#		print self._indis


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
#		for i in self:
#			print i
		OUT += ", ".join([str(i.fitness) for i in self]) +'\n'
		for i in self:
			OUT += str(i.hash) + "  " 
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

		
	def aP(self,f,N=1,Nmin=0):
		"""
			adjustProb
			min( max( f , 0 ) , 1  ) => 0..f..1
			round(f*N)/N => round to the closest binned value
			N is no. of bins.
			# round() vs. int() <=> bin center vs. left edge
		"""
		if N <=0:
			return 0.
		else:
			return ( min( max( round(f * N) , Nmin) , N-1) ) / N

	def checkGAParameters(self):
		"""
			MutateGeneProb = Probabilty, min. 1 gene (pragmatic, not mathematical reasoning)
			MutateIndiProb = Probabilty 
			ElitistFrac = Number, from 0 to all indis
			WinToReporduceFrac = Share; from ElitistFrac to all indis, at least 1
			GeneCousins = Number of genes, from 0 to indi length
		"""
		self.MutateGeneProb = self.aP(self.MutateGeneProb, len(self[0]), 1)
		print "* MutateGeneProb", self.MutateGeneProb, self.MutateGeneProb*len(self[0])
		self.MutateIndiProb = self.aP(self.MutateIndiProb, len(self), 1)
		print "* MutateIndiProb" , self.MutateIndiProb, self.MutateIndiProb *len(self)
		self.ElitistFrac = self.aP(self.ElitistFrac, len(self))
		print "* ElitistFrac" , self.ElitistFrac, self.ElitistFrac *len(self)
		self.WinToReporduceFrac = self.aP(self.WinToReporduceFrac, len(self), 1) 		
		print "* WinToReporduceFrac" , self.WinToReporduceFrac, self.WinToReporduceFrac *len(self)
		self.GeneCousins = self.aP(self.GeneCousins, len(self[0]) , 1.)
		print "* GeneCousins" , self.GeneCousins, self.GeneCousins *len(self[0])
		# Weights: 0=fitness, 1=uniform, 2=linear, (3=exp)
#		self.WeightMode 

	
	def getFitsN(self,N):
		"get list of fitnesses up to N"
		return [i.fitness for i in self[0:N]]

	def getFitSumN(self,N):
		return sum(self.getFitsN(N))	
	
	def getNofFitSum(self, SUMM):
		"take indis making SUMM of the generation fitness" 
		for i in range(len(self)-1):
			if self.getFitSumN(i)/self.getFitSumN(len(self)) >= SUMM:
				return i
		return len(self)


	def getFitProb(self, pos):
		"""
			probability weighted by fitness, # min-capped for indis-to-keep
			depends on self.WinToReporduceFrac fraction
		"""
		N = int( self.WinToReporduceFrac * len(self))
		if pos >= N or N <= 0:
			return 0
#		fits = [ max(i.fitness , 1./N) for i in self[0:N] ]
		fits = [ i.fitness for i in self[0:N] ]
		fitSumN = sum( fits )
		fitProb = fits[pos] / fitSumN
		return fitProb


	def getRankProb(self, pos):
		"""
			linear probability by position=rank
			depends on self.WinToReporduceFrac fraction
			for Rank Selection + Roulette  Wheel
		"""
		N = int( self.WinToReporduceFrac * len(self))
		#N = len(self)
		if pos >= N or N <= 0:
			return 0
		fits = [ max(2.*(N-i)/N/(N+1) , 1./N) for i in range(0,N) ]
		fitSumN = sum( fits )
		fitProb = fits[pos] / fitSumN
		return fitProb

	def getProb(self, pos):
		"""
			depends on self.WeightMode
		"""
		if   self.WeightMode == 0 :#/self.__WeightModes :
			return self.getFitProb(pos)
		elif self.WeightMode == 1 :#/self.__WeightModes :
			return self.getRankProb(pos)
	
	def getHashes(self):
		return [i.hash for i in self]
	
	def getUniqueIndis(self):
		# Magic follows, don't touch!
		## set := only unique, return length of set  
		return len(list(set(self.getHashes())))

	def getUniqueGenes(self):
		u=[]
		# Magic follows, don't touch!
		for g in range(len(self[0])):
			u.append(len(list(set([ind[g].val for ind in self]))))
		return u

	
	def printFitness(self,N):
		print " Fitness dispositions: ",[ '{0:.3g}'.format(f) for f in self.getFitsN(N)], " Sum:", self.getFitSumN(N)

	def printChroms(self,N):
		print " Chrom. dispositions: ",[ f.chrom for f in self[0:N]], " \nSum:", self.getFitSumN(N)

#### Changing Methods

	def levelUp(self):
		self._num += 1


	def fillMemory(self,ind):
		"""
			Memory Structure:
			( [GenNo_1, fitness_1] ,  [GenNo_2, fitness_2] , ... )
		"""
		h = ind.hash
		if not h in self._memory:
			self._memory[h] = [ (self._num,ind.fitness) ]
		else:
			if self._memory[h][-1][0] < self._num and self._memory[h][-1][1] > 0:	
				self._memory[h].append( (self._num,ind.fitness) ) 
			#else	
			#  pass
			#print "  * found "+h+" in the GenMem!"

	def updateMemory(self):
		for i in self:
			self.fillMemory(i)

	def getMemory(self,ind):
		h = ind.hash
		if not h in self._memory:
			return [ (-1,-1) ]
		else:
			return self._memory[h]

	def getFitnessMemory(self,ind):
		m = self.getMemory(ind)
		return m[-1][1]

	def getMeanFitnessMemory(self,ind):
		m = self.getMemory(ind)
		return sum([f[1] for f in m]) / len(m)

	def evalFit(self,ind):
		print "overload me!"
		prefit = self.getMeanFitnessMemory(ind)
		if prefit > 0: #found individual in prev. generations!
			print "    Know "+ ind.hash +" already!"
			return prefit
		fit = 1./( 1. + float(sum(ind.getall())) )
		ind.fitness = fit
	
	def evalFitAll(self):
		for i in self:
			self.evalFit(i)
	
	def sortFittest(self):
		winners     = zip(  [i.fitness for i in self]  ,  self  )
#		print winners
#		self.i = 0
		sortedindis = list(reversed(zip(*sorted( winners ))[1]))
		for i in range(len(sortedindis)):
			self[i] = sortedindis[i].mitosis()


### Selection + Reproduction

	def RouletteWheelIndi(self):
		"""
			depends on self.WinToReporduceFrac fraction
			#depends on self.ElitistFrac number
			depends on self.WeightMode
		"""
		N = int( self.WinToReporduceFrac * len(self))
		r = random.uniform(0.,1.)
		sumRankFit = 0.
		winner = 0
		nTries = 0
#		while r > sumRankFit and nTries < 10*len(self):
#			winner = random.randint( 0 , N )
#			sumRankFit += self.getProb(winner)
#			nTries += 1
#				print " ", winner, sumRankFit, self.getProb(winner),  nTries

		sumRankFit = 0.
		indiQueue = range(len(self))
		random.shuffle(indiQueue)
#		print [ [i] for i in range(len(self)) ] 
#		print len(self), range(len(self)) , indiQueue
		indiFitnesses = [ self.getProb(i) for i in indiQueue  ]
		winner = indiQueue[0]
		for i in range(len(self)):
			sumRankFit += indiFitnesses[i]
#			print " ", i, winner, sumRankFit, self.getProb(winner)
			if r > sumRankFit or i == len(self)-1:
				winner = i
				break
		
#		print "* Roulette needed ",nTries," tries"
		return winner

	def Selection_RouletteWheel(self):
		"""
			#depends on self.WinToReporduceFrac fraction
			depends on self.ElitistFrac number
			depends on self.WeightMode
		"""
		fitSumN = 1.
		W = int( self.ElitistFrac * len(self))
#		N = int( self.WinToReporduceFrac * len(self))
		
		offspring = [] # index of indis
		offspring += range(0,W)

		for i in range(len(self) - len(offspring)):
			winner = self.RouletteWheelIndi()
			offspring.append(winner)
		offspring.sort(reverse=False)
#		offspring = sorted(offspring)
		print "Sorted Offspring: ", offspring
		newindis = [] # indis
 		for i in offspring:
			newindis.append(self[i].mitosis())
		for i in range(len(self)):
			self[i] = newindis[i]


	def Selection_NFittest(self):
		"""
			depends on self.WinToReporduceFrac fraction
			depends on self.ElitistFrac number
			depends on self.WeightMode
		"""	
		N = int( self.WinToReporduceFrac * len(self))
		
		Nindis = len(self)
#		fitSumN   = self.getFitSumN(N)
		#print fitSumN
		offspringN = []
		for i in range(Nindis):
			#"calculate portion of children normed to fitness sum, at least one"
#			NoOfWinner = max(1, int(1.*Nindis*(i._fitness/fitSumN)))
			NoOfWinner = int(1. * Nindis * self.getProb(i))
			offspringN.append(NoOfWinner)
		print Nindis, offspringN, sum(offspringN)

		#"append if list too short, i.e. there are too few"
		if sum(offspringN) < Nindis :
			for c in range( Nindis-sum(offspringN) ):
				offspringN[c%len(offspringN)] += 1
		
		#"reduce if list too long, i.e. there are too many"
		if sum(offspringN) > Nindis :
#			for c in range( N,N-(sum(offspringN)-Nindis),-1 ):
			for c in range( sum(offspringN)-Nindis ):
				offspringN[c%len(offspringN)] -= 1
		
		print " Offspring: ",offspringN, " ; Total: ", sum(offspringN)
		# offspringN has for each position=indi no. of copies
		# reduce by the already "done" N first indis
		offspringN = [x-1 for x in offspringN if x > 0]
		Nactual = Nindis - sum(offspringN)
		# start with the first position to be replaced = N+1
		looser=Nactual
		for winner in range(Nactual):
			#print offspringN[winner]
			for n in range(offspringN[winner]):
				#print looser, winner,"," ,
				self[looser] = indi.mitosis(self[winner])
				looser+=1

	def show_SelectionProb(self):
		"""
			depends on self.WinToReporduceFrac fraction
			depends on self.WeightMode
		"""	
		for i in range(len(self)):
			print i, self.getProb(i), " fit"

### Crossover

	def crossover(self):
		self.mixCrossover()

	def mixCrossover(self):
		"""
			depends on self.ElitistFrac number
		"""
		W = int( self.ElitistFrac * len(self))
		for g in range(len(self[0])): # for each gene
			# encapsulate value in one-element-array
			genpool=[ind[g].val for ind in self]
			random.shuffle(genpool)
#			random.shuffle(genpool) # not necessary, but soothing
			#print genpool
			for ind in range(W,len(self._indis)):
				self[ind][g].val = genpool[ind]

	#def pairedCrossover(self):


### Mutation


	def mutateAll(self):
		for i in self:
			i.mutateAll()		
		#self._num += 1

	def mutateClones(self):
		"""
			depends on self.MutateGeneProb
		"""
		clones2mutate = []       # list of indexes of indis to be mutated
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
		if len(clones2mutate) > 0:
			print "* found clones:" , len(clones2mutate)
			print "* ", clones2mutate
			NGenesTot = int(len(self[0]) * self.MutateGeneProb)
			print "* Mutating ", NGenesTot, " genes"
			for i in clones2mutate:
				self[i].mutateAnyN(NGenesTot)
	
	def mutateCousins(self,Perturb=False):
		"""
			depends on self.MutateGeneProb
			depends on self.GeneCousins
		"""
		cousins2mutate = []       # list of indexes of indis to be mutated
		#get indeces which have earlier accurances
		#by going from start, checking all following
		for i in range(len(self)-1):   # last one never needs to be checked 
			if not i in cousins2mutate:    #else it's a registered clone
				for j in range(i+1,len(self)):
					if not j in cousins2mutate:    #else it's a registered clone
						if self[i] % self[j] == 1:
							cousins2mutate.append(j)

		if len(cousins2mutate) > 0:
			print "* found cousins:" , len(cousins2mutate)
			print "* ", cousins2mutate
			NGenesTot = int(len(self[0]) * self.MutateGeneProb)
			print "* Mutating ", NGenesTot, " genes"
			for i in cousins2mutate:
				if Perturb==True:
					self[i].mutateAnyN(NGenesTot)
				else:
					self[i].perturbAll()
	

	def mutateRandom(self):
		"""
			depends on self.ElitistFrac number
			depends on self.MutateGeneProb
			depends on self.MutateIndiProb
		"""
		W = int( self.ElitistFrac * len(self))
		NGenesTot = int(len(self[0]) * self.MutateGeneProb)
		
		print "* Mutated: ",
		for i in range(W,len(self)):
			if random.uniform(0.,1.) < self.MutateIndiProb:
				print i,
				self[i].mutateAnyN(NGenesTot)
		print

	def perturbRandom(self):
		"""
			depends on self.ElitistFrac number
			depends on self.MutateIndiProb
		"""
		W = int( self.ElitistFrac * len(self))
		
		print "* Mutated: ",
		for i in range(W,len(self)):
			if random.uniform(0.,1.) < self.MutateIndiProb:
				print i,
				self[i].perturbAll()
		print

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
	Gen0 = GEN.clone(proto,30)
	
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	#print Gen0[0]
	writeStateToFile(Gen0,"./evol0/",True,True) #joined,gziped
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0[0]
	writeStateToFile(Gen0,"./evol0/",True,True) #joined,gziped
	del Gen0
	Gen0 = 	readStateFromFile("./evol0/",True,True) #joined,gziped
	print Gen0[0]

	Gen0.mutateClones()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	#print Gen0[0]
	writeStateToFile(Gen0,"./evol0/",False,True) #joined,gziped
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0[0]
	writeStateToFile(Gen0,"./evol0/",False,True) #joined,gziped
	del Gen0
	Gen0 = 	readStateFromFile("./evol0/",False,True) #joined,gziped
	print Gen0[0]

	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	#print Gen0[0]
	writeStateToFile(Gen0,"./evol0/",True,False) #joined,gziped
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0[0]
	writeStateToFile(Gen0,"./evol0/",True,False) #joined,gziped
	del Gen0
	Gen0 = 	readStateFromFile("./evol0/",True,False) #joined,gziped
	print Gen0[0]
	
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	#print Gen0[0]
	writeStateToFile(Gen0,"./evol0/",False,False) #joined,gziped
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0[0]
	writeStateToFile(Gen0,"./evol0/",False,False) #joined,gziped
	del Gen0
	Gen0 = 	readStateFromFile("./evol0/",False,False) #joined,gziped
	print Gen0[0]


def test_Memory():
	print "*** Test Generation"
	proto = indi( getCar())
	proto.mutateAll()
	Gen0 = GEN.clone(proto,9)
	Gen0.evalFitAll()
	Gen0.sortFittest()
	
	print Gen0._memory
	
	Gen0.updateMemory()
	print Gen0._memory
	
	Gen0.mutateClones()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	Gen0.updateMemory()
	print Gen0._memory

	Gen0.levelUp()
	Gen0.updateMemory()
	print Gen0._memory



def test_GEN():
	print "*** Test Generation"
	proto = indi( getCar())
	proto.mutateAll()
	Gen0 = GEN([proto])
	print "minimal Gen: " , Gen0

	print "----------"
	Gen0 = GEN.clone(proto,100)
	Gen0.MutateGeneProb = 0.5      #float 0..1
	Gen0.MutateIndiProb = 0.2       #float 0..1
	Gen0.WinToReporduceFrac = 0.5 #float 0..1
	Gen0.ElitistFrac = 0.1      #int 0..len(self)
	# Weights: 0=fitness, 1=uniform, 2=linear, (3=exp)
	Gen0.WeightMode = 0 
	Gen0.GeneCousins = 0.2 
	Gen0.checkGAParameters()
	print "Unique Indis: ", Gen0.getUniqueIndis() , " , Unique Genes: ", Gen0.getUniqueGenes()
#	print Gen0
	Gen0.mutateClones()
	print "mutateClones"
	print "Unique Indis: ", Gen0.getUniqueIndis() , " , Unique Genes: ", Gen0.getUniqueGenes()
#	print Gen0
	Gen0.mutateRandom()
	print "mutateRandom"
	print "Unique Indis: ", Gen0.getUniqueIndis() , " , Unique Genes: ", Gen0.getUniqueGenes()
#	print Gen0
	Gen0.perturbRandom()
	print "perturbRandom"
	print "Unique Indis: ", Gen0.getUniqueIndis() , " , Unique Genes: ", Gen0.getUniqueGenes()
#	print Gen0
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print "Unique Indis: ", Gen0.getUniqueIndis() , " , Unique Genes: ", Gen0.getUniqueGenes()
#	print Gen0	
	Gen0.evalFit = evalFitOverload
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print "Unique Indis: ", Gen0.getUniqueIndis() , " , Unique Genes: ", Gen0.getUniqueGenes()
#	print Gen0

	print "----------"
	Gen0.WinToReporduceFrac = 0.5 
	Gen0.WeightMode = 0
#	print Gen0.WeightMode, Gen0.WinToReporduceFrac, Gen0.ElitistFrac, Gen0.MutateGeneProb, Gen0.MutateIndiProb 
	Gen0.checkGAParameters()
#	print Gen0.WeightMode, Gen0.WinToReporduceFrac, Gen0.ElitistFrac, Gen0.MutateGeneProb, Gen0.MutateIndiProb 

	print "----------"
	fs,rs,ps = 0,0,0
	for i in range(len(Gen0)):
		print i, Gen0[i].fitness, Gen0.getFitProb(i), Gen0.getRankProb(i), Gen0.getProb(i)
		fs += Gen0.getFitProb(i)
		rs += Gen0.getRankProb(i)
		ps += Gen0.getProb(i)
	print "Sum of Probs:", Gen0.getFitSumN(5), fs,rs,ps

	Gen0.WeightMode = 1 #/ Gen0.__WeightModes
	for i in range(len(Gen0)):
		print Gen0.getProb(i)
	print "----------"

	Gen0.Selection_RouletteWheel()
	print "perturbRandom"
	print "Unique Indis: ", Gen0.getUniqueIndis() , " , Unique Genes: ", Gen0.getUniqueGenes()
#	print Gen0
	Gen0.evalFitAll()
	Gen0.sortFittest()

if __name__ == '__main__':
	main()
