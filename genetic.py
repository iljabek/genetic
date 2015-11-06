#!/usr/bin/env python
"Genetic Algorithm Homebrew = GAH"

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
	#test_IO()
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
			#self._val = gene.randrange_float(self._tupl[0],self._tupl[1],self._tupl[2])
			self._val = gene.randrange_float(*self._tupl)
			#truncate long floats?
		else:
			#self._val = random.uniform(self._tupl[0],self._tupl[1])
			self._val = random.uniform(*self._tupl[:2])
	
	def set(self,val):
		if self._typ == 0:
			if val in self._tupl:
				self._val = val
			else: print "  Gen SET ERROR"
				#self._val = self._tupl[int(val)%len(self._tupl)]
		#elif self._typ == 1: 
			#if val in numpy.arange(self._tupl[0],self._tupl[1],self._tupl[2]).tolist():
				#self._val = val
			#else: print "  Gen SET ERROR"
		else:
			if self._tupl[0]<=val and val<=self._tupl[1]:
				self._val = val
			else: print "  Gen SET ERROR"

	def get(self):
		return self._val
	
	def __str__(self):
		return str(self.get())

class indi(object):
	def __init__(self,genes,vals=[]):
		self._genes   = copy.deepcopy(genes)
		self._fitness = -0.000001
		#self._name    = name
		self._name    = self.hash()
		if len(vals)>0:
			for i in range(len(self._genes)):
				self._genes[i].set(vals[i])

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
		"Hashing using all genes' values, makes inidis uniqe"
		c = ";".join([str(i) for i in self.getall()])
		cm = str( hashlib.md5(  c  ).hexdigest().encode('utf-8') )
#		icm = cm[0:7]
		icm = "{0:09d}".format(int(  cm[0:7]   ,16))
		return icm

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
		self._memory = {}
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
	
	def getFitness(self):
		return [i._fitness for i in self._indis]

	def getFitSumN(self,N):
		return sum([i._fitness for i in self._indis[:N]])

	def getFitSum(self):
		return self.getFitSumN(self.getNindis())
	
	def getNindis(self):
		return len(self._indis)
	
	def getHashes(self):
		return [a.hash() for a in self._indis]
	
	def getUniqueGenes(self):
		u=[]
		for g in range(len(self.getWinner()._genes)):
			u.append(len(list(set([ind._genes[g]._val for ind in self._indis]))))
		return u

	
	def printFitness(self,N):
		print " Fitness dispositions: ",[ '{0:.3g}'.format(ind._fitness) for ind in self._indis[:N]], " Sum:", sum([ ind._fitness for ind in self._indis[:N]])
		#print cntInidis, offspringN, sum(offspringN)

	
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

	def mutateRandomKbyN(self,K,N,W):
		"Select K random individuals and Mutate by N genes"
		WToKeep = W % len(self._indis)
		mutants=[ random.randint(WToKeep,len(self._indis)-1) for pick in range(K) ]
		#print " to be mutated: ", mutants
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

	def mutateSameOrRandomKbyNProtectW(self,K,N,W):
		WToKeep = W % len(self._indis)
		mutants=[]
		allHashes=self.getHashes()
		#get indeces which have earlier accurances
		#print allHashes 
		for i in range(len(allHashes)-1):
			if not i in mutants:
				#print "searching ",i, " ,Sum= ", len(mutants) , " from K= ", K
				j=i+1
				while allHashes[i] in allHashes[j:]:
					#print "   looking at ",j, "at ", allHashes.index(allHashes[i],j),
					j = allHashes.index(allHashes[i],j)
					#print " found ",allHashes[i], " at ",j, " total ", len(mutants)
					mutants.append(j)
					j+=1
					#print "   to check", allHashes[j:]
					#if len(mutants)>=K:
						#break
				#if len(mutants)>=K:
					#break
		print " to be mutated because same: ", len(mutants)
		if len(mutants)<=K:
			"0.. = mutate winner, 1.. = keep winner"
			mutants += [ random.randint(WToKeep,len(self._indis)-1) for pick in range(K-len(mutants)) ]
		print " to be mutated all: ", len(mutants)
		for ind in mutants:
			i = self._indis[ind]
			#print i
			i.mutate(N)
			hashes = [a.hash() for a in self._indis]
			for tries in range(3):
				if i.hash() in hashes:
					i.mutate(N+tries)
			if i.hash() in hashes:
				i.mutateAll()
		#self._num += 1
		#print self.getHashes()

	def crossover(self,W):
		self.mixCrossover(W)

	def mixCrossover(self,W):
		for g in range(len(self.getWinner()._genes)):
			genpool=[[ind._genes[g]._val] for ind in self._indis]
			#print "  Unique "+ind._genes[g]._name+": ",len(list(set([ind._genes[g]._val for ind in self._indis])))
			random.shuffle(genpool)
			random.shuffle(genpool)
			#print genpool
			for ind in range(W,len(self._indis)):
				self._indis[ind]._genes[g].set(genpool[ind][0])

			#print "  Unique Individuals: ", len(list(set(self.getHashes())))

		#self._num += 1

	#def pairedCrossover(self):
		#Npairs = len(self._indis)/2
		#for pair in range(Npairs): #2*pair, 2*pair+1
	
	def evalFitAll(self):
		for i in self._indis:
			self.evalFit(i)
	
	def sortFittest(self):
		winners     = zip(  [i._fitness for i in self._indis]  ,  self._indis  )
		self._indis = list(reversed(zip(*sorted( winners ))[1]))

	def fillMemory(self,ind):
		h=ind.hash()
		if not h in self._memory:
			self._memory[h] = ind._fitness
		#else:
			#pass
			#print "  * found "+h+" in the GenMem!"

	def updateMemory(self):
		for i in self._indis:
			h=i.hash()
			if not h in self._memory:
				self._memory[h] = i._fitness
			#else:
				#print "  * found "+h+" in the GenMem!"

	def getMemory(self,ind):
		h=ind.hash()
		if not h in self._memory:
			return -1
		else:
			return self._memory[h]


	def rankFitness(self,ind, SP):
		return 2.-SP + 2.*(SP-1)* (ind)/(len(self._indis)-1)
	
	def linRepro(self,SP):
		fitS = [rankFitness(i) for i in range(len(self._indis))]
		fitSumN = sum(fitS)


	def getRankProb(self, pos):
		"for Rank Selection + Roulette  Wheel"
		Nall = 1.
#		i0 = len(self._indis)
		N = len(self._indis)
		if pos > N:
			return 0
		else:
#			return 2.*Nall/i0 * (1. - pos/(i0-1))
			return 2.*(N-pos)/N/(N+1) 

	def rouletteWheelSelection(self):
		fitSumN = 1.
		offspring = []
		for i in range(len(self._indis)):
			r = random.uniform(0.,1.)
#			print i, r
			sumRankFit = 0.
			winner = 0
			nTries = 0
			while r > sumRankFit and nTries < 10*len(self._indis):
				winner = random.randint(0,len(self._indis))
				sumRankFit += self.getRankProb(winner)
				nTries += 1
#				print " ", winner, sumRankFit, self.getRankProb(winner),  nTries
			offspring.append(winner)
		print sorted(offspring)
#		self._indis[looser] = indi.mitosis(self._indis[winner])
		newindis = []
		for i in sorted(offspring):
			newindis.append(indi.mitosis(self._indis[i]))
		for i in range(len(self._indis)):
			self._indis[i] = newindis[i]




	def reproNFittest(self,N):
		cntInidis = len(self._indis)
		fitSumN   = self.getFitSumN(N)
		#print fitSumN
		offspringN = []
		for i in self._indis[:N]:
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
		#print " Fitness dispositions: ",[ ind._fitness for ind in self._indis[:N]], 
		print offspringN, sum(offspringN)
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
				
	
	def evalFit(self,ind):
#		print "overload me!"
		fit = sum(ind.getall()[:5])
		ind._fitness = fit

	
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
			newf = gzip.open(path+"/Gen____.tmp", "wb")
		else:
			f = open(fullpath,"rb")
			newf = open(path+"/Gen____.tmp", "wb")
		pickle.dump( Gen , newf )
		for chunk in iter(lambda: f.read(1024), b""):
			newf.write(chunk)
		newf.close()
		f.close()
		os.remove(fullpath)
		os.rename(path+"/Gen____.tmp",fullpath)
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

def test_IO():
	proto = indi( getBeam())
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

def test_world():
	proto = indi( getBeam())
	Gen0 = GEN.clone(proto,100)
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	print Gen0
	print 
	print "Selection, Reproduction"
	print 
	Gen0.rouletteWheelSelection()
	print Gen0
#	Gen0.reproNFittest(3)
	print 
	print "Crossover"
	print 
	Gen0.crossover(3)
	Gen0.evalFitAll()
	Gen0.sortFittest()
	Gen0.levelUp()
	print Gen0
	print 
	print "Mutation"
	print 
	Gen0.mutateRandomKbyN(5,5,0)
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


