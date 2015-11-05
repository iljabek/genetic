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
	
"""
(gene = 0..1)
indi  = [g1,g2,g3,g4,g5]
gen   = [i1,i2,i3] 


"""	



	
class indi(object):
	def __init__(self,genes,ingenes=[]):
		self.genes   = copy.deepcopy(genes) # list of (0..1)
		self.fitness = -1.e-12 # bad for integral
		if len(ingenes) == len(genes):
			for i in range(len(self.genes)): 
				self.genes[i] = ingenes[i] #value outside limits
		self.i = 0

	def __getitem__(self, item):
		if isinstance(item, slice):
			return indi(self.genes[item])
		else:
			return self.genes[item]

	def __iter__(self):
		return self

	def next(self):
		if self.i == len(self.genes):
			self.i = 0
			raise StopIteration()
		else:
			self.i += 1 
			return self.genes[self.i-1] 
	
	def __len__(self):
		return len(self.genes)

	@classmethod
	def mitosis(cls,ind):
		return cls(ind.genes)

	def getall(self):
		return self.genes[:]

	def hash(self):
		c = ";".join([str(i) for i in self.getall()])
		cm = str( hashlib.md5(  c  ).hexdigest().encode('utf-8') )
		icm = "{0:09d}".format(int(  cm[0:7]   ,16))

		return icm

		#return hashlib.md5(";".join([str(i) for i in self.getall()])).encode('utf-8')).hexdigest()[0:7]
		#for decimal instead of hex  
#		return "{0:09d}".format(int(str(hashlib.md5(";".join([str(i) for i in self.getall()])).encode('utf-8'))).hexdigest()[0:7],16)

	def mutateI(self,I):
		self.genes[I] = random.uniform(0,1) # index check; binning of input
	
	def mutateAll(self):
		for g in range(len(self.genes)):
			self.mutateI(g)
	

	def mutateAnyN(self,N):
		mutees = []
		for i in range(N):
			mutee = random.randint(0,len(self.genes)-1)
			#don't mutate same gene twice
			if (mutee in mutees):
				tries=0
				while (mutee in mutees) or tries<len(self.genes):
					mutee = random.randint(0,len(self.genes)-1)
					tries+=1
			self.mutateI(mutee)
			mutees.append(mutee)


	def __str__(self):
		OUT="I'm "+self.hash()+"! Fitness: " + str(self.fitness) +  "\n "
		for g in self.genes:
			OUT += str(g) + " " 
		return OUT[:-1]


class GEN(object):
	def __init__(self,indis):
		self._num = 0
		self._indis = []
		self.i = 0
		self._memory = {}
		if isinstance(indis,list):
			for i in indis:
				self._indis.append(copy.deepcopy(i))
		if isinstance(indis,GEN):
			for i in indis:
				self._indis.append(copy.deepcopy(i))
			self._num = indis._num
			self._memory = copy.deepcopy(indis._memory)




	def __getitem__(self, item):
		if isinstance(item, slice):
			return GEN(self._indis[item])
		else:
			return self._indis[item]

	def __iter__(self):
		return self

	def next(self):
		if self.i == len(self._indis):
			self.i = 0
			raise StopIteration()
		else:
			self.i += 1 
			return self._indis[self.i-1] 
	
	def __len__(self):
		return len(self._indis)
		

	@classmethod
	def clone(cls,proto,N):
		"""
			clone from proto-individuum N-times => populate
			use constructor in the end
		"""
		folk = []
		for i in range(N):
			folk.append(indi.mitosis(proto))
		return cls(folk)

	def getWinner(self):
		return self[0]
	
	def getFitness(self):
		"equivalent [i.fitness for i in Gen0] "
		return [i.fitness for i in self]

	def getFitSumN(self,N):
		return sum([i.fitness for i in self[:N]])

	def getFitSum(self):
		return self.getFitSumN(len(self))
	
	
	def getHashes(self):
		return [i.hash() for i in self]
	
	def getUniqueGenes(self):
		u=[]
		for g in range(len(self.getWinner())):
			u.append(len(list(set([ind[g] for ind in self]))))
		return u

	
	def printFitness(self,N):
		subGen = self[:N]
		print " Fitness dispositions: ",[ '{0:.3g}'.format(ind.fitness) for ind in subGen], " Sum:", sum([ ind.fitness for ind in subGen])
		#print cntInidis, offspringN, sum(offspringN)

	
	def __str__(self):
		OUT="GEN Nr."+str(self._num)+ ": "
		", ".join([str(i.fitness) for i in self]) +'\n'
		for i in self:
			OUT += str(i.hash()) +" ("+ str(i.fitness)+" fit) , "
		return OUT

	"Changing Methods"

	def levelUp(self):
		self._num += 1

	def mutateAll(self):
		for i in self:
			i.mutateAll()		
		#self._num += 1

	def mutateAllN(self,N):
		for i in self:
			i.mutateAnyN(N)
#			hashes = [a.hash() for a in self]
#			for tries in range(3):
#				if i.hash() in hashes:
#					i.mutate(N)
#			if i.hash() in hashes:
#				i.mutateAll()
		#self._num += 1

	def mutateClones(self):
		clones2mutate=[] # list of indeces of indis to be mutated
		allHashes=self.getHashes() # list of hashes
		#get indeces which have earlier accurances
		#by going from start, checking all following
		#print allHashes 
		for i in range(len(allHashes)-1):# last one never needs to be checked 
			if not i in clones2mutate: #else it's a registered clone
				j=i+1 #start checking from next
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
		if len(mutants)>=K:
			"0.. = mutate winner, 1.. = keep winner"
			mutants += [ random.randint(WToKeep,len(self._indis)-1) for pick in range(K-len(mutants)) ]
		print " to be mutated all: ", len(mutants)
		for ind in mutants:
			i = self._indis[ind]
			#print i
			i.mutate(N)
			#hashes = [a.hash() for a in self._indis]
			#for tries in range(3):
				#if i.hash() in hashes:
					#i.mutate(N)
			#if i.hash() in hashes:
				#i.mutateAll()
		#self._num += 1
		#print self.getHashes()

	def crossover(self):
		self.mixCrossover()

	def mixCrossover(self):
		for g in range(len(self.getWinner()._genes)):
			genpool=[[ind._genes[g]._val] for ind in self._indis]
			#print "  Unique "+ind._genes[g]._name+": ",len(list(set([ind._genes[g]._val for ind in self._indis])))
			random.shuffle(genpool)
			random.shuffle(genpool)
			#print genpool
			for ind in range(len(self._indis)):
				self._indis[ind]._genes[g].set(genpool[ind][0])

			#print "  Unique Individuals: ", len(list(set(self.getHashes())))

		#self._num += 1

	#def pairedCrossover(self):
		#Npairs = len(self._indis)/2
		#for pair in range(Npairs): #2*pair, 2*pair+1
	
	def evalFitAll(self):
		for i in self:
			self.evalFit(i)
	
	def sortFittest(self):
		winners     = zip(  [i.fitness for i in self]  ,  self  )
		self.i = 0
		sortedindis = list(reversed(zip(*sorted( winners ))[1]))
		for i in range(len(sortedindis)):
			self._indis[i] = indi.mitosis(sortedindis[i])
#		self._indis = list(reversed(zip(*sorted( winners ))[1]))

	def fillMemory(self,ind):
		h=ind.hash()
		if not h in self._memory:
			self._memory[h] = ind.fitness
		#else:
			#pass
			#print "  * found "+h+" in the GenMem!"

	def updateMemory(self):
		for i in self:
			h=i.hash()
			if not h in self._memory:
				self._memory[h] = i.fitness
			#else:
				#print "  * found "+h+" in the GenMem!"

	def getMemory(self,ind):
		h=ind.hash()
		if not h in self._memory:
			return -1
		else:
			return self._memory[h]

		
	def reproNFittest(self,N):
		allN = len(self)
		fitSum   = self.getFitSumN(N)
		#print fitSum
		childN = []
		for i in self[:N]:
			#"calculate fraction of children normed to fitness sum, at least one"
			iWinnerN = max(1, int(1.*allN*(i.fitness/fitSum)))
			childN.append(iWinnerN)
		#print allN, childN, sum(childN)
		#"append if list too short, i.e. there are too few"
		if sum(childN) < allN :
			for c in range( allN-sum(childN) ):
				childN[c%len(childN)] += 1
		#print allN, childN, sum(childN)
		#"reduce if list too long, i.e. there are too many"
		if sum(childN) > allN :
			for c in range( N,N-(sum(childN)-allN),-1 ):
				childN[c%len(childN)] -= 1
		print childN, sum(childN)
		childN = [x-1 for x in childN if x > 0] # start from 0
		Nactual = len(childN)
		looser=Nactual # start with first non-winner
		for winner in range(Nactual):
			#print childN[winner]
			for n in range(childN[winner]):
				#print looser, winner,"," ,
				self[looser] = indi.mitosis(self._indis[winner])
				looser+=1
		#self._num += 1
				
	
	def evalFit(self,ind):
		print "overload me!"
		fit = sum(ind.getall()[:5])
		ind.fitness = fit

	
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



def getBeam():
#	beam = []
#	beam.append( gene("Energy",  [     1,   100, 0.5]        ) )
#	beam.append( gene("Power",   [   0.2,     5, 0.1]        ) )
#	beam.append( gene("TunLen",  [    10,  1000,   5]        ) )
#	beam.append( gene("TunRad",  [    50,   500,  50]        ) )
#	beam.append( gene("TarLen",  [    20,   200,  10]        ) )
#	beam.append( gene("TarRad",  [     1,    10, 0.5]        ) )
#	beam.append( gene("TarPos",  [ 13963, 14063,  10]        ) )
#	beam.append( gene("Cur1",    [   100,   350,  10]        ) )
#	beam.append( gene("Cur2",    [   100,   350,  10]        ) )
#	beam.append( gene("PosFocT", [     1,     9,   1]        ) )
#	beam.append( gene("Mat",     ("myGraphite","myBeryllium","Water","quartz","Tungsten")) )
#	return beam
	return [0,0,0,0,0,0,0,0,0,0]

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
	Gen0 = GEN.clone(proto,10)
	print Gen0
	print Gen0.getUniqueGenes()
	Gen0.mutateAllN(1)
	print Gen0.getUniqueGenes()
	Gen0.mutateAll()
	print Gen0
	print Gen0.getUniqueGenes()
	print "* Slicing"
	print Gen0[0]
	print Gen0[0:2]
	for i in Gen0[3:5]:
		print i
	print Gen0.getWinner()
	print "len=",len(Gen0)
	Gen0.evalFitAll()
#	print Gen0
	Gen0.sortFittest()
#	print Gen0
	print Gen0.getUniqueGenes()
	return 0
	
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
	c = indi([0,0,0,0,0])
	print c
	print "mutateAll"
	c.mutateAll()
	print c
	print "mutateAll"
	c.mutateAll()
	print c
	print "mutateAnyN 1"
	c.mutateAnyN(1)
	print c
	print "mutateAnyN 5"
	c.mutateAnyN(5)
	print c	
	print "mitosis"
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


