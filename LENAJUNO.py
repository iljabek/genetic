#!/usr/bin/env python
import genetic
import sys
import os
import numpy as np


class GENBeam(genetic.GEN):
	"Custom Generation Class: overload Fitness evaluation"

	def evalFitAll(self):
		"submit BeamPar.reg"
		for i in self._indis:
			self.evalFit(i)
	

	def evalFit(self,ind):
		prefit = self.getMemory(ind)
		if prefit > 0: #found individuum in prev. generations!
			print "    Know "+ ind._name +" already!"
			return prefit


def writeToOrderList(filepath, Gen, ind):
	"""
		Usage
		for ind in Gen0._indis:
			writeToOrderList("$HOME/SIMULATIONS/RUN1/Gen_0/BeamPar.reg", ind)
	"""
	#return True # for test only
	#Gen Scan Chunk Stat PEnergy PowOpt Pow Mat TarDiam TarLen TarDist Cur1 Cur2 TunRad TunLen PosFoc 
	indistr=""
	with open(filepath,  "a") as myfile:
		indistr += str(Gen) + '\t'
		indistr += ind.hash() +'\t'
		indistr += "10000" +'\t'
		indistr += str(ind.getall()[0]) +'\t'
		#indistr += "0" +'\t'
		indistr += str(ind.getall()[1]) +'\t'
		indistr += str(ind.getall()[2]) +'\t'
		indistr += str(ind.getall()[3]) +'\t'
		indistr += str(ind.getall()[4]) +'\t'
		indistr += str(ind.getall()[5]) +'\t'
		indistr += str(ind.getall()[6]) +'\t'
		indistr += str(ind.getall()[7]) +'\t'
		indistr += str(ind.getall()[8]) +'\t'
		indistr += str(ind.getall()[9]) +'\t'
		indistr += str(ind.getall()[10]) +'\t'
		#indistr += "100\t0.\t10.\t"
		#indistr += str(ind._fitness)+'\n'
		indistr += '\n'
		myfile.write(indistr)


def getBeam():
	"Genome prototype"
	beam = []
	#Date Scan Chunk Stat PEnergy PowOpt Pow Mat TarDiam TarLen TarDist Cur1 Cur2 TunRad TunLen PosFoc BinN BinMin BinMax Fitness
	beam.append( genetic.gene("PEnergy", [     1,   100, 0.01   ]        ) )#0
	beam.append( genetic.gene("Pow",     [   0.2,     5, 0.0001 ]        ) )#1
	beam.append( genetic.gene("Mat",     ("myGraphite","myBeryllium","Water","quartz","Tungsten")) )
	beam.append( genetic.gene("TarDiam", [     1,    10, 0.001  ]        ) )#5
	beam.append( genetic.gene("TarLen",  [    20,   200, 0.01   ]        ) )#4
	beam.append( genetic.gene("TarDist", [ 13963, 14063, 0.01   ]        ) )#6
	beam.append( genetic.gene("Cur1",    [   100,   350, 0.01   ]        ) )#7
	beam.append( genetic.gene("Cur2",    [   100,   350, 0.01   ]        ) )#8
	beam.append( genetic.gene("TunRad",  [    50,   500, 0.1    ]        ) )#3
	beam.append( genetic.gene("TunLen",  [    10,  1000, 0.1    ]        ) )#2
	beam.append( genetic.gene("PosFoc",  [     1,     9, 0.001  ]        ) )#9
	return beam


def main(argv):
	""" Main Program """
	
	evolpath = argv[0] # .../SIM.../


	NGEN, GenSize, maxFit, GAHargs = 100,100,0.9, (300,100,0.95,  0.5,5,50,2,0)
	"EVOLUTION Procedure"
	print NGEN, GenSize, maxFit, GAHargs

	if not os.path.exists(evolpath+"/GenAll.pkl.gz"):
		print "GAH is starting for the first time:"
		print "Creating the initial population and "
		print "Creating the Gen0 pickle file to proceed..."

		"Initialise Generation: file/directory structure, GEN Class"
		if not os.path.exists(evolpath):
			os.makedirs(evolpath)
	
		proto = genetic.indi( getBeam() ) # Proto-Individual 
		Gen0 = GENBeam.clone(proto,GenSize) # create inital empty population with size GenSize
		Gen0.mutateAll() # shuffle up for all individuals all genes

		if not os.path.exists(evolpath+"/Gen_"+str(Gen0._num)):
			os.makedirs(evolpath+"/Gen_"+str(Gen0._num))

		genetic.writeStateToFile(Gen0, evolpath)

		print "Writing the BeamPar.reg..."
		for ind in Gen0._indis:
			writeToOrderList(evolpath+"/Gen_"+str(Gen0._num)+"/BeamPar.reg" , Gen0._num , ind)
		
		
		print "Submitting condor jobs..."
		bashCommand =  evolpath + "/../SimConGen.sh " 
		bashCommand += evolpath + "/Gen_"+str(Gen0._num)+"/BeamPar.reg" 
		#bashCommand += " > " + evolpath + "/Gen_"+str(Gen0._num)+"/submit.log"
		import subprocess
		print bashCommand
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output = process.communicate()[0]
		open(evolpath + "/Gen_"+str(Gen0._num)+"/submit.log",  "a").write(output+"\n\n\n\n")
		
	#Gen0.evalFitAll() # evaluate all fitnesses
	#Gen0.sortFittest() # sort by fitness
	#Gen0.levelUp() # Gen=1

	#print Gen0	


	return 0

	genFits = []
	f=0.
	try: # until CTRL-C or NMaxGenerations
		for G in range(0,NGEN):
			#iterateBeamAndIO( pathevo, *args) ### Iterate Generation With Write/Read (slower)
			f=iterateBeam( Gen0, *args) ### Iterate Generation in RAM
			genFits.append(f)
			if f>maxFit:
				break
	except KeyboardInterrupt:
		pass
	
	print Gen0.getWinner()
	print f
	#statEvol(pathevo)

	return genFits 
	



#def EVOL

def initBeam(evolpath,GenSize=100):
	"Initialise Generation: file/directory structure, GEN Class"
	if not os.path.exists(evolpath):
		os.makedirs(evolpath)
	
	proto = genetic.indi( getBeam() ) # Proto-Individual 
	Gen0 = GENBeam.clone(proto,GenSize) # create inital empty population with size GenSize
	Gen0.mutateAll() # shuffle up for all individuals all genes
	Gen0.evalFitAll() # evaluate all fitnesses
	Gen0.sortFittest() # sort by fitness
	Gen0.levelUp() # Gen=1

	#print Gen0	
	genetic.writeStateToFile(Gen0, evolpath)
	return Gen0


#def iterateBeamAndIO(evolpath, fitSumShare,NMax, mutants,mutGens,keepW):
def iterateBeamAndIO(evolpath, *args):
	"evolpath = Path for GEN"
	Gen0 = genetic.readStateFromFile(evolpath)
	
	iterateBeam(Gen0, *args)

	#for ind in Gen0._indis:
		#writeToOrderList("2206", evolpath+"/Beams.reg", ind)

	genetic.writeStateToFile(Gen0, evolpath)
	

#def iterateBeam(Gen0, fitSumShare,NMax, mutants,mutGens,keepW):#,g):
def iterateBeam(Gen0, *args):
	"""
	fitSumShare = (0..1) share of the sum of Fitnesses to reproduce
	mutants = (0..NGEN) how many at most to mutate, if all other unequal
	mutGens = (0..nGens) how many genes to mutate at once
	keepW = protect W individuals from mutation 
	"""
	fitSumShare,NMax, mutants,mutGens,keepW = args

	print 
	print 
	print "**** GENERATION: ",Gen0._num ,"*********"
	print "**********************************"
	hashes = Gen0.getHashes()
	print "  Unique Individuals: ", len(list(set(hashes))), " Unique Genes: ", Gen0.getUniqueGenes()

	print " Selection, Reproduction"
	fs = Gen0.getFitness()
	#fitSumShare = 0.5 ###
	"adaptive Reproduction: take indis making x% of the generation fitness" 
	fitSumFrac = sum(fs)*fitSumShare
	N=(i for i,v in enumerate( [sum(fs[:i+1]) for i in range(len(fs)) ] ) if v>fitSumFrac ).next()+1
	"Max n of individuals to keep"
	if N > NMax:
		N = NMax
	#N = max(NMax,N)
	print "  ",N, " individuals combining ", sum(fs[N:])/sum(fs)*100.,"% of the total fitness"
	Gen0.reproNFittest(N)
	hashes = Gen0.getHashes()
	print "  Unique Individuals: ", len(list(set(hashes))), " Unique Genes: ", Gen0.getUniqueGenes()


	if len(list(set(hashes))) < len(hashes): 
		print " Crossover, if not all the same"
		Gen0.crossover()
		hashes = Gen0.getHashes()
		print "  Unique Individuals: ", len(list(set(hashes))), " Unique Genes: ", Gen0.getUniqueGenes()
	#mutants = 50 ###
	#mutGens = 3 ###

	print " Mutation of ", mutants, " by " , mutGens
	#Gen0.mutateWorstKbyN(10,2)
	#Gen0.mutateRandomKbyN(50,3)
	#Gen0.mutateRandomKbyN(mutants,mutGens,keepW)
	Gen0.mutateSameOrRandomKbyNProtectW(mutants,mutGens,keepW)
	hashes = Gen0.getHashes()
	print "  Unique Individuals: ", len(list(set(hashes))), " Unique Genes: ", Gen0.getUniqueGenes()

	print " Evaluate, sort"
	Gen0.evalFitAll()
	Gen0.sortFittest()

	#Gen0.updateMemory()
	
	Gen0.levelUp()
	
	Gen0.printFitness(10)
	
	print 
	return Gen0.getWinner()._fitness 
		
	

if __name__ == '__main__':
	main(sys.argv[1:])
