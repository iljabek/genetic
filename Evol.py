#!/usr/bin/env python
#import genetic
from GEN.gene import gene
from GEN.indi import indi
from GEN import GEN
# GEN, indi, gene
import os
import sys

def gaussian(x, mu, sig):
	import numpy as np
	gaus = np.exp(-np.power((float(x) - float(mu))/float(sig), 2.) / 2.) 
	gaus /= ( 2. * np.pi * np.sqrt(float(sig)) )
	return gaus 

#class GENBeam(GEN.GEN):
#	"Custom Generation Class: overload Fitness evaluation"
def evalFitOverload(ind):
	"""
		Custom overloaded Fitness evaluation
		takes one indi, returns fitness
	"""
	fit = 1.
	fit *= gaussian(ind.getall()[0],0.3,0.03)  #Pow
	fit *= gaussian(ind.getall()[1],0.3,0.03)  #Pow
	fit *= gaussian(ind.getall()[2],0.3,0.03)  #Pow
	fit *= gaussian(ind.getall()[3],0.3,0.03)  #Pow
	fit *= gaussian(ind.getall()[4],0.3,0.03)  #Pow
	fit *= gaussian(ind.getall()[5],0.3,0.03)  #Pow
	ind.fitness = fit


def getBeam():
	"Genome prototype"
	beam = []
	beam.append( gene(0) )
	beam.append( gene(0) )
	beam.append( gene(0) )
	beam.append( gene(0) )
	beam.append( gene(0) )
	beam.append( gene(0) )
	return beam


def main(argv):
	""" Main Program """
	#pass
	evolpath=argv[0]

	params = [] # genetic parameter
	params.append(300)    #0  Max Gens
	params.append(100)    #1  Gen size = indis in Gen
	params.append(0.99)   #2  final max fitness
	
	params.append(0.5)    #3  MutateGeneProb
	params.append(0.2)    #4  MutateIndiProb
	params.append(0.5)    #5  WinToReporduceFrac
	params.append(0.015)    #6  ElitistFrac
	params.append(0)    #7  WeightMode
	params.append(0.33)    #8  GeneCousins

	print params

	### Init Generation, return GEN and Write both 
	iGen = initBeam( 
			evolpath = evolpath, 
			GenSize  = params[1] 
			) 
#	print iGen
	print "Unique Indis: ", iGen.getUniqueIndis() , 
	print ", Unique Genes: ", iGen.getUniqueGenes()
	iGen.MutateGeneProb = params[3]      #float 0..1
	iGen.MutateIndiProb = params[4]      #float 0..1
	iGen.WinToReporduceFrac = params[5] #float 0..1
	iGen.ElitistFrac = params[6]    #int 0..len(self)
	# Weights: 0=fitness, 1=uniform, 2=linear, (3=exp)
	iGen.WeightMode = params[7] 
	iGen.GeneCousins = params[8] 
	iGen.checkGAParameters()

	Ntrials = params[0]
	trials=[]
	try: # until CTRL-C or NMaxGenerations
		for t in range(Ntrials):
			pass
			print 
			print "**** GENERATION: ",iGen._num ,"*********"
			print "**********************************"
			print "  Unique Individuals: ", iGen.getUniqueIndis(), " Unique Genes: ", iGen.getUniqueGenes()

			print " Evaluate, sort"
			iGen.evalFitAll()
			iGen.sortFittest()

			#iGen.updateMemory()
			
			
			iGen.printChroms(10)
			iGen.printFitness(10)
#			iGen.show_SelectionProb()	
			print " Selection, Reproduction"
#			iGen.Selection_RouletteWheel()
			iGen.Selection_NFittest()

			if iGen.getUniqueIndis() < len(iGen): 
				print " Crossover, if not all the same"
				iGen.crossover()
				hashes = iGen.getHashes()
				print "  Unique Individuals: ", iGen.getUniqueIndis(), " Unique Genes: ", iGen.getUniqueGenes()

			print " Mutation"
			
			iGen.mutateClones()
			print "  Unique Individuals: ", iGen.getUniqueIndis(), " Unique Genes: ", iGen.getUniqueGenes()

#			iGen.mutateCousins()
#			print "  Unique Individuals: ", iGen.getUniqueIndis(), " Unique Genes: ", iGen.getUniqueGenes()

			iGen.mutateRandom()
			print "  Unique Individuals: ", iGen.getUniqueIndis(), " Unique Genes: ", iGen.getUniqueGenes()

			iGen.perturbRandom()
			print "  Unique Individuals: ", iGen.getUniqueIndis(), " Unique Genes: ", iGen.getUniqueGenes()

			iGen.levelUp()
			trials.append( iGen[0].mitosis() )
#			open(evolpath+"/params.txt","a").write(" ".join(map(str,trials[t]))+"\n")
#			GEN.writeStateToFile(iGen, evolpath, joined=False,gziped=False)
#			GEN.writeStateToFile(iGen, evolpath, joined=True,gziped=True)
	except KeyboardInterrupt:
		pass

	print [ f.chrom for f in trials]


	
	print 
	return 0


	
	

def initBeam(evolpath,GenSize=100):
	"Initialise Generation: file/directory structure, GEN Class"
	if not os.path.exists(evolpath):
		os.makedirs(evolpath)
	
	proto = GEN.indi( getBeam() ) # Proto-Individual 
#	Gen0 = GENBeam.clone(proto,GenSize) # create inital empty population with size GenSize
	Gen0 = GEN.GEN.clone(proto,GenSize) # create inital empty population with size GenSize
	Gen0.evalFit = evalFitOverload
	Gen0.mutateAll() # shuffle up for all individuals all genes
	Gen0.evalFitAll() # evaluate all fitnesses
	Gen0.sortFittest() # sort by fitness
	Gen0.levelUp() # Gen=1

	print "* Gen0 initialized"
#	print Gen0	
#	GEN.writeStateToFile(Gen0, evolpath, joined=False,gziped=False)
	GEN.writeStateToFile(Gen0, evolpath, joined=True,gziped=True)
	return Gen0









def iterateBeamAndIO(evolpath, *args):
	"evolpath = Path for GEN"
	Gen0 = GEN.readStateFromFile(evolpath, joined=False,gziped=False)
	
	iterateBeam(Gen0, *args)

	#for ind in Gen0._indis:
		#writeToOrderList("2206", evolpath+"/Beams.reg", ind)

	GEN.writeStateToFile(Gen0, evolpath, joined=False,gziped=False)
	

def iterateBeam(Gen0, *args):
	"""
	fitSumShare = (0..1) share of the sum of Fitnesses to reproduce
	mutants = (0..NGEN) how many at most to mutate, if all other unequal
	mutGens = (0..nGens) how many genes to mutate at once
	keepW = protect W individuals from mutation 
	"""
	fitSumShare,NMax, mutants,mutGens,keepW = args


if __name__ == '__main__':
	main(sys.argv[1:])
