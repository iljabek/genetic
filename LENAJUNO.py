#!/usr/bin/env python
import genetic
import sys
import os
import numpy as np


class GENBeam(genetic.GEN):
	"Custom Generation Class: overload Fitness evaluation"

def submitGen0ToCondor(Gen0, evolpath):
		import subprocess
		print "Submitting condor jobs..."
		bashCommand =  evolpath + "/../SimConGen.sh " 
		bashCommand += evolpath + "/Gen_"+str(Gen0._num)+"/BeamPar.reg" 
		#bashCommand += " > " + evolpath + "/Gen_"+str(Gen0._num)+"/submit.log"
		print bashCommand
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output = process.communicate()[0]
		open(evolpath + "/Gen_"+str(Gen0._num)+"/submit.log",  "a").write(output+"\n\n\n\n")
		
def fakesubmitGen0ToCondor(Gen0, evolpath):
	"""
		"Gen_0/GenFitList_0.txt"
		"#Hash 2sig 3sig max mean"
	"""
	import random
	
	filepath = evolpath + "/Gen_"+str(Gen0._num)+"/GenFitList_"+str(Gen0._num)+".txt"
	indistr=""
	cnt=0
	for ind in Gen0._indis:
		with open(filepath,  "a") as myfile:
			indistr = ind.hash() +'\t'
			indistr += str(random.randint(0,999)) +'\t'
			indistr += str(random.randint(0,999)) +'\t'
			indistr += str(random.randint(0,999)) +'\t'
			indistr += str(random.randint(0,999)) +'\t'
			indistr += '\n'
			myfile.write(indistr)
			cnt += 1
	print "  Written ",cnt, " entries"



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
	beam.append( genetic.gene("PEnergy", [     1,   100, 0.5    ]        ) )#0
	beam.append( genetic.gene("Pow",     [   0.2,     5, 0.0001 ]        ) )#1
	beam.append( genetic.gene("Mat",     ("myGraphite","myBeryllium","Water","quartz","Tungsten")) )
	beam.append( genetic.gene("TarDiam", [     1,    10, 0.001  ]        ) )#5
	beam.append( genetic.gene("TarLen",  [    20,   200, 0.01   ]        ) )#4
	beam.append( genetic.gene("TarDist", [ 13963, 14063, 0.01   ]        ) )#6
	beam.append( genetic.gene("Cur1",    [   100,   350, 0.01   ]        ) )#7
	beam.append( genetic.gene("Cur2",    [   100,   350, 0.01   ]        ) )#8
	beam.append( genetic.gene("TunRad",  [    50,   500, 0.1    ]        ) )#3
	beam.append( genetic.gene("TunLen",  [    10,  1000, 0.1    ]        ) )#2
	beam.append( genetic.gene("PosFoc",  [     1,     9, 1      ]        ) )#9
	return beam


def main(argv):
	""" Main Program """
	
	# scope-privileged varables
	
	evolpath = argv[0] # .../SIM.../

	"""
	NGEN = max num of generations to stop
	maxFit = max fitness to reach to stop
	GenSize = num of individuals per generation

	fitSumShare = (0..1) share of the sum of Fitnesses to reproduce
	mutants = (0..NGEN) how many at most to mutate, if all other unequal
	mutGens = (0..nGens) how many genes to mutate at once
	keepW = protect W individuals from mutation 
	"""

	NGEN , GenSize , maxFit = 100 , 100 , 0.9
	fitSumShare , NMax , mutants , mutGens , keepW =  0.5 , 5 , 50 , 2 , 0
	"EVOLUTION Procedure"
	print NGEN, GenSize, maxFit, fitSumShare,NMax, mutants,mutGens,keepW

	Gen0 = None #current Generation pointer
	
	if not os.path.exists(evolpath+"/GenAll.pkl.gz"): # can I recover?
		print "GAH is starting for the first time:"
		print "Creating the initial population and "
		print "Creating the Gen0 pickle file to proceed..."

		"Initialise Generation: file/directory structure, GEN Class"
		if not os.path.exists(evolpath):
			os.makedirs(evolpath)
	
		proto = genetic.indi( getBeam() ) # Proto-Individual
		Gen0 = GENBeam.clone(proto,GenSize) # create inital empty population with size GenSize
		Gen0.mutateAll() # shuffle up for all individuals all genes
		
		#Example of inserting an individual
		alien = genetic.indi( getBeam() ,[25.,5.,"myBeryllium",4.,80.,14043.,225.,225.,200.,300.,5])
		Gen0._indis[0] = alien

		if not os.path.exists(evolpath+"/Gen_"+str(Gen0._num)):
			os.makedirs(evolpath+"/Gen_"+str(Gen0._num))
		
		"Save pickle of Gen0, create register list, submit Condor"
		genetic.writeStateToFile(Gen0, evolpath)
		"Initialise Generation: file/directory structure, GEN Class"
		if not os.path.exists(evolpath+"/Gen_"+str(Gen0._num)):
			os.makedirs(evolpath+"/Gen_"+str(Gen0._num))
		print "Writing the BeamPar.reg..."
		for ind in Gen0._indis:
			writeToOrderList(evolpath+"/Gen_"+str(Gen0._num)+"/BeamPar.reg" , Gen0._num , ind)
		fakesubmitGen0ToCondor(Gen0, evolpath)
		
	else: #found pickle, can recover
		# recover -> if (fitness file not comlete) stop |else| evolve -> submit new generation -> hibernate 
		
		"Gen_0/GenFitList_0.txt"
		"#Hash 2sig 3sig max mean"
		print "Recovering previous state..."
		Gen0 = genetic.readStateFromFile(evolpath)		
		Gstr = str(Gen0._num)
		print "Found Gen " + Gstr
		
		print "Individuals"
		for ind in Gen0._indis:
			print ind.hash()
		
		fitfile = evolpath+"/Gen_"+Gstr+"/GenFitList_"+Gstr+".txt"
		
		if not os.path.exists(fitfile):
			print "Cannot find fitness file! check " + fitfile
			return 1
			
		print "Found fitness file "+evolpath+"/Gen_"+Gstr+"/GenFitList_"+Gstr+".txt"
		print "Reading individuals..."
		
		nlines = 0
		ninds = 0
		for l in open(fitfile):
			if l[0] == "#": continue
			nlines += 1

			ind = l.split('\t')[0]
			indfit = float(l.split('\t')[3]) # DEFINE FITNESS
			
			if ind[0] in Gen0.getHashes():				
				print "Found " + ind[0]+ " ! Fitness: ", indfit  
				ninds += 1
				#find the index in the list, set fitness				
				Gen0._indis[ Gen0.getHashes().index(ind[0]) ]._fitness = indfit

		print "Read ",nlines," , found ",ninds, " matching individuals"
		
		print "Individuals' fitnesses"
		Gen0.printFitness(len(Gen0._indis))

		print "**** GENERATION: ",Gen0._num ,"*********"
		print "**********************************"
		hashes = Gen0.getHashes()
		print "  Unique Individuals: ", len(list(set(hashes))), " Unique Genes: ", Gen0.getUniqueGenes()
		
		Gen0.sortFittest()

		Gen0.updateMemory()


		print " Selection, Reproduction"
		fs = Gen0.getFitness()

		"adaptive Reproduction: take indis making x% of the generation fitness" 
		fitSumFrac = sum(fs)*fitSumShare
		N=(i for i,v in enumerate( [sum(fs[:i+1]) for i in range(len(fs)) ] ) if v>fitSumFrac ).next()+1
		"Max n of individuals to keep"
		N = min(NMax,N)
		print "  ",N, " individuals combining ", sum(fs[N:])/sum(fs)*100.,"% of the total fitness"
		Gen0.reproNFittest(N)

		Gen0.levelUp() # it's a new generation  


		hashes = Gen0.getHashes()
		print "  Unique Individuals: ", len(list(set(hashes))), " Unique Genes: ", Gen0.getUniqueGenes()


		if len(list(set(hashes))) < len(hashes): 
			print " Crossover, if not all the same"
			Gen0.crossover()
			hashes = Gen0.getHashes()
			print "  Unique Individuals: ", len(list(set(hashes))), " Unique Genes: ", Gen0.getUniqueGenes()


		print " Mutation of ", mutants, " by " , mutGens
		Gen0.mutateSameOrRandomKbyNProtectW(mutants,mutGens,keepW)
		hashes = Gen0.getHashes()
		print "  Unique Individuals: ", len(list(set(hashes))), " Unique Genes: ", Gen0.getUniqueGenes()

		print " Evaluate"
		#/////		Gen0.evalFitAll()
		
		"Save pickle of Gen, create register list, submit Condor"		
		genetic.writeStateToFile(Gen0, evolpath)
		"Initialise Generation: file/directory structure, GEN Class"
		if not os.path.exists(evolpath+"/Gen_"+str(Gen0._num)):
			os.makedirs(evolpath+"/Gen_"+str(Gen0._num))
		print "Writing the BeamPar.reg..."
		for ind in Gen0._indis:
			writeToOrderList(evolpath+"/Gen_"+str(Gen0._num)+"/BeamPar.reg" , Gen0._num , ind)
		fakesubmitGen0ToCondor(Gen0, evolpath)


		#Gen0.printFitness(10)

	return 0


if __name__ == '__main__':
	main(sys.argv[1:])
