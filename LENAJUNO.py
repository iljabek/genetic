#!/usr/bin/env python
# -*- coding: utf-8 -*-
import genetic
import sys
import os
import numpy as np


class GENBeam(genetic.GEN):
	"Custom Generation Class: overload Fitness evaluation"

def submitGen0ToCondor(Gen0, evolpath):
		import subprocess
		print "Submitting condor jobs..."
		BeamParPath = evolpath + "/Gen_"+str(Gen0._num)+"/BeamPar.reg"
		RunName = evolpath.split("/")[-2]
		bashCommand =  "/home/home2/institut_3b/meloni/SIMULATIONS/SimConGen.sh " 
		bashCommand += BeamParPath + " "  # $1
		bashCommand += evolpath + " "  # $2
		bashCommand += RunName  # $3
		#bashCommand += " > " + evolpath + "/Gen_"+str(Gen0._num)+"/submit.log"
		print bashCommand
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		for c in iter(lambda: process.stdout.read(1), ''):
			sys.stdout.write(c)
		#output = process.communicate()[0]
		#open(evolpath + "/Gen_"+str(Gen0._num)+"/submit.log",  "a").write(output+"\n\n\n\n")
		
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


def CalculateFitness(Gen0, evolpath, ind):
		import subprocess
		#print "Calculating fitness..."
#		fitpath= "/home/home2/institut_3b/meloni/SIMULATIONS/"
		fitpath= "/".join(evolpath.split("/")[:-2])
		bashCommand="root -l -q -b "
		bashCommand += fitpath + '/fitness_calculator.C("'
		bashCommand += evolpath.split("/")[-2]+'","'+str(Gen0._num)+'","'+ ind.hash()+'")'
		print bashCommand
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		for c in iter(lambda: process.stdout.read(1), ''):
			sys.stdout.write(c)
		#output = process.communicate()[0]
		#open(evolpath + "/Gen_"+str(Gen0._num)+"/submit.log",  "a").write(output+"\n\n\n\n")

def getParList(Gen, ind):
	indistr = str(Gen) + '\t'                 #Generation 
	indistr += ind.hash() +'\t'               #Scan
	indistr += "30000" +'\t'                  #Statistics
	indistr += str(ind.getall()[0]) +'\t'     #Proton Energy
		#indistr += "0" +'\t'
	indistr += str(ind.getall()[1]) +'\t'     #Power
	indistr += str(ind.getall()[2]) +'\t'     #Target Material
	indistr += str(ind.getall()[3]) +'\t'     #Target Diameter
	indistr += str(ind.getall()[4]) +'\t'     #Target Length
	indistr += str(ind.getall()[5]) +'\t'     #Target Position
	indistr += str(ind.getall()[6]) +'\t'     #Current Horn 1
	indistr += str(ind.getall()[7]) +'\t'     #Current Horn 2
	indistr += str(ind.getall()[8]) +'\t'     #Tunnel Radius
	indistr += str(ind.getall()[9]) +'\t'     #Tunnel Length
	indistr += str(ind.getall()[10]) +'\t'    #Positive Focussing Time
	return indistr


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
		indistr = getParList(Gen, ind)
		#indistr += "100\t0.\t10.\t"
		#indistr += str(ind._fitness)+'\t'
		indistr += '\n'
		myfile.write(indistr)


def writeToOverviewList(filepath, Gen, ind):
	#return True # for test only
	#Gen Scan Stat PEnergy Pow Mat TarDiam TarLen TarDist Cur1 Cur2 TunRad TunLen PosFoc Fitness 
	indistr=""
	with open(filepath,  "a") as myfile:
		indistr = getParList(Gen, ind)
		#indistr += "100\t0.\t10.\t"
		indistr += str(ind._fitness)+'\t'
		indistr += '\n'
		myfile.write(indistr)


def getBeam():
	"Genome prototype"
	beam = []
	#Date Scan Chunk Stat PEnergy PowOpt Pow Mat TarDiam TarLen TarDist Cur1 Cur2 TunRad TunLen PosFoc BinN BinMin BinMax Fitness
	beam.append( genetic.gene("PEnergy", [     1,   100,  1.0    ]        ) )#0 GeV
	beam.append( genetic.gene("Pow",     [   0.2,     5,  0.2    ]        ) )#1 MW
	beam.append( genetic.gene("Mat",     ("myGraphite","myBeryllium","Water","quartz","Tungsten")) )
	beam.append( genetic.gene("TarDiam", [     1,    10,  0.5    ]        ) )#5 mm
	beam.append( genetic.gene("TarLen",  [    20,   200,  1.0    ]        ) )#4 cm
	beam.append( genetic.gene("TarDist", [ 13963, 14063,  1.0    ]        ) )#6 cm
	beam.append( genetic.gene("Cur1",    [ 100e3, 350e3, 10.e3   ]        ) )#7 A
	beam.append( genetic.gene("Cur2",    [ 100e3, 350e3, 10.e3   ]        ) )#8 A
	beam.append( genetic.gene("TunRad",  [    50,   500, 10.0    ]        ) )#3 cm
	beam.append( genetic.gene("TunLen",  [    10,   500, 10.0    ]        ) )#2 m
	beam.append( genetic.gene("PosFoc",  [     1,     9,  1      ]        ) )#9 y
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

	NGEN , GenSize , maxFit = 100 , 100 , 0.7
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
		Gen0 = GENBeam.clone(proto,GenSize) # create initial empty population with size GenSize
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
		for l in open(fitfile,"r"):
			if l[0] == "#": continue
			if len(l.split('\t')) != 5: continue
			nlines += 1

			ind = l.split('\t')[0]
			indfit = 0
			indfit = float(l.split('\t')[3]) # DEFINE FITNESS
			
			if ind in Gen0.getHashes():				
				print "Found " + ind+ " ! Fitness: ", indfit  
				ninds += 1
				#find the index in the list, set fitness				
				Gen0._indis[ Gen0.getHashes().index(ind) ]._fitness = indfit

		print "Read ",nlines," , found ",ninds, " matching individuals"
		
		if ninds < 0.90*GenSize:
			print "Not enough individuals evaluated or wrong hashes in the fitness file! check " + fitfile
			return 2
		Gen0.sortFittest()

		print "Writing Generation overview"
		for ind in Gen0._indis:
			writeToOverviewList(evolpath+"/Gen_"+str(Gen0._num)+"/GenOverview.txt" , Gen0._num , ind)


		print "Individuals' fitnesses"
		Gen0.printFitness(len(Gen0._indis))

		print "**** GENERATION: ",Gen0._num ," Done *********"
		print "**********************************"
		hashes = Gen0.getHashes()
		print "  Unique Individuals: ", len(list(set(hashes))), " Unique Genes: ", Gen0.getUniqueGenes()
		

		Gen0.updateMemory()

		if Gen0._indis[0]._fitness > maxFit:
			print "FINISHING! Found individual with fitness ", Gen0._indis[0]._fitness
			print "Go to: " + evolpath+"/Gen_"+Gstr+"/"+Gen0._indis[0].hash()
			return 0

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
			Gen0.crossover(keepW)
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
		
		
		if os.path.exists(evolpath+"/Gen_"+str(Gen0._num)+"/BeamPar.reg"):
			print "Danger! Maybe submission process already running?"
			return 3
			
		for ind in Gen0._indis:
			writeToOrderList(evolpath+"/Gen_"+str(Gen0._num)+"/BeamPar.reg" , Gen0._num , ind)
		fakesubmitGen0ToCondor(Gen0, evolpath)


		#Gen0.printFitness(10)

	return 0


if __name__ == '__main__':
	main(sys.argv[1:])
