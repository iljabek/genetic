#!/usr/bin/env python
import genetic
import os
import numpy as np


def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu)/sig, 2.) / 2.)

class GENBeam(genetic.GEN):
	"Custom Generation Class: overload Fitness evaluation"
	#@staticmethod
	def evalFit(self,ind):
		prefit = self.getMemory(ind)
		if prefit > 0: #found individual in prev. generations!
			print "    Know "+ ind._name +" already!"
			return prefit
			
	#Date Scan Chunk Stat PEnergy PowOpt Pow Mat TarDiam TarLen TarDist Cur1 Cur2 TunRad TunLen PosFoc BinN BinMin BinMax Fitness
		fit = 1.
		fit *= gaussian(ind.getall()[0],25.,5.) #+ 1.*gaussian(ind.getall()[0],10.,3.)  #En
		fit *= gaussian(ind.getall()[1],5.,1.)  #Pow
		fit *= dict(zip(("myGraphite","myBeryllium","Water","quartz","Tungsten"),[0.1,0.99,0.1,0.2,0.01]))[ind.getall()[2]]    #Mat		
		fit *= gaussian(ind.getall()[3],4.,0.5)  #TarDia
		fit *= gaussian(ind.getall()[4],80.,10.)  #TarLen
		fit *= gaussian(ind.getall()[5],14043.,3.)  #TarPos
		fit *= gaussian(ind.getall()[6],225.,30.) #+ 0.5*gaussian(ind.getall()[0],350.,10.)   #I1
		fit *= gaussian(ind.getall()[7],225.,30.)  #I2
		fit *= gaussian(ind.getall()[8],200.,15.) #+ 1.*gaussian(ind.getall()[0],300.,3.)   #TunRad
		fit *= gaussian(ind.getall()[9],300.,100.)  #TunLen
		fit *= gaussian(ind.getall()[10],5.,1.)  #PosFocT
		ind._fitness = fit


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


def main():
	""" Main Program """
	#pass
	evolpath="./evol8/"
	#def EVOL(pathevo, NGEN, GenSize, fitMax, *args):
	#EVOL(evolpath,100,100,0.5,50,3,0)
	#EVOL(evolpath,100,100,0.5,50,1,1)
	#EVOL(evolpath,100,100,0.5,50,2,0)
	#EVOL(evolpath,100,100,0.9,50,5,0)
	#args = fitSumShare,NMax, mutants, mutGens,keepW
	#EVOL(evolpath,*args)
	
	evolpath="./trialA"
	args = (300,100,0.95,  0.5,5,50,2,0)
	print args
	open(evolpath+"/params.txt","a").write("#NGEN GenSize fitSumShare NMax mutants mutGens keepW \n")
	open(evolpath+"/params.txt","a").write("#Parameters: "+" ".join(map(str,args))+"\n")
	open(evolpath+"/params.txt","a").write("#Fittest evolution: "+"\n")
	trials=[0]*100
	try: # until CTRL-C or NMaxGenerations
		for t in range(len(trials)):
			pass
			trials[t] = EVOL(evolpath,*args)[:]
			open(evolpath+"/params.txt","a").write(" ".join(map(str,trials[t]))+"\n")
	except KeyboardInterrupt:
		pass

	print trials
	#for l in trials:
		#open(evolpath+"/params.txt","a").write(" ".join(map(str,l))+"\n")
	#statEvol(evolpath)


def statEvol(evolpath):
	"Statistical Evaluation of Evolution"
	"Expemplary implementation for OneFile+Gzip Evolutions"
	import glob
	import pickle
	import gzip
	gens=[]
	winnerFitnesses=[]
	meanFitnesses=[]
	winnerEnergy=[]
	
	winnerGenes=[]
	for g in getBeam(): 
		if g._typ != 0 : #if g._typ != 0:
			winnerGenes.append([g._name,[]])
	
	allGenes=[] # allGenes[gen][ind][gene]=val
	#allGenes.append([])
	
	infile = gzip.open( evolpath+"/GenAll.pkl.gz", "rb" )
	Gen0 = pickle.load( infile )
	infile.seek(0)
	
	nGen = Gen0._num
	iGen = 0
	try:
		while iGen<nGen:
			print iGen, 
			Gen0 = pickle.load( infile )
			iGen += 1
			gens.append(Gen0._num)
			winnerFitnesses.append(Gen0.getWinner()._fitness)
			winnerEnergy.append(Gen0.getWinner().getGene("PEnergy"))
			meanFitnesses.append(1.*Gen0.getFitSum()/Gen0.getNindis())
			for g in range(len(winnerGenes)):
				winnerGenes[g][1].append(Gen0.getWinner().getGene(winnerGenes[g][0]))
			
			#print allGenes
			#allGenes[iGen-1].append([])
			#for i in range(len(Gen0._indis)):
				#print allGenes
				#allGenes[iGen-1][i].append([])
				#print
				#for g in range(len(winnerGenes)):
					#allGenes[iGen-1][i].append([])
					#allGenes[iGen-1][i][g] = Gen0._indis[i].getGene(winnerGenes[g][0])
					#print Gen0._indis[i].getGene(winnerGenes[g][0]),
			#allGenes.append([[] for i in range(len(Gen0._indis))])
			#for i in range(len(Gen0._indis)):
				#allGenes[iGen-1][i]=[[] for i in range(len(winnerGenes))]
				#for g in range(len(winnerGenes)):
					#allGenes[iGen-1][i][g]=Gen0._indis[i].getGene(winnerGenes[g][0])
			#print allGenes
			
			print Gen0.getFitSum(), Gen0.getNindis() #, winnerGenes[0][0], winnerGenes[0][1]
	except EOFError :
		print "No more Generations to read" 

		
	print gens, winnerFitnesses
	[dummy, winnerFitnesses] = zip(*sorted(zip(gens[:], winnerFitnesses)))
	[dummy, winnerEnergy] = zip(*sorted(zip(gens[:], winnerEnergy)))
	[dummy, meanFitnesses] = zip(*sorted(zip(gens[:], meanFitnesses)))
	for g in range(len(winnerGenes)):
		[dummy, winnerGenes[g][1]] = zip(*sorted(zip(gens[:], winnerGenes[g][1])))
	gens.sort()
	print gens, winnerFitnesses
	
	
	### ROOT Plotting
	import ROOT as rt
	from array import array
	rt.gROOT.SetBatch(1)
	canv = rt.TCanvas('c1','',0,0,700,400)
	arr_x = array("d", gens)
	arr_y = array("d", winnerFitnesses)
	#arr_xerr = array("d", RunNrsErr)
	#arr_yerr = array("d", RatesErr)
	#print arr_x, arr_y, len(arr_x), len(arr_y)
	#tgraph = rt.TGraphErrors(len(arr_x),arr_x,arr_y,arr_xerr,arr_yerr)
	tgraph = rt.TGraph(len(arr_x),arr_x,arr_y)
	tgraph.SetTitle("Evolution of the fittest ; Generation # ; fitness")
	tgraph.Draw("A*")
	#arDef = rt.TArrow(ThreshEff[1],200,ThreshEff[1],250,0.02,"|>")
	#arDef.SetAngle(15)
	#arDef.SetLineWidth(2)
	#arDef.Draw()
	canv.SaveAs(evolpath+"/plot_BestFitness.png")
	canv.SaveAs(evolpath+"/plot_BestFitness.root")

	#arr_y = array("d", winnerEnergy)
	#tgraph = rt.TGraph(len(arr_x),arr_x,arr_y)
	#tgraph.SetTitle("Evolution of the fittest Energy ; Generation # ; Energy in MeV")
	#tgraph.Draw("A*")
	#canv.SaveAs(evolpath+"/plot_BestFitness_E.png")

	for g in range(len(winnerGenes)):
		arr_y = array("d", winnerGenes[g][1])
		tgraph = rt.TGraph(len(arr_x),arr_x,arr_y)
		tgraph.SetTitle("Evolution of the fittest " + winnerGenes[g][0]+ "; Generation # ; " + winnerGenes[g][0])
		tgraph.Draw("A*")
		canv.SaveAs(evolpath+"/plot_BestFitness_"+str(g)+".png")


	arr_y = array("d", meanFitnesses)
	tgraph = rt.TGraph(len(arr_x),arr_x,arr_y)
	tgraph.SetTitle("Evolution of the mean fitness ; Generation # ; fitness")
	tgraph.Draw("A*")
	canv.SaveAs(evolpath+"/plot_MeanFitness.png")


def EVOL(pathevo, NGEN, GenSize, maxFit, *args):
	"EVOLUTION Procedure"
	
	Gen0 = initBeam(pathevo,GenSize) ### Init Generation, return GEN and Write both 
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
	main()
