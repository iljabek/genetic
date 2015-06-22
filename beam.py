#!/usr/bin/env python
import genetic
import os
import pickle
import gzip
import numpy as np


def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu)/sig, 2.) / 2.)

class GENBeam(genetic.GEN):  
	@staticmethod
	def evalFit(ind):		
	#Date Scan Chunk Stat PEnergy PowOpt Pow Mat TarDiam TarLen TarDist Cur1 Cur2 TunRad TunLen PosFoc BinN BinMin BinMax Fitness
		fit = 1.
		fit *= gaussian(ind.getall()[0],25.,5.) + 0.5*gaussian(ind.getall()[0],10.,3.)  #En
		fit *= gaussian(ind.getall()[1],5.,1.)  #Pow
		fit *= dict(zip(("myGraphite","myBeryllium","Water","quartz","Tungsten"),[0.1,0.99,0.1,0.2,0.01]))[ind.getall()[2]]    #Mat		
		fit *= gaussian(ind.getall()[3],4.,0.5)  #TarDia
		fit *= gaussian(ind.getall()[4],80.,10.)  #TarLen
		fit *= gaussian(ind.getall()[5],14043.,3.)  #TarPos
		fit *= gaussian(ind.getall()[6],225.,30.) + 0.5*gaussian(ind.getall()[0],350.,10.)   #I1
		fit *= gaussian(ind.getall()[7],225.,30.)  #I2
		fit *= gaussian(ind.getall()[8],200.,15.) + 0.5*gaussian(ind.getall()[0],300.,3.)   #TunRad
		fit *= gaussian(ind.getall()[9],300.,100.)  #TunLen
		fit *= gaussian(ind.getall()[10],5.,1.)  #PosFocT
		ind._fitness = fit

	def evalFitAll(self):
		for i in self._indis:
			GENBeam.evalFit(i)

def writeToOrderList(date, filename, ind):
	#Date Scan Chunk Stat PEnergy PowOpt Pow Mat TarDiam TarLen TarDist Cur1 Cur2 TunRad TunLen PosFoc BinN BinMin BinMax Fitness
	indistr=""
	with open(filename, "a") as myfile:
		indistr += date + '\t'
		indistr += ind.hash() +'\t'
		indistr += "10000" +'\t'
		indistr += str(ind.getall()[0]) +'\t'
		indistr += "0" +'\t'
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
		indistr += "100\t0.\t10.\t"
		indistr += str(ind._fitness)+'\n'
		myfile.write(indistr)

def getBeam():
	beam = []
	#Date Scan Chunk Stat PEnergy PowOpt Pow Mat TarDiam TarLen TarDist Cur1 Cur2 TunRad TunLen PosFoc BinN BinMin BinMax Fitness
	beam.append( genetic.gene("PEnergy", [     1,   100]        ) )#0
	beam.append( genetic.gene("Pow",     [   0.2,     5]        ) )#1
	beam.append( genetic.gene("Mat",     ("myGraphite","myBeryllium","Water","quartz","Tungsten")) )
	beam.append( genetic.gene("TarDiam", [     1,    10]        ) )#5
	beam.append( genetic.gene("TarLen",  [    20,   200]        ) )#4
	beam.append( genetic.gene("TarDist", [ 13963, 14063]        ) )#6
	beam.append( genetic.gene("Cur1",    [   100,   350]        ) )#7
	beam.append( genetic.gene("Cur2",    [   100,   350]        ) )#8
	beam.append( genetic.gene("TunRad",  [    50,   500]        ) )#3
	beam.append( genetic.gene("TunLen",  [    10,  1000]        ) )#2
	beam.append( genetic.gene("PosFoc",  [     1,     9]        ) )#9
	return beam


def main():
	""" Main Program """
	#pass
	evolpath="./evol3/"
	#EVOL(evolpath)
	statEvol(evolpath)

def statEvol(evolpath):
	gens=[]
	winnerFitnesses=[]
	meanFitnesses=[]
	winnerEnergy=[]
	
	### collect data
	import glob
	#path = "./evol1/*.pkl"
	#for fname in glob.glob(path):
		#print(fname)
		#Gen0 = pickle.load( open( fname, "rb" ) )
	
	#evolpath = "./evol2/"
	infile = gzip.open( evolpath+"/GenAll.pkl.gz", "rb" )
	#Gen0 = pickle.load( gzip.open( evolpath+"/GenAll.pkl.gz", "rb" ) )	
	nGen=0
	try:
		while nGen<10000:
			print(nGen)
			Gen0 = pickle.load( infile )
			nGen += 1
			gens.append(Gen0._num)
			winnerFitnesses.append(Gen0.getWinner()._fitness)
			winnerEnergy.append(Gen0.getWinner().getGene("PEnergy"))
			meanFitnesses.append(1.*Gen0.getFitSum()/Gen0.getNindis())
			print Gen0.getFitSum(), Gen0.getNindis()
	except EOFError :
		print "No more Generations to read" 

		
	print gens, winnerFitnesses
	[dummy, winnerFitnesses] = zip(*sorted(zip(gens[:], winnerFitnesses)))
	[dummy, winnerEnergy] = zip(*sorted(zip(gens[:], winnerEnergy)))
	[dummy, meanFitnesses] = zip(*sorted(zip(gens[:], meanFitnesses)))
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

	arr_x = array("d", gens)
	arr_y = array("d", winnerEnergy)
	tgraph = rt.TGraph(len(arr_x),arr_x,arr_y)
	tgraph.SetTitle("Evolution of the fittest Energy ; Generation # ; Energy in MeV")
	tgraph.Draw("A*")
	canv.SaveAs(evolpath+"/plot_BestFitness_E.png")

	arr_x = array("d", gens)
	arr_y = array("d", meanFitnesses)
	tgraph = rt.TGraph(len(arr_x),arr_x,arr_y)
	tgraph.SetTitle("Evolution of the mean fitness ; Generation # ; fitness")
	tgraph.Draw("A*")
	canv.SaveAs(evolpath+"/plot_MeanFitness.png")



def EVOL(pathevo):
	#pathevo="./evol3/"
	initBeam(pathevo)
	NGEN=3000
	for G in range(0,NGEN):
		iterateBeam(pathevo,G)
	#Gen0 = pickle.load( open( pathevo+"/Gen"+str(NGEN)+".pkl", "rb" ) )
	#Gen0 = pickle.load( open( pathevo+"/GenAll.pkl", "rb" ) )
	Gen0 = pickle.load( gzip.open( pathevo+"/GenAll.pkl.gz", "rb" ) )	
	print Gen0
		

def initBeam(evolpath):
	if not os.path.exists(evolpath):
		os.makedirs(evolpath)
	
	proto = genetic.indi( getBeam())
	#Gen0 = genetic.GEN.clone(proto,100)
	Gen0 = GENBeam.clone(proto,100)
	Gen0.mutateAll()
	Gen0.evalFitAll()
	Gen0.sortFittest()
	Gen0.levelUp()

	for ind in Gen0._indis:
		writeToOrderList("2106", evolpath+"/Beams.reg", ind)
		
	print Gen0	
	#pickle.dump( Gen0, open( evolpath+"/Gen0.pkl", "wb" ) )
	#pickle.dump( Gen0, open( evolpath+"/GenAll.pkl", "wb" ) )
	pickle.dump( Gen0, gzip.open( evolpath+"/GenAll.pkl.gz", "wb" ) )

def iterateBeam(evolpath,g):
	#Gen0 = pickle.load( open( evolpath+"/Gen"+str(g)+".pkl", "rb" ) )	
	#Gen0 = pickle.load( open( evolpath+"/GenAll.pkl", "rb" ) )	
	Gen0 = pickle.load( gzip.open( evolpath+"/GenAll.pkl.gz", "rb" ) )	
	print 
	print 
	print 
	print "**** GEREATION: ",Gen0._num ,"*********"
	print "**********************************"
	print 
	print "Selection, Reproduction"
	print 
	fs = Gen0.getFitness()
	print 
	"adoptive Reproducion: take indis making 50% of the generation fitness" 
	N=(i for i,v in enumerate( [sum(fs[:i+1]) for i in range(len(fs)) ] ) if v>Gen0.getFitSum()*0.5 ).next()
	print N+1
	Gen0.reproNFittest(N+1)
	#Gen0.evalFitAll()
	#Gen0.sortFittest()
	#print Gen0
	print 
	print "Mutation"
	print 
	#Gen0.mutateWorstKbyN(10,2)
	#Gen0.mutateRandomKbyN(50,2)
	Gen0.mutateRandomKbyN(50,3)
	print 
	print "Crossover"
	print 
	Gen0.crossover()
	print 
	print "Evaluate, sort"
	print 
	Gen0.evalFitAll()
	Gen0.sortFittest()
	Gen0.levelUp()
	Gen0.printFitness(N+1)
	#print Gen0._indis[0]

	for ind in Gen0._indis:
		writeToOrderList("2206", evolpath+"/Beams.reg", ind)
		
	#pickle.dump( Gen0, open( evolpath+"/Gen"+str(g+1)+".pkl", "wb" ) )
	#with open(evolpath+"/Gen"+str(g)+".pkl", "rb") as old, open(evolpath+"/Gen"+str(g+1)+".pkl", "wb") as new:
	#with open(evolpath+"/GenAll.pkl", "rb") as old, open(evolpath+"/Gen_tmp.pkl", "wb") as new:
	with gzip.open(evolpath+"/GenAll.pkl.gz", "rb") as old, gzip.open(evolpath+"/Gen_tmp.pkl.gz", "wb") as new:
		#new.write(string)
		pickle.dump( Gen0, new )
		for chunk in iter(lambda: old.read(1024), b""):
			new.write(chunk)
	#os.remove(evolpath+"/GenAll.pkl")
	#os.rename(evolpath+"/Gen_tmp.pkl",evolpath+"/GenAll.pkl")
	os.remove(evolpath+"/GenAll.pkl.gz")
	os.rename(evolpath+"/Gen_tmp.pkl.gz",evolpath+"/GenAll.pkl.gz")
	

if __name__ == '__main__':
	main()
