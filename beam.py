#!/usr/bin/env python
import genetic
import os
import pickle
import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu)/sig, 2.) / 2.)

class GENBeam(genetic.GEN):  
	@staticmethod
	def evalFit(ind):		
	#Date Scan Chunk Stat PEnergy PowOpt Pow Mat TarDiam TarLen TarDist Cur1 Cur2 TunRad TunLen PosFoc BinN BinMin BinMax Fitness
		fit = 1.
		fit *= gaussian(ind.getall()[0],25,5)  #En
		fit *= gaussian(ind.getall()[1],5,1)  #Pow
		fit *= dict(zip(("myGraphite","myBeryllium","Water","quartz","Tungsten"),[0.1,0.99,0.1,0.2,0.01]))[ind.getall()[2]]    #Mat		
		fit *= gaussian(ind.getall()[3],4,0.5)  #TarDia
		fit *= gaussian(ind.getall()[4],80,10)  #TarLen
		fit *= gaussian(ind.getall()[5],14043,3)  #TarPos
		fit *= gaussian(ind.getall()[6],225,30)  #I1
		fit *= gaussian(ind.getall()[7],225,30)  #I2
		fit *= gaussian(ind.getall()[8],200,50)  #TunRad
		fit *= gaussian(ind.getall()[9],300,100)  #TunLen
		fit *= gaussian(ind.getall()[10],5,1)  #PosFocT
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

def main():
	#pass
	initBeam("./evol1/")
	NGEN=500
	for G in range(0,NGEN):
		iterateBeam("./evol1",G)
	Gen0 = pickle.load( open( "evol1/Gen"+str(NGEN)+".pkl", "rb" ) )
	print Gen0
		

def getBeam():
	beam = []
	#Date Scan Chunk Stat PEnergy PowOpt Pow Mat TarDiam TarLen TarDist Cur1 Cur2 TunRad TunLen PosFoc BinN BinMin BinMax Fitness
	beam.append( genetic.gene("PEnergy", [     1,   100, 0.5]        ) )#0
	beam.append( genetic.gene("Pow",     [   0.2,     5, 0.1]        ) )#1
	beam.append( genetic.gene("Mat",     ("myGraphite","myBeryllium","Water","quartz","Tungsten")) )
	beam.append( genetic.gene("TarDiam", [     1,    10, 0.5]        ) )#5
	beam.append( genetic.gene("TarLen",  [    20,   200,  10]        ) )#4
	beam.append( genetic.gene("TarDist", [ 13963, 14063,  10]        ) )#6
	beam.append( genetic.gene("Cur1",    [   100,   350,  10]        ) )#7
	beam.append( genetic.gene("Cur2",    [   100,   350,  10]        ) )#8
	beam.append( genetic.gene("TunRad",  [    50,   500,  50]        ) )#3
	beam.append( genetic.gene("TunLen",  [    10,  1000,   5]        ) )#2
	beam.append( genetic.gene("PosFoc",  [     1,     9,   1]        ) )#9
	return beam


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
	pickle.dump( Gen0, open( evolpath+"/Gen0.pkl", "wb" ) )

def iterateBeam(evolpath,g):
	Gen0 = pickle.load( open( evolpath+"/Gen"+str(g)+".pkl", "rb" ) )	
	print "**** GEREATION: ",Gen0._num
	print "**********************"
	print 
	print "Selection, Reproduction"
	print 
	Gen0.reproNFittest(10)
	#Gen0.evalFitAll()
	#Gen0.sortFittest()
	#print Gen0
	print 
	print "Crossover"
	print 
	Gen0.crossover()
	print 
	print "Mutation"
	print 
	#Gen0.mutateWorstKbyN(10,2)
	Gen0.mutateRandomKbyN(50,2)
	Gen0.evalFitAll()
	Gen0.sortFittest()
	Gen0.levelUp()
	#print Gen0._indis[0]

	for ind in Gen0._indis:
		writeToOrderList("2206", evolpath+"/Beams.reg", ind)
		
	pickle.dump( Gen0, open( evolpath+"/Gen"+str(g+1)+".pkl", "wb" ) )



if __name__ == '__main__':
	main()
