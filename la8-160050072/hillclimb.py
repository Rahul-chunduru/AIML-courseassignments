import sys
import math
from random import shuffle
from draw import drawTour
import graph_plot
from collections import defaultdict
import argparse
import random
import itertools
import pdb
from random import choice

#########################################################################################################
################# Provided code #########################################################################
#########################################################################################################
cities = 0
nodeDict = {}
numberOfRuns = 5

class Stack:
	"A container with a last-in-first-out (LIFO) queuing policy."
	def __init__(self):
		self.list = []

	def push(self,item):
		"Push 'item' onto the stack"
		self.list.append(item)

	def pop(self):
		"Pop the most recently pushed item from the stack"
		return self.list.pop()

	def isEmpty(self):
		"Returns true if the stack is empty"
		return len(self.list) == 0

class Node():
	def __init__(self,index, xc, yc):
		self.i = index
		self.x = xc
		self.y = yc


def generateFile(cities, seed):
	MIN = 0
	MAX = 5000   
	random.seed(seed)
	i = 1
	filename = "tsp"+str(cities)
	with open(filename, "w") as f:
		for _ in itertools.repeat(None, cities):
			f.write("{p} {p0} {p1}\n".format(p=i, p0=random.randint(MIN, MAX), p1=random.randint(MIN, MAX)))
			i = i + 1
	return filename


def takeInput(file):
	global cities
	f = open(file,'r').read().splitlines()
	cities = len(f)
	for a in f:
		m = a.split()
		i = int(m[0])
		x = float(m[1])
		y = float(m[2])
		nodeDict[i] = Node(i, x, y)
	return


def save2optNeighbours(tour):
	""" You can print the list on stdout to check if your getting correct 2opt-neighbours
		or look into 2optNeighbours.txt file in your current directory"""
	tourList = generate2optNeighbours(tour)
	print(tour)
	print(tourList)
	filename = "2optNeighbours.txt"
	file = open(filename, 'w')
	for i in tourList:
		file.write("%s\n" % i)

def save3optNeighbours(tour):
	""" You can print the list on stdout to check if your getting correct 2opt-neighbours
		or look into 2optNeighbours.txt file in your current directory"""
	tourList = generate3optNeighbours(tour)
	print(tour)
	print(tourList)
	filename = "3optNeighbours.txt"
	file = open(filename, 'w')
	for i in tourList:
		file.write("%s\n" % i)

def generateRandomTour(r2seed):
	global cities
	print("number of cities are ",cities)
	random.seed(r2seed)
	tour = [x for x in range(1,cities+1)]
	shuffle(tour)
	return tour

def getTourLength(tour):
	global cities
	if len(tour) == 0:
		return 0

	length = 0
	if len(tour) == 2:
		return getDistance(nodeDict[tour[0]],nodeDict[tour[1]])

	for x in range(len(tour)-1):
		length += getDistance(nodeDict[tour[x]],nodeDict[tour[x+1]]) 
	
	length += getDistance(nodeDict[tour[0]],nodeDict[tour[-1]])

	return length

def getDistance(n1, n2):
	return math.sqrt((n1.x-n2.x)*(n1.x-n2.x) + (n1.y-n2.y)*(n1.y-n2.y))

unionFind= [] 

def union(x,y):
	k1 = unionFind[x]
	k2 = unionFind[y]
	for x in range(cities+1):
		if unionFind[x] == k1:
			unionFind[x] = k2


def find(x,y):
	return unionFind[x] == unionFind[y]


#############################################################################################

def generate2optNeighbours(tour):
	global cities
	all_possible_neighbours = []

	"*** YOUR CODE HERE ***"
	

	"*** --------------  ***"
	l = len(tour)

	for i in range(1 ,l):
		for j in range(i + 1 , l):
			if i == 1 and j == l -1 : 
				continue
			M = tour.copy()
			x = M[i:j+1]
			x.reverse()
			M[i:j+1] = x
			all_possible_neighbours.append(M)
	return all_possible_neighbours

def generate3optNeighbours(tour):
	global cities
	all_possible_neighbours = []

	"*** YOUR CODE HERE ***"
	tour.append(tour[0])
	local = tour.copy()
	c = 0
	for i in range(cities):
		for j in range( i + 2 , cities): 
			for k in range(j + 2 , cities):
				if k == cities - 1 and i == 0:
					continue
				c = c + 1 
				neighbour1 = tour.copy()
				neighbour2 = tour.copy()
				neighbour3 = tour.copy()
				neighbour4 = tour.copy()
				s1 = tour[i + 2:j]
				s2 = tour[j + 2 :k]
				s1_r = s1.copy()
				s1_r.reverse()
				s2_r = s2.copy()
				s2_r.reverse()
				# print("s1",s1)
				# print("s2",s2)

				# n1
				S = []
				S.append(local[i])
				S.append(local[k])
				S = S + s2_r.copy() 
				S.append(local[j + 1])
				S.append(local[i + 1])
				S = S + s1.copy()
				S.append(local[j])
				S.append(local[k + 1])
				# print("S", S, "i" , i  , "k"  , k)
				neighbour1[i : k + 2] = S
				# print("n1" , neighbour1)

				# n2
				S = []
				S.append(local[i])
				S.append(local[j + 1])
				S = S + s2.copy()
				S.append(local[k])
				S.append(local[j])
				S = S + s1_r.copy()
				S.append(local[i + 1])
				S.append(local[k + 1])
				neighbour2[i : k + 2] = S.copy()

				# n3
				S = []
				S.append(local[i])
				S.append(local[j])
				S = S + s1_r.copy()
				S.append(local[i + 1])
				S.append(local[k])
				S = S + s2_r.copy()
				S.append(local[j + 1])
				S.append(local[k + 1])
				neighbour3[i : k + 2] = S.copy()

				# n4
				S = []
				S.append(local[i])
				S.append(local[j + 1])
				S = S + s2.copy()
				S.append(tour[k])
				S.append(tour[i + 1])
				S = S + s1.copy()
				S.append(tour[j])
				S.append(tour[k + 1])
				neighbour4[i : k + 2] = S.copy()


				all_possible_neighbours.append(neighbour1[:-1])
				all_possible_neighbours.append(neighbour2[:-1])
				all_possible_neighbours.append(neighbour3[:-1])
				all_possible_neighbours.append(neighbour4[:-1])



	# print(c)
	

	"*** --------------  ***"
	return all_possible_neighbours    


def generate3optand2optNeighbours(tour):
	# helper function
	all_possible_neighbours = []
	optNeighbours2 = generate2optNeighbours(tour)
	optNeighbours3 = generate3optNeighbours(tour)
	all_possible_neighbours = optNeighbours2 + optNeighbours3
	# uncomment this line to check the number of neighbours
	# print(len(all_possible_neighbours), len(optNeighbours2), len(optNeighbours3))
	return all_possible_neighbours

def generateRandomNeighbour(tour):
	global cities
	random_neighbour = []

	"*** YOUR CODE HERE ***"
	

	"*** --------------  ***"

	return random_neighbour


def firstChoiceHillClimb(initial_tour,num_iter=100000):

	tourLengthList = []
	minTour = []

	"*** YOUR CODE HERE ***"
	

	"*** --------------  ***"
	return tourLengthList, minTour



def hillClimbFull(initial_tour, getNeighbours):
	""" Use the given tour as initial tour, Use your generate2optNeighbours() to generate
		all possible 2opt neighbours and apply hill climbing algorithm. Store the tour lengths
		that you are getting after every hill climb step in the list tourLengthList.
		Store the minimum tour found after the hill climbing algorithms in minTour.
		Your code will return the tourLengthList and minTour.     
		You will find 'task2.png' in current directory which shows hill climb algorithm performace
		The tourLengthList will be used to generate a graph which plots tour lengths with each step.
		that is hill climb iterations against tour length"""

	global cities
	tourLengthList = []
	minTour = []

	"*** YOUR CODE HERE ***"
	# Returns the tour + finally found local minima.
	
	"*** --------------  ***"
	def onestep(tour):
		all_neighgours = getNeighbours(tour)
		score_list = [getTourLength(C) for C in all_neighgours]
		improv = min(score_list)
		if(improv >= getTourLength(tour)):
			return tour
		else:
			return all_neighgours[score_list.index(min(score_list))]

	current = initial_tour 
	while True:
		
		X = onestep(current)
		tourLengthList.append(getTourLength(X))
		if X == current:
			break
		else:
			current = X


	minTour = current
	# print(tourLengthList)
	return tourLengthList, minTour

def nearestNeighbourTour(initial_city):
	tour = []
	global nodeDict
	global cities

	"*** YOUR CODE HERE ***"
	
	"*** --------------  ***"
	# PairwiseDist = [[ getDistance(nodeDict[i] , nodeDict[j])  for j in range(1 , cities + 1)] for i in range(1 , cities + 1)]
	tour.append(initial_city)
	for i in range(1 , cities):
		current = tour[-1]
		distances = [ float("inf") if k in tour else getDistance(nodeDict[current] , nodeDict[k])  for k in range(1 , cities + 1) ]
		nearest = distances.index(min(distances)) + 1 
		tour.append(nearest)

	
	return tour

def eucledianTour(initial_city):
	global unionFind, cities, nodeDict
	edgeList = []

	"*** YOUR CODE HERE ***"
	# part 1
	
	

	"*** --------------  ***"
	# Edge list.
	# use list compressiong
	edgeList = [[i , j,  getDistance(nodeDict[i] , nodeDict[j])]   for i in range(1 , cities + 1) for j in range(i + 1, cities + 1)] 
	# print(len(edgeList))

	'''KRUSKAL's algorithm'''

	mst = []
	for x in range(cities+1):
		unionFind.append(x)
	
	edgeList.sort(key=lambda x:int(x[2]))
	for x in edgeList:
		if(find(x[0],x[1]) == False):
			mst.append((x[0],x[1]))
			union(x[0],x[1])

	'''FINISHES HERE'''
	fin_ord = finalOrder(mst, initial_city)

	return fin_ord





def finalOrder(mst, initial_city):

	fin_order = []
	"*** YOUR CODE HERE ***"
	# for part 3
	#  the order. 
	mst.reverse() # to ensure left - right.
	# print(mst)
	Adjacency = [[] for i in range( 1 , cities + 1)]
	# for i in range(cities):
	# 	for j in range( 1  , cities + 1):
	# 		if (i + 1 , j) in mst or (j , i + 1) in mst :
	for c in mst:
			Adjacency[c[0] - 1].append(c[1])
			Adjacency[c[1] - 1].append(c[0])
	

				

	# print(mst)
	# def fin_order_helper(city):
	# 	fin_order.append(city)
	# 	for k in Adjacency[city -1]:
	# 		if k in fin_order:
	# 			continue
	# 		else:
	# 			fin_order_helper(k)

	# stack = [initial_city]
	# while len(stack) > 0:
	# 	current = stack[-1]
	# 	stack = stack[:-1]
	# 	if current not in fin_order:
	# 		fin_order.append(current)
	# 	# children = Adjacency[current - 1]
	# 	# dummy = []
	# 	for c in mst:
	# 		if c[0] == current and c[1] not in fin_order:
	# 			stack.append(c[0])
	# 			stack.append(c[1])
	# 			break
	# 		elif c[1] == current and c[0] not in fin_order:
	# 			stack.append(c[1])
	# 			stack.append(c[0])
	# 			break
	stack = [initial_city]
	while len(stack) > 0:
		current = stack[-1]
		stack = stack[:-1]
		fin_order.append(current)
		children = Adjacency[current - 1]
		# dummy = []
		for c in children:
			if c in fin_order:
				continue
			else:
				stack.append(c)
	
	# print(fin_order)
	# Hugely depends on the order 
	# fin_order.append(initial_city)
	# for c in Adjacency[initial_city -1]:
	# 	fin_order_helper(c)
	




	# print(fin_order , len(fin_order))

	"*** --------------  ***"
	return fin_order

 
##################################################################################################
####### DO NOT CHANGE THIS CODE ###########################################################################
###########################################################################################################
def hillClimbWithNearestNeighbour(start_city, getNeighbours):
	tour = nearestNeighbourTour(start_city)
	tourLengthList, min_tour = hillClimbFull(tour, getNeighbours)
	return tourLengthList
	

def hillClimbWithEucledianMST(initial_city, getNeighbours):
	tour = eucledianTour(initial_city)
	tourLengthList, minTour = hillClimbFull(tour , getNeighbours)
	
	#drawTour(nodeDict, minTour)
	return tourLengthList

def firstChoiceHillClimbing(initial_city):
	tour = eucledianTour(initial_city)
	tourLengthList, minTour = firstChoiceHillClimb(tour)
	return tourLengthList


def hillClimbWithRandomTour(tour, getNeighbours):
	tourLengthList = []
	tourLengthList, minTour = hillClimbFull(tour, getNeighbours)
	return tourLengthList

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', '-f', action='store', dest='file', help="Provide a file name (if file given then no need to provide city and random seed option that is -n and -r)")
	parser.add_argument('--cities', '-n', action='store', type=int, dest='cities', help="Provide number of cities in a tour")
	parser.add_argument('--r1seed', action='store', type=int, dest='r1seed', default=1, help="random seed")
	parser.add_argument('--r2seed', action='store', type=int, dest='r2seed', default=1, help="random seed")
	parser.add_argument('--task', '-t', action='store', type=int, dest="task", help="task to execute")
	parser.add_argument('--start_city', '-i', action='store', type=int, default=1, dest='start_city', help="Initial city")
	parser.add_argument('--submit', action='store_true', help="final submission")

	args = parser.parse_args()

	if args.submit:
		takeInput("data/st70.tsp");
	elif args.file:
		takeInput(args.file)
	elif args.cities:
		file = generateFile(args.cities, args.r1seed)
		takeInput(file)
	else:
		print("Please provide either a file or combination of number of cities and random seed")
		sys.exit()

	if not args.task:
		print("Please provide task number to execute")
		sys.exit()

	if args.task == 1:
		tour = generateRandomTour(args.r2seed)
		save2optNeighbours(tour)

	if args.task == 5:
		tour = generateRandomTour(args.r2seed)
		save3optNeighbours(tour)


	if not args.submit:
		if args.task == 2:
			tour = generateRandomTour(args.r2seed)
			tourLengthList = hillClimbWithRandomTour(tour, generate2optNeighbours)
			print(tourLengthList[-1])
			graph_plot.generateGraph(tourLengthList, "task2.png")

		if args.task == 3:
			tourLengthList = hillClimbWithNearestNeighbour(args.start_city, generate2optNeighbours)
			print(tourLengthList[-1])
			graph_plot.generateGraph(tourLengthList, "task3.png")

		if args.task == 4:
			tourLengthList = hillClimbWithEucledianMST(args.start_city, generate2optNeighbours)
			print(tourLengthList[-1])
			graph_plot.generateGraph(tourLengthList, "task4.png")

		if args.task == 6:
			tour = generateRandomTour(args.r2seed)
			tourLengthList = hillClimbWithRandomTour(tour, generate3optand2optNeighbours)
			print(tourLengthList[-1])
			graph_plot.generateGraph(tourLengthList, "task6.png")            

		if args.task == 7:
			tourLengthList = hillClimbWithNearestNeighbour(args.start_city, generate3optand2optNeighbours)
			print(tourLengthList[-1])
			graph_plot.generateGraph(tourLengthList, "task7.png")

		if args.task == 8:
			tourLengthList = hillClimbWithEucledianMST(args.start_city, generate3optand2optNeighbours)
			print(tourLengthList[-1])
			graph_plot.generateGraph(tourLengthList, "task8.png")

		if args.task == 9:
			tourLengthList = firstChoiceHillClimbing(args.start_city)
			print(tourLengthList[-1])
			graph_plot.generateGraph(tourLengthList, "task9.png")


	else:
		if args.task == 2:
			data = []
			for i in range(1, numberOfRuns+1):
				random_seed = i
				tour = generateRandomTour(random_seed)
				tourLengthList = hillClimbWithRandomTour(tour, generate2optNeighbours)
				data.append(tourLengthList)

			graph_plot.generateFinalGraph(data, "task2_submit.png", 2)

		if args.task == 3:
			data = []
			for i in range(1, numberOfRuns+1):
				start_city = i
				tourLengthList = hillClimbWithNearestNeighbour(start_city, generate2optNeighbours)
				data.append(tourLengthList)

			graph_plot.generateFinalGraph(data, "task3_submit.png", 3)

		if args.task == 4:
			tourLengthList = hillClimbWithEucledianMST(args.start_city, generate2optNeighbours)
			graph_plot.generateGraph(tourLengthList, "task4_submit.png")

		if args.task == 6:
			data = []
			for i in range(1, numberOfRuns+1):
				random_seed = i
				tour = generateRandomTour(random_seed)
				tourLengthList = hillClimbWithRandomTour(tour, generate3optand2optNeighbours)
				data.append(tourLengthList)

			graph_plot.generateFinalGraph(data, "task6_submit.png", 2)

		if args.task == 7:
			data = []
			for i in range(1, numberOfRuns+1):
				start_city = i
				tourLengthList = hillClimbWithNearestNeighbour(start_city, generate3optand2optNeighbours)
				data.append(tourLengthList)

			graph_plot.generateFinalGraph(data, "task7_submit.png", 3)

		if args.task == 8:
			tourLengthList = hillClimbWithEucledianMST(args.start_city, generate3optand2optNeighbours)
			graph_plot.generateGraph(tourLengthList, "task8_submit.png")

###################################################################################

