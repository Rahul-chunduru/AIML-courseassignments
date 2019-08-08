import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
	def __init__(self, state, action, path_cost, parent_node, depth):
		self.state = state
		self.action = action
		self.path_cost = path_cost
		self.parent_node = parent_node
		self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
	"""
	Search the deepest nodes in the search tree first.
	"""

	def convertStateToHash(values):
		""" 
		values as a dictionary is not hashable and hence cannot be used directly in the explored set.
		This function changes values dict into a unique hashable string which can be used in the explored set.
		"""
		l = list(sorted(values.items()))
		modl = [a+b for (a, b) in l]
		return ''.join(modl)

	## YOUR CODE HERE
	# util.raiseNotDefined()
	startNode = Node(problem.getStartState(), None , 0 , -1 , 0)
	Explored_List = []
	frontier = [startNode]
	while len(frontier) > 0:  
		node = frontier[-1]  
		# print convertStateToHash(node.state)
		# print node.depth
		if problem.isGoalState(node.state):
			# print problem.nodes_expanded
			return node.state
		else:
			nS = node.state.copy()
			cD = problem.getSuccessors(nS)
			frontier = frontier[:-1]
			Explored_List.append(convertStateToHash(nS))
			for c in cD:
				if convertStateToHash(c[0]) in Explored_List:
					continue
				else:
					frontier.append(Node(c[0] , c[1] , node.path_cost + 1 , node  , node.depth + 1 ))
		# print len(frontier)
	print "Couldn't find solution"
	return False


######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
	"""
	A heuristic function estimates the cost from the current state to the nearest
	goal in the provided SearchProblem.  This heuristic is trivial.
	"""
	return 0

def heuristic(state, problem):
	# It would take a while for Flat Earther's to get accustomed to this paradigm
	# but hang in there.
	# print problem.G.node[state]
	start = [(problem.G.node[state]['x'] , 0 , 0),(problem.G.node[state]['y'] , 0 , 0) ]
	end = [(problem.G.node[problem.end_node]['x'] , 0 , 0),(problem.G.node[problem.end_node]['y'] , 0 , 0) ]
	return util.points2distance(start ,end)
	# util.raiseNotDefined()

def AStar_search(problem, heuristic=nullHeuristic):

	# Costs = {}
	Explored_List = []
	startNode = Node(problem.getStartState(), None , 0 , None , 0)
	# my nodes state is the required thing.
	# print startNode
	frontier = util.PriorityQueue()
	frontier.push(startNode ,  (0 + heuristic(problem.getStartState(), problem)))
	finalNode = None
	while frontier.isEmpty() == False:
		node = frontier.pop()
		if node.state in Explored_List:
			continue
		if problem.isGoalState(node.state):
			# Costs[node.state] = [node.state , node.path_cost]
			finalNode = node
			# print node.path_cost, "found"
			break
		else:
			Explored_List.append(node.state)
			L = problem.getSuccessors(node.state)
			for n in L:
				if n[0] in Explored_List:
					continue
				cCost =  node.path_cost + n[2]
				frontier.push(Node(n[0], node ,  cCost , node, node.depth +  1),  ( cCost + heuristic(n[0], problem)))

	
	if finalNode == None:
		print finalNode
		return []
	Path = []
	cNode = finalNode
	Path.append(cNode.state)
	while cNode.state != startNode.state:
		cNode = cNode.parent_node
		# print cNode.depth
		Path.append(cNode.state)

	Path.reverse()
	# print Path
	return Path





	# util.raiseNotDefined()