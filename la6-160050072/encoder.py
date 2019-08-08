import sys

# IO
f = open(sys.argv[1],"r+")	
lines = f.readlines()
maze = []
for line in lines:
	rows = line.strip().split()
	maze.append(rows)
p = 1.0


# print(sys.argv)
# print("Hey tron" , len(sys.argv[1]))
if len(sys.argv) > 2: 
	p = float(sys.argv[2])

# print("probability" , p)
# exit()

# get the variables
numState = 0 
numActions = 4 
nr, nc = len(maze) , len(maze[0]) 
stop = []
states = [[-1 for _ in range(nc)] for _ in range(nr)]

for x in range(nr):
	for y in range(nc):
		if maze[x][y] == '1':
			continue
		if maze[x][y] == '2':
			start = numState
		if maze[x][y] == '3':
			stop.append(numState)	
		states[x][y] = numState
		numState += 1 

# Define transitions
transitions = {}
for x in range(nr):
	for y in range(nc):
		if maze[x][y] == '1' or maze[x][y] == '3': 
			# skip if a wall or a stop state 
			continue
		# calculate valid moves  
		nN = maze[x - 1][y]
		nE = maze[x][y + 1]
		nS = maze[x + 1][y]
		nW = maze[x][y - 1]

		validM = (nN != '1') + (nE != '1') + (nS != '1') + (nW != '1') 
		validL = [nN , nE  , nS , nW]
		stateL = [states[x - 1][y] , states[x][y + 1] , states[x + 1][y], states[x][y - 1]]
		if validM == 0: 
			validM = 1
		tP = p + (1 - p) / validM
		fP = (1 - p) / validM
		# # north
		# # nN = maze[x - 1][y]
		s = states[x][y]
		# if nN == '1':
		# 	#  reward , probability
		# 	transitions[ s, 0 , s ] = [-1 , 1]
		# elif nN == '2' or nN == '0':
		# 	# transitions[ s, 0 , states[x - 1][y] ] = [-1 , tP]
		# 	# transitions[ s, 0 , states[x - 1][y] ] = [-1 , tP]
		# 	# transitions[ s, 0 , states[x - 1][y] ] = [-1 , tP]
		# 	# transitions[ s, 0 , states[x - 1][y] ] = [-1 , tP]
		# 	for i in range(4):
		# 		if 
		# else:
		# 	# for final state give a probability of 1.
		# 	transitions[ s, 0 , states[x - 1][y]] = [2 * nr * nc , 1]

		# # east

		# s = states[x][y]
		# if nE == '1':
		# 	#  reward , probability
		# 	transitions[ s, 1 , s ] = [-1 , 1]
		# elif nE == '2' or nE == '0':
		# 	transitions[ s, 1 , states[x][y + 1] ] = [-1 , 1]
		# else:
		# 	# for final state give a probability of 1.
		# 	transitions[ s, 1 , states[x][y + 1]] = [2 * nr * nc , 1]

		# # south

		# s = states[x][y]
		# if nS == '1':
		# 	#  reward , probability
		# 	transitions[ s, 2 , s ] = [-1 , 1]
		# elif nS == '2' or nS == '0':
		# 	transitions[ s, 2 , states[x + 1][y] ] = [-1 , 1]
		# else:
		# 	# for final state give a probability of 1.
		# 	transitions[ s, 2 , states[x + 1][y]] = [2 * nr * nc , 1]

		# # west

		# s = states[x][y]
		# if nW == '1':
		# 	#  reward , probability
		# 	transitions[ s, 3, s ] = [-1 , 1]
		# elif nW == '2' or nW == '0':
		# 	transitions[ s, 3, states[x][y - 1] ] = [-1 , 1]
		# else:
		# 	# for final state give a probability of 1.
		# 	transitions[ s, 3, states[x][y - 1]] = [2 * nr * nc , 1]
		for i in range(4):
			if validL[i] == '1':
				transitions[s , i , s] = [ -1, 1]
				continue 
			for j in range(4):
				reward = -1 # base reward. 
				if validL[j] == '3':
					reward = 100 + numState
				if i == j: 
					if tP == 0.0:
						continue
					transitions[s , i , stateL[i]] = [reward , tP]
				elif stateL[j] == -1:
					# not a valid move
					continue
				else:
					if fP == 0.0:
						continue
					transitions[s , i , stateL[j]] = [reward , fP]

# num States
print("numStates" , numState)
# num Actions
print("numActions" , 4)
# start
print("start", start)
# end
print("end" , " ".join(list(map( str , stop))))
# transitions
for edge in transitions:
	print("transition" , edge[0] , edge[1] , edge[2]  , transitions[edge][0] , transitions[edge][1])
# discount
print("discount" , 0.99)




