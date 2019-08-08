import sys
import random
import numpy as np

random.seed(0)
# IO
f = open(sys.argv[1],"r+")	
lines = f.readlines()

g = open(sys.argv[2],"r+")	
values = g.readlines()

p = 1.0 
if len(sys.argv) > 3:
	p = float(sys.argv[3])

# print(sys.argv)
# print("probability" , p)

maze = []
for line in lines:
	rows = line.strip().split()
	maze.append(rows)

values = values[:-1]
values = [x.strip().split() for x in values]


#  encoding
directions = {}
directions[0] = 'N'
directions[1] = 'E'
directions[2] = 'S'
directions[3] = 'W'


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
			posx = x 
			posy = y
		if maze[x][y] == '3':
			stop.append(numState)	
		states[x][y] = numState
		numState += 1 


actions = []
while start not in stop:
	#####################
	
	pick = random.random()
	if pick <= p:
		act = int(values[start][1])
	else:	
		Actp = []
		for i in range(3 , -1  , -1):
			# print("using the unobvious")
			tempx = posx - pow( - 1 , i // 2) * (1 - (i % 2))
			tempy = posy + pow( - 1 , i // 2) * ( i % 2)
			temp = states[tempx][tempy]
			if temp == -1:
				continue 
			else:
				Actp.append(i)
		random.shuffle(Actp)
		act = random.choice(Actp) #using random.choice

	#####################
	actions.append(act)
	posx = posx - pow( - 1 , act // 2) * (1 - (act % 2))
	posy = posy + pow( - 1 , act // 2) * (act % 2)
	start = states[posx][posy]
	# print(posx , posy, act)

actions = list(map(lambda x: directions[x] , actions))
print(" ".join(actions)) # print the directions.
# print(len(actions))




