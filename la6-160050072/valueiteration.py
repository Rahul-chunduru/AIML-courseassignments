import sys, time
t0 = time.clock()
f = open(sys.argv[1],"r+")	
lines = f.readlines()

# Initialilzation
transitions = {}
edges = {}
numStates , numActions, start , stop, discount = 0 , 0 , -1 , [], 1  

# IO -------------
for line in lines:
	words = line.strip().split()
	if words[0] == 'numStates':
		numStates = int(words[1])
	elif words[0] == 'numActions':
		numActions = int(words[1])
	elif words[0] == 'start':
		#  initialize edges
		for i in range(numStates):
			for j in range(numActions):
				edges[i , j] = []	
		start = int(words[1])
	elif words[0] == 'end':
		stop = list(map( int, words[1:])) # making the end list
	elif words[0] == 'transition':
		transitions[int(words[1]) ,int(words[2]) , int(words[3]) ] = [float(words[4]) , float(words[5])]
		if(float(words[5])) > 0 :
			edges[int(words[1]),int(words[2]) ].append(int(words[3]))
	elif words[0] == 'discount':
		discount = float(words[1])
#  Value iteration. 
# print(edges)

V = [ 0 for _ in range(numStates)] # initialize value vector
diff = 1 
req_diff = 10 **-16
iterations = 0 
while diff > req_diff:
	iterations += 1 
	K = [ 0 if s1 in stop else max([ sum([ transitions[s1 , acc , s2][1]*(transitions[s1 , acc , s2][0] + discount * V[s2]) \
	 	for s2 in edges[s1,acc]])  \
			for acc in range(numActions)]) for s1 in range(numStates)]
	diff = max(map(lambda t : abs(t[0] - t[1]) , zip(K , V)))
	V = K[:]

# getting the solution
# is this how the end states are to be handled
for s in range(numStates):
	Q = [ sum([ transitions[s, acc , s2][1]*(transitions[s , acc , s2][0] + discount * V[s2]) \
	 	for s2 in edges[s , acc]])  \
			for acc in range(numActions)]
	# action = 0
	# if V[s] in Q:
	# 	action = Q.index(V[s]) 
	# else:
	# 	action = -1 
	action = Q.index(max(Q))
	if s in stop:
		action = -1

	# print(V[s] , action)	
	V[s] = round(V[s] , 9) 
	print(V[s] , action)  # set precision to 9 digits
print("iterations",iterations)

# print(time.clock()-t0)









f.close()