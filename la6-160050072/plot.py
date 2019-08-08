# importing the required module 
import matplotlib.pyplot as plt 

# x axis values 
x = []
for i in range(11):
	x.append(i * 0.1)
# corresponding y axis values 
y = [454,134,84,96,66,48,36,38,30,22,22]

# plotting the points 
plt.plot(x, y) 

# naming the x axis 
plt.xlabel('p') 
# naming the y axis 
plt.ylabel('No of actions needed') 

# giving a title to my graph 
plt.title('Plot of no of actions vs p ') 

# function to show the plot 
plt.show() 
