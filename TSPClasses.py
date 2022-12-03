#!/usr/bin/python3


import math
import numpy as np
import random
import time
import copy



class TSPSolution:
	def __init__( self, listOfCities):
		self.route = listOfCities
		self.cost = self._costOfRoute()
		#print( [c._index for c in listOfCities] )

	def _costOfRoute( self ):
		cost = 0
		last = self.route[0]
		for city in self.route[1:]:
			cost += last.costTo(city)
			last = city
		cost += self.route[-1].costTo( self.route[0] )
		return cost

	def enumerateEdges( self ):
		elist = []
		c1 = self.route[0]
		for c2 in self.route[1:]:
			dist = c1.costTo( c2 )
			if dist == np.inf:
				return None
			elist.append( (c1, c2, int(math.ceil(dist))) )
			c1 = c2
		dist = self.route[-1].costTo( self.route[0] )
		if dist == np.inf:
			return None
		elist.append( (self.route[-1], self.route[0], int(math.ceil(dist))) )
		return elist


def nameForInt( num ):
	if num == 0:
		return ''
	elif num <= 26:
		return chr( ord('A')+num-1 )
	else:
		return nameForInt((num-1) // 26 ) + nameForInt((num-1)%26+1)








class Scenario:

	HARD_MODE_FRACTION_TO_REMOVE = 0.20 # Remove 20% of the edges
	'''
	params:
	city_locations: x, y coordinates of points
	difficulty: string
	rand_seed: int
	'''
	def __init__( self, city_locations, difficulty, rand_seed ):
		self._difficulty = difficulty

		if difficulty == "Normal" or difficulty == "Hard":
			self._cities = [City( pt.x(), pt.y(), \
								  random.uniform(0.0,1.0) \
								) for pt in city_locations]
		elif difficulty == "Hard (Deterministic)":
			random.seed( rand_seed )
			self._cities = [City( pt.x(), pt.y(), \
								  random.uniform(0.0,1.0) \
								) for pt in city_locations]
		else:
			self._cities = [City( pt.x(), pt.y() ) for pt in city_locations]


		num = 0
		for city in self._cities:
			#if difficulty == "Hard":
			city.setScenario(self)
			city.setIndexAndName( num, nameForInt( num+1 ) )
			num += 1

		# Assume all edges exists except self-edges
		ncities = len(self._cities)
		self._edge_exists = ( np.ones((ncities,ncities)) - np.diag( np.ones((ncities)) ) ) > 0

		if difficulty == "Hard":
			self.thinEdges()
		elif difficulty == "Hard (Deterministic)":
			self.thinEdges(deterministic=True)

	def getCities( self ):
		return self._cities


	def randperm( self, n ):				#isn't there a numpy function that does this and even gets called in Solver?
		perm = np.arange(n)
		for i in range(n):
			randind = random.randint(i,n-1)
			save = perm[i]
			perm[i] = perm[randind]
			perm[randind] = save
		return perm

	def thinEdges( self, deterministic=False ):
		ncities = len(self._cities)
		edge_count = ncities*(ncities-1) # can't have self-edge
		num_to_remove = np.floor(self.HARD_MODE_FRACTION_TO_REMOVE*edge_count)

		can_delete	= self._edge_exists.copy()

		# Set aside a route to ensure at least one tour exists
		route_keep = np.random.permutation( ncities )
		if deterministic:
			route_keep = self.randperm( ncities )
		for i in range(ncities):
			can_delete[route_keep[i],route_keep[(i+1)%ncities]] = False

		# Now remove edges until 
		while num_to_remove > 0:
			if deterministic:
				src = random.randint(0,ncities-1)
				dst = random.randint(0,ncities-1)
			else:
				src = np.random.randint(ncities)
				dst = np.random.randint(ncities)
			if self._edge_exists[src,dst] and can_delete[src,dst]:
				self._edge_exists[src,dst] = False
				num_to_remove -= 1




class City:
	def __init__( self, x, y, elevation=0.0 ):
		self._x = x
		self._y = y
		self._elevation = elevation
		self._scenario	= None
		self._index = -1
		self._name	= None

	def setIndexAndName( self, index, name ):
		self._index = index
		self._name = name

	def setScenario( self, scenario ):
		self._scenario = scenario

	''' <summary>
		How much does it cost to get from this city to the destination?
		Note that this is an asymmetric cost function.
		 
		In advanced mode, it returns infinity when there is no connection.
		</summary> '''
	MAP_SCALE = 1000.0
	def costTo( self, other_city ):

		assert( type(other_city) == City )

		# In hard mode, remove edges; this slows down the calculation...
		# Use this in all difficulties, it ensures INF for self-edge
		if not self._scenario._edge_exists[self._index, other_city._index]:
			return np.inf

		# Euclidean Distance
		cost = math.sqrt( (other_city._x - self._x)**2 +
						  (other_city._y - self._y)**2 )

		# For Medium and Hard modes, add in an asymmetric cost (in easy mode it is zero).
		if not self._scenario._difficulty == 'Easy':
			cost += (other_city._elevation - self._elevation)
			if cost < 0.0:
				cost = 0.0					# Shouldn't it cost something to go downhill, no matter how steep??????


		return int(math.ceil(cost * self.MAP_SCALE))


'''
This object is used to hold state information in the priority queue of our Branch and Bound algorithm
'''
class State:
	#this is a class (static) variable that will increment everytime a new state is created
	counter = 0
	def __init__(self):
		#This is to uniquely identify the state in the priority queue
		self.state_id = None
		#We need this so that we can determine the child states
		self.curr_ind = None
		#This is used to place a state in the HeapPriorityQueue
		self.priority_key = None 
		self.is_solution = None
		#These used to help us calculate the priority key
		self.lower_bound = None
		self.depth = None
		# This matrix will only contain paths that exist (it won't contain any infinite values. This is to save space. 
		# Once there is an infinite value, why keep it? We are never going to use it again)
		# The matrix helps us determine the cities that haven't been visited
		self.matrix = {}
		#This is for storing the order of cities visited
		self.prev_visited = []
		#This is called to set the individual node id based off of the static counter variable
		self.assign_state_id()
	
	def assign_state_id(self):
		self.state_id = State.counter
		State.counter += 1


'''
This class actls like an enumeration to help us identify which initial algorithm we use to find the initial bssf
'''
class InitialStrategies:
	#This strategy uses a more restrictive selection process to drill to a solution faster than the default settings
	RESTRICTED_BRANCH_AND_BOUND = 0
	#This is the most restrictive setting of the strategy above. It simply selects the state with the lowest bound at each level
	GREEDY = 1
	#This selects the best n states and then selects a random one of them
	NOISY_GREEDY = 3

'''
This class has functions that are used across the other algorithms. Partial path is used for our branch and bound solution. It is also used in our greedy algorithm to determine which path to take. 
'''
class SharedUtils:
	def __init__(self, priority_key_funct=None):
		#This is the function used to calculate the priority key
		self.calculate_priority_key = priority_key_funct

	'''
	This function takes the state, reduces the matrix, and updates the lower_bound. 
	It doesn't return anything because all of the information is updated in the state

	Time complexity: O(n^3) 
	- because we go through all of the rows and all of the columns (n^2 cells) at most 3 times and two of those times we find the min for 
	n rows/columns which I assume takes O(n) time. 2 * n^3 + n^2 = O(n^3) 
	- first time is because we need to delete the unused cells
	- second is because we need to find the min of each row and subtract it from each value in the row
	- third is the same as above but with the columns

	Space complexity: O(n^2)
	- the matices are the dominant parts in terms of space for this function. 
	- There are at most 2 of them.
	- each one is of size n^2 elements
	- 2 * n^2 = O(n^2)
	'''
	def reduce_matrix(self, state):
		#edge case
		if len(state.matrix) == 0:
			return None
		
		#If this is a start state we are reducing, we need to initialize the lower bound so we can add to it
		if state.lower_bound == None:
			state.lower_bound = 0
		
		#Get rid of the 'from' row, the 'to' column and the other corresponding cells if needed
		#if prev is not none, then this is a child state and we need to get rid of some things
		if len(state.prev_visited) != 0:
			#prev is the index in the cities list corresponding to the previous city 
			#Delete the row using the final element in prev_visited array
			del state.matrix[state.prev_visited[-1]]
			#Delete the 'to' column using the curr_ind property which is the index of the current city
			rows_to_del = []
			for row in state.matrix:
				#This is constant time lookup
				if state.curr_ind in state.matrix[row]:
					del state.matrix[row][state.curr_ind]
					#if row is empty, store its key so we can delete it later
					if len(state.matrix[row]) == 0:
						rows_to_del.append(row)
						
			#delete the empty rows
			for row in rows_to_del:
				del state.matrix[row]

			#We've already deleted the from-to cell, we just need to delete the to-from cell if needed
			if state.curr_ind in state.matrix and state.prev_visited[-1] in state.matrix[state.curr_ind]:
				del state.matrix[state.curr_ind][state.prev_visited[-1]]
				#if the row is empty, we need to delete the whole row
				if len(state.matrix[state.curr_ind]) == 0:
					del state.matrix[state.curr_ind]
		
		#Now we do the actual reduction starting by going through the rows
		#This is a dictionary where we use the column as the first key instead of the row for easy access when we iterate through the columns later
		columns = {}
		for row in state.matrix:

			#Find the min of the row
			min_val = min(state.matrix[row].values())

			#add min val to the lower bound and the subtract from the matrix
			state.lower_bound += min_val
			#As we iterate through the columns, we add the values to the appropriate columns dict so that we can easily find the min of each column in a later step
			for col in state.matrix[row]:
				#now update the values in the matrix by subtracting the min
				#Note that we will never have a value of infinity in our matrix because we just delete those values 
				state.matrix[row][col] -= min_val
				#add the column key to the column dictionary if needed and instantiate a dictionary to hold the values of that column with the row as the key
				if col not in columns:
					columns[col] = {}
				#add the updated value with the row as the key
				columns[col][row] = state.matrix[row][col]

		#Now we use our columns dictionary to find the min of each column and subtract it from all the other values in the column
		for col in columns:
			#Get the key (row) associated with he min value
			min_key = min(columns[col], key=columns[col].get)
			#get the min val
			min_val = columns[col][min_key]
			state.lower_bound += min_val
			#if min_val is 0 then escape early
			if min_val == 0:
				continue

			for row in columns[col]:
				#because our columns dictionary and state.matrix dict have the same rows and columns, we can use those variables to index into our state.matrix
				#dictionary
				state.matrix[row][col] -= min_val
		
	'''
	This function takes a parent state and looks at all of the possible edges from the current city to the other cities in the matrix
	If there is one, then it generates a child state.

	Time complexity: O(n^4)
	- We create at most n-1 child states.
	- Creating a child state takes O(n^3) time because reduce_matrix() is called 
	- We sort the child states which takes O(nlogn) time if we use a merge sort
	- (n - 1) * O(n^3) + n * log(n) = O(n^4)

	Space complexity: O(n^3)
	- We hold at most n - 1 child states
	- Each child state has a space complexity of O(n^2)
	- (n - 1) * O(n^2) = O(n^3)
	'''
	def create_child_states(self, parent_state, cities, start_city_ind, test_states=None):
		
		# create a list to hold all the children that we will generate. We will sort it later by priority key so we can account for the most important ones first 
		# if we are doing the beam search
		children = []

		# We know that the parent_state won't have a curr_ind value that isn't in the matrix
		# assert(parent_state.curr_ind != None)
		# assert(parent_state.curr_ind in parent_state.matrix)

		if parent_state.curr_ind not in parent_state.matrix:
			return children

		# For all possible children of parent_state:
		# find the 'to' columns at 'from' row 
		for col in parent_state.matrix[parent_state.curr_ind]:
			#We only create the child state if it doesn't go back to the start state when depth of parent is less than num cities - 1
			#Yes, I know there are redundancies in this if clause
			if ((parent_state.depth < (len(cities) - 1)) and (col != start_city_ind)) or ((parent_state.depth == (len(cities) - 1)) \
				and (col == start_city_ind)):
				child = self.create_child_state(parent_state, col, cities, start_city_ind)
				children.append(child)
				# Then we know this is a test. We can just add all of the states to the array because our test only includes 5 cities which is bounded.
				# and barely takes up any memory
				if test_states != None: test_states.append(child)

		# return sorted(children, key=priority_key)
		return sorted(children, key=lambda child: child.priority_key)

	'''
	Description: This function creates an individual child state

	Time complexity: O(n^3) 
	- Making a copy of the parent state takes O(n^2) time
	- Reducing the matrix takes O(n^3) time and this is the dominant part of the function
	- Everything else is constant time
	- O(n^2) + O(n^3) + O(1) = O(n^3)

	Space complexity: O(n^2)
	- Each state has at most n^2 cells in it's matrix
	- Each state has at most n cities in its prev_visited
	- O(n) + O(n^2) = O(n^2)
	'''
	def create_child_state(self, parent_state, to_column, cities, start_city_ind):

		#copy parent state 
		child_state = copy.deepcopy(parent_state)
		child_state.assign_state_id()



		#put a back pointer to parent index 
		child_state.prev_visited.append(parent_state.curr_ind)

		#set the curr_ind to the city the parent is going to
		child_state.curr_ind = to_column

		#increment the depth
		child_state.depth += 1
	
		#update the distance from the start city
		from_city = cities[child_state.prev_visited[-1]]
		to_city = cities[child_state.curr_ind]

		#set is_solution if needed
		if child_state.depth == len(cities) and child_state.curr_ind == start_city_ind:

			child_state.is_solution = True

			#make the priority_key = 0 so that it is garunteed to be put on the top of the priority queue
			child_state.priority_key = 0

			#If we already have the solution, we can return early
			return child_state

		#reduce_matrix will delete corresponding row and column in the child state matrix and update the lower_bound of child_state
		self.reduce_matrix(child_state)

		#calculate_priority_key(child_state)
		self.calculate_priority_key(child_state)

		return child_state

	'''
	Description: This function creates a start state. This needs to be different than the child state because it is independent of a parent state and we 
	can't just copy a parent state like we do in create_child_state(). 

	Time complexity: O(n^3)
	- Reducing the matrix is the most significant part of the function and it takes O(n^3) time.

	Space complexity: O(n^2)
	- Each state has at most n^2 cells in it's matrix
	- Each state has at most n cities in its prev_visited
	- O(n) + O(n^2) = O(n^2)
	'''
	def create_start_state(self, cities, start_city_ind):
		state = State()
		#loop through all of the 'from' rows. 
		for i in range(len(cities)):
			#loop through all of the 'to' coloumns
			for j in range(len(cities)):
				#Get the cities so we can see if an edge between them exists
				from_city = cities[i]
				to_city = cities[j]
				cost = from_city.costTo(to_city)
				# if an edge from city at index i to city at index j exists, add it to our 'matrix'
				if cost != math.inf:
					#We populate it as a dictionary of dictionaries for quick look up and easy deletions
					#Each level of the matrix dictionary acts like a set of [rows] and [columns] that we can iterate through when reducing it
					if i not in state.matrix:
						state.matrix[i] = {}
					state.matrix[i][j] = cost

		state.curr_ind = start_city_ind
		state.depth = 0
		self.reduce_matrix(state)

		self.calculate_priority_key(state)

		return state

	'''
	Description: This function checks if a state is a solution. If it is, then it creates a solution object and checks if the cost is less than the bssf
	If it is, a new bssf set and count is incremented in the solve function with the returned solution object.

	Time complexity: O(n)
	- If it is a solution, we have to loop through n cities to create the solution

	Space complexity: O(n)
	- The solution object grows by a factor of n because it holds the edges between n cities
	'''
	def check_if_solution(self, state, bssf_dist, cities):
		if state.is_solution:
			solution = self.create_solution(state, cities)
			if solution.cost < bssf_dist:
				return solution
		return None

	'''
	Description: This is called when we have found a solution and we need to create a TSPSolution object 
	
	Time complexity: O(n)
	- We have to loop through n cities to create the solution

	Space complexity: O(n)
	- The solution object grows by a factor of n because it holds the edges between n cities
	'''
	def create_solution(self, state, cities):
		city_list = []
		for i in state.prev_visited:
			city_list.append(cities[i])
		return TSPSolution(city_list)

	'''
	This function is used to create a results dictionary

	Time complexity: O(1)
	Space complexity: O(1)
	'''
	def create_results(self, out):
		results = {}
		results['cost'] = out.bssf_dist
		results['time'] = out.end_time - out.start_time
		results['count'] = out.count
		results['soln'] = out.bssf
		results['max'] = out.max
		results['total'] = out.total
		results['pruned'] = out.pruned
		return results
	
	def get_edges(self, cities):
		edges = {}
		for i in range(len(cities)):
			if i not in edges:
				edges[i] = {}
			for j in range(len(cities)):
				cost = cities[i].costTo(cities[j])
				if cost != math.inf:
					edges[i][j] = cost
		return edges

