#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from PriorityQueue import HeapPriorityQueue
import copy

'''
Descriptions of functions and classes along with any discussion of space or time complexity will be placed in multi-line comments. 
Descriptions of what is happening in the code (variables, logic, etc) will be in single-line comments. 
'''

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
		#this is used for the solution
		self.curr_dist = None
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
	#This starts with an initial bssf of inf 
	NO_INITIAL = 2


'''
This is the class that is called by the gui to solve the TSP problem specified
'''
class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

		#These bssf variables need to be kept in this outer class because we want to be able to use a bssf from one strategy in another strategy and compare 
		#the bssf among them all
		self.bssf = None
		self.bssf_dist = math.inf

	def setupWithScenario( self, scenario ):
		self._scenario = scenario
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < math.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' 
	Description: This algorithm uses the BranchAndBoundSolver inner class that I created but with the hyper-parameters set such that it acts as a greedy
	solver. We just set the initial strategy to NO_INITIAL, the max_children to 1 because we only want to select the best child, the max_queue_size to 1 because we
	only want to select the best at each state, the depth importance to 0 so we only select based off of lower bound (it won't matter anyway because we are selecting
	the best of states only on the same depth), and the lower bound importance to 1 because that is what we will determine our decision on. 

	Time Complexity: O(n^3) 
	- because we call solve() at most n times starting at each city. 
	- In each call to solve() we expand at most n states and choose the best one after sorting the child states
	Space Complexity: 
	'''

	def greedy( self,time_allowance=60.0 ):
		#using these parameters in my b&b algorithm results in a greedy algorithm!
		counter = 0
		cities = self._scenario.getCities()
		ncities = len(cities)
		solver = TSPSolver.BranchAndBoundSolver(self, InitialStrategies.NO_INITIAL, 1, 1, 0, 1)
		solver.rand_seed = counter
		result =  solver.solve()
		time_used = result['time']
		while time_used < time_allowance and result['cost'] == math.inf and counter < ncities:
			counter += 1
			solver.rand_seed = counter
			#We pass in the difference of time allowance and time used so that we don't go over time
			result = solver.solve(time_allowance - time_used)
			time_used += result['time']
		return result
	
	''' 
	Description: This function uses the BranchAndboundSolver with its default settings to solve the problem. The default settings are static variables in the class 
	and can be tuned there. 

	Time Complexity: 
	Space Complexity: 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		#create an instance of our branch and bound solver inner class object and call solve on it
		#It needs to use an inner class because the object needs to be able to access this class' data

		solver = TSPSolver.BranchAndBoundSolver(self)
		return solver.solve(time_allowance)

		
	def fancy( self,time_allowance=60.0 ):
		pass
	
	'''
	Because we are using so many different ways of solving and each way uses different data structures and techniques, it is more organized to use inner-classes

	This class is what is used for the BranchAndBound algorithm. It has hyper-parameters that can be tuned to find the balance between quickly finding a solution
	and finding a more optimal one. 
	'''
	class BranchAndBoundSolver(object):

		#These default values are the values that can be fine tuned to improve the not initial bssf pruning
		#These are used when TSPSolver calls branchAndBound() because nothing is passed into init()
		DEFAULT_MAX_CHILD_STATES = 5
		DEFAULT_MAX_QUEUE_SIZE = 20
		DEFAULT_DEPTH_IMPORTANCE = 1
		DEFAULT_BOUND_IMPORTANCE = .8

		#The greedy algorithm seems to work best for the initial algorithm because it's really fast. However, I didn't spend much time tuning the RESTRICTED* 
		#parameters
		DEFAULT_INITIAL_STRATEGY = InitialStrategies.GREEDY

		#These are the values that can be fine-tuned for finding the initial BSSF if the initial strategy is RESTRICTED_BRANCH_AND_BOUND
		#These are used when the instance of BrandAndBoundSolver in TSPSolver uses BranchAndBoundSolver to find the initial bssf. This uses more restrictive
		#parameters to find a solution faster. These are passed into init inside of the find_initial_bssf() function if the initial_strategy is 
		# RESTRICTED_BRANCH_AND_BOUND
		RESTRICTED_MAX_CHILD_STATES = 5
		RESTRICTED_MAX_QUEUE_SIZE = 10
		RESTRICTED_DEPTH_IMPORTANCE = 1
		RESTRICTED_BOUND_IMPORTANCE = .5

		#This is the amount of time used to find the initial bssf
		#We need this because sometimes the search space is very large for the initial bssf and we need a signal to send what we've already found in the more
		#restricted search space. 
		INITIAL_TIME_ALLOWANCE = 15.0


		def __init__(self, solver_instance, initial_strategy=DEFAULT_INITIAL_STRATEGY, max_child_states=DEFAULT_MAX_CHILD_STATES, \
			max_queue_size=DEFAULT_MAX_QUEUE_SIZE, depth_importance=DEFAULT_DEPTH_IMPORTANCE, bound_importance=DEFAULT_BOUND_IMPORTANCE):
			#We passed in the instance of the outer class so we can access everything via the 'outer' object variable
			self.outer = solver_instance
			#This tells us how we are going to get our bssf
			self.initial_strategy = initial_strategy
			#We will use a specialized heap queue for the problem
			self.priority_queue = HeapPriorityQueue()
			#This is for debugging purposes. This determines which city we start on
			self.rand_seed = 10
			#This is holds the start city index so that we don't generate states that point to this index intil we are at a depth of ncities - 1
			self.start_city_ind = None
			#This is used to tell us at what depth we can include the start city in our calculations
			self.n_cities = None
			self.cities = None

			'''
			The folowing are hyper-parameters that help with pruning in the branch and bound algorithm. 
			They depth_importance and bound_importance contribute to priority_key value assigned to a state P. The function to calculate priority_key is below:

			priority_key = ((bound_importance * lower_bound(P)) + 1) / ((depth_importance * curr_state_depth) + 1)

			We need to add 1 to both the numerator and denominator so that if the importance constants are set to 0, it won't lead to undefined or 0 values. 
			'''
		
			#max_child_states: This is the max number of child states that we can add to the queue at each expansion. This promotes more drilling when lower.
			# a vlue of infinity would give this parameter no effect. 
			self.max_child_states = max_child_states

			#max_queue_size: This is the max size of the heap so that we only keep the most promising states and don't inflate the size of memory used too much. The lower this is, the more memory we save and 
			#would help to promote drilling. 
			# this hyper-parameter has no effect when set to infinity. 
			self.max_queue_size = max_queue_size

			#depth_importance: this hyper-parameter can be tuned to help us determine the effect of depth on the priority key. Must be between 0 and 1.
			#a value of 1 will make it so that there is full effect of depth on the priority key.
			#a value of 0 will make it so that there is no effect of depth on the priority key. At that point, the only determining factor on priority key would be bound.
			#The higher the constant, the more of an effect depth has on the priority key value.
			self.depth_importance = depth_importance

			#bound_importance: This hyper-parameter can be tuned to help us determine the effect of bound on the priority key
			# a value of 1 will make it so that there is full effect of the constant on the priority key.
			# a value of 0 will make it so that there is no effect of bound on the priority key. At that point the only determining factor in the priority key would be depth
			# If we have a high constant value, a high bound value will drive the key value down more. A low constant value would make it so that a high bound value wouldn't affect the priority key as much and 
			# depth would play a more important role at that point. 
			self.bound_importance = bound_importance

		'''
		This function calculates the key that is used to determine where it is in the priority queue

		Time complexity: O(1)
		Space complexity: O(1)
		'''
		def calculate_priority_key(self, state):
			state.priority_key = ((self.bound_importance * state.lower_bound) + 1) /((self.depth_importance * state.depth) + 1)

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
		def create_child_states(self, parent_state):
			
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
				if ((parent_state.depth < (self.n_cities - 1)) and (col != self.start_city_ind)) or ((parent_state.depth == (self.n_cities - 1)) \
					and (col == self.start_city_ind)):
					child = self.create_child_state(parent_state, col)
					children.append(child)
					# Then we know this is a test. We can just add all of the states to the array because our test only includes 5 cities which is bounded.
					# and barely takes up any memory
					if self.test_states != None: self.test_states.append(child)

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
		def create_child_state(self, parent_state, to_column):

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
			from_city = self.cities[child_state.prev_visited[-1]]
			to_city = self.cities[child_state.curr_ind]
			child_state.curr_dist += from_city.costTo(to_city)

			#set is_solution if needed
			if child_state.depth == self.n_cities and child_state.curr_ind == self.start_city_ind:

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
		def create_start_state(self):
			state = State()
			#loop through all of the 'from' rows. 
			for i in range(self.n_cities):
				#loop through all of the 'to' coloumns
				for j in range(self.n_cities):
					#Get the cities so we can see if an edge between them exists
					from_city = self.cities[i]
					to_city = self.cities[j]
					cost = from_city.costTo(to_city)
					# if an edge from city at index i to city at index j exists, add it to our 'matrix'
					if cost != math.inf:
						#We populate it as a dictionary of dictionaries for quick look up and easy deletions
						#Each level of the matrix dictionary acts like a set of [rows] and [columns] that we can iterate through when reducing it
						if i not in state.matrix:
							state.matrix[i] = {}
						state.matrix[i][j] = cost

			state.curr_ind = self.start_city_ind
			state.depth = 0
			state.curr_dist = 0
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
		def check_if_solution(self, state):
			if state.is_solution:
				solution = self.create_solution(state)
				if solution.cost < self.outer.bssf_dist:
					return solution
			return None

		'''
		Description: This is called when we have found a solution and we need to create a TSPSolution object 
		
		Time complexity: O(n)
		- We have to loop through n cities to create the solution

		Space complexity: O(n)
		- The solution object grows by a factor of n because it holds the edges between n cities
		'''
		def create_solution(self, state):
			city_list = []
			for i in state.prev_visited:
				city_list.append(self.cities[i])
			return TSPSolution(city_list)
		
		'''
		Description: This function is called at the beginning of solve() to help improve pruning and get to a more optimal solution quicker by setting a bssf
		early on. 
		A smaller time_allowance can be passed in if you don't want to spend a lot of time finding an initial solution. 
		This function calls solver functions and the bssf is storeed in the outer TSPSolver class so we can access them from within. Because of this, 
		we don't need to return any solution objects. 

		The time and space complexities depend on the hyper-parameters selected. See the actual solve() functions for each strategy to see their respective
		time and space complexities.

		'''
		def find_initial_bssf(self, time_allowance=60.0):
			#This is used to increment the totals for the solution as needed. I am assuming that we don't increment total solutions with the initial solution, 
			#but we do increment all of the other totals variables (cost, count, max, and pruned).
			result = None
			#This is where we go through all the implemented algorithms that can be used to find an initial bssf and find the one that is assigned for this instance.
			if self.initial_strategy == InitialStrategies.GREEDY:
				#access the greedy algorithm implemented in the outer class
				result = self.outer.greedy()

			elif self.initial_strategy == InitialStrategies.RESTRICTED_BRANCH_AND_BOUND:
				#create a new branch and bound solver instance and find a solution with more restricted parameters
				#We don't want these to be changable by the user because we will do the fine tuning ourselves
				#Because we have access to the outer class and the outer class has access to the inner class, we call the constructor via the outer
				#class
				#we can pass the same outer class in as the solver_instance

				#We create a new solver instead of just calling solve because we need to be able to skip over the find initial bssf function call
				#It also keeps code a little more independent in my opinion
				solver = self.outer.BranchAndBoundSolver(self.outer, None, self.RESTRICTED_MAX_CHILD_STATES, self.RESTRICTED_MAX_QUEUE_SIZE, \
					self.RESTRICTED_DEPTH_IMPORTANCE, self.RESTRICTED_BOUND_IMPORTANCE)
				#we pass in the INITIAL_TIME_ALLOWANCE variable to give ourselves less time to find a solution 
				result = solver.solve(self.INITIAL_TIME_ALLOWANCE)
				
			elif self.initial_strategy == InitialStrategies.NO_INITIAL:
				return math.inf
			else:
				print("Error in find_initial_bssf!")
				return math.inf
			

			#If we get to this point, we know that result is not null, so we can use it as needed

			#increment the totals variables
			self.total += result['total']
			self.count += result['count']
			self.pruned += result['pruned']
			if result['max'] > self.max:
				self.max = result['max']

			#return the cost found
			return result['cost']

		'''
		This function is used to create a results dictionary

		Time complexity: O(1)
		Space complexity: O(1)
		'''
		def create_results(self):
			results = {}
			results['cost'] = self.outer.bssf_dist
			results['time'] = self.end_time - self.start_time
			results['count'] = self.count
			results['soln'] = self.outer.bssf
			results['max'] = self.max
			results['total'] = self.total
			results['pruned'] = self.pruned
			return results

		'''
		Description: This is the function that is called to solve the TSP problem with the parameters specified in the constructor. 
		Time allowance is the amount of time that is allotted to find the best solution possible.
		is_test returns an extra array of states that is used in our testing suite.

		In our discussion of complexities, we will use the following facts
		- n = number of cities in TSP problem
		- k = max cache size
		- p = max child states

		Time complexity: O(n^4 * !n) at worst
		- Finding the initial bssf should be a lot faster, so it won't be significant in our analysis
		- There are n levels and every node except for leaf nodes is expanded to n-1 children. This results in !n children at most which means we go through
		the while loop at most O(!n)
		- within our while loop we expand the state into child states
		- each expansion takes O(n^4) time because create_child_states is called
		- insertions and deletions from the queue are insignificant compared to the time it takes to expand children, but each are O(logn)
		- Something smaller + (!n * (n^4 + (3 * logn)) = O(n^4 * !n)
		However, if I use a smaller p (not infinity), the time complexity changes
		- The number of tree nodes decreases to O(p^n)
		- so the time complexity becomes O(n^4 * p^n)
		- If we decrease the max cache size, a lot of the states we don't care about (the ones with low priority) wont even be stored and that will 
		- further decrease our time complexity. 

		Space complexity: O(n^2 * !n)
		- There are at most !n states
		- Each state takes O(n^2) space because of the matrix
		However, if I use a smaller k (not infinity) our space complexity now becomes O(k * n^2) because our queue doesn't hold more than k states

		'''
		def solve(self, time_allowance=60.0, is_test=False):

			random.seed(self.rand_seed)
			self.cities = self.outer._scenario.getCities()
			self.n_cities = len(self.cities)

			self.start_time = time.time()

			#reset the totals variables needed for solving this problem
			self.max = 0
			self.count = 0
			self.pruned = 0
			# set total to 0 because this is a self.count of the number of child states and not the total number of states
			self.total = 0
			#if it is a test, we want to keep track of the important states to return
			self.test_states = None
			if is_test: self.test_states = []

			# select a random city index to start at (second param is inclusive)
			self.start_city_ind = random.randint(0, self.n_cities - 1)

			# Generate the initial state, calculate its priority key (lower bound included), and add it to the queue
			start_state = self.create_start_state()
			if is_test: self.test_states.append(start_state)

			#We create a data tuple to pass into the priority queue instead of just the state object because our priority queue is generic and needs the following parameters
			#(key, object_id, object)
			data = (start_state.priority_key, start_state.state_id, start_state)
			self.priority_queue.insert(data)

			# either set bssf to infinity or use an algorithm like greedy to find an initial solution
			# don't self.count this as a solution found
			#If there is an initial strategy specified, this is the first call, if it isn't, that means that this call to solve is being used as a bssf 
			#itself.
			if self.initial_strategy != None:
				#This is the first call so we set the bssf to None
				self.outer.bssf = None
				self.outer.bssf_dist = self.find_initial_bssf()
			elif self.outer.bssf != None:
				self.outer.bssf_dist = self.outer.bssf.cost
			else:
				self.outer.bssf_dist = math.inf

			# while the queue is not empty and we are in the time bound and there is a solution:
			while not self.priority_queue.is_empty() and time.time() - self.start_time < time_allowance and \
				start_state != None:

				#take the state at the top of the queue
				state = self.priority_queue.delete_min()

				#If it isn't small enough, we can just skip it and increment pruned
				if state.priority_key > self.outer.bssf_dist:
					self.pruned += 1
					continue

				#expand that state into child states
			#	This function returns a sorted list of all of the child states 
				child_states = self.create_child_states(state)

				#increment self.total as needed
				self.total += len(child_states)

				#for each child state up to the max children (This allows us to go ghrough the best children first because the list is sorted by key)
				for i in range(len(child_states)):

					#Check the case where we've already checked the max number of child states 
					if i > self.max_child_states:
						#Add the number of states that we aren't visiting
						self.pruned += len(child_states) - i
						break

					#if the child state is a solution, update bssf and increment self.count
					#No need to add the solution to the queue
					solution = self.check_if_solution(child_states[i])
					if solution != None:
						self.outer.bssf = solution
						self.outer.bssf_dist = self.outer.bssf.cost
						self.count += 1
						#If this is being used to find an initial bssf we return on the first solution found
						if self.initial_strategy == None: 
							self.end_time = time.time()
							return self.create_results()


					#else if child_state.lower_bound < bssf: 
					elif not child_states[i].is_solution and child_states[i].lower_bound < self.outer.bssf_dist:
						#if queue size is maxed:
						#we delete one because we will then insert one after. We will never exceed the max this way. 
						#We know we can delete here because we will always be insertin in this block
						if self.priority_queue.size() >= self.max_queue_size:
							self.priority_queue.delete_leaf()
							self.pruned += 1

						data = (child_states[i].priority_key, child_states[i].state_id, child_states[i])
						self.priority_queue.insert(data)

						#if queue size > max: update max
						if self.priority_queue.size() > self.max:
							self.max = self.priority_queue.size()

					#else: increment self.pruned because we don't do anything with this state because it isn't good enough for our standards
					else:
						self.pruned += 1


			self.end_time = time.time()

			#Set all the needed stats in the result object
			results = self.create_results()
			
			if is_test:
				return results, self.test_states

			#otherwise just return results
			return results



