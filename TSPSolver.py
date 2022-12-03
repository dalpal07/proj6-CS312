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
This is the class that is called by the gui to solve the TSP problem specified
'''
class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario

	###---SOVLER FUNCTIONS---###
	
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
		solver = self.GreedySolver(self, 3)
		return solver.solve(time_allowance)
	
	''' 
	Description: This function uses the BranchAndboundSolver with its default settings to solve the problem. The default settings are static variables in the class 
	and can be tuned there. 

	see the BranchAndBoundSolver class for time and space complexity analysis
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		#create an instance of our branch and bound solver inner class object and call solve on it
		#It needs to use an inner class because the object needs to be able to access this class' data

		#use the default settings
		solver = TSPSolver.BranchAndBoundSolver(self)
		return solver.solve(time_allowance)

		
	def fancy( self,time_allowance=60.0 ):
		pass

	###---SHARED FUNCTIONS---###

	'''
	Description: This function is called at the beginning of solve() to help improve pruning and get to a more optimal solution quicker by setting a bssf
	early on. 
	A smaller time_allowance can be passed in if you don't want to spend a lot of time finding an initial solution. 
	This function calls solver functions and the bssf is storeed in the outer TSPSolver class so we can access them from within. Because of this, 
	we don't need to return any solution objects. 

	The time and space complexities depend on the hyper-parameters selected. See the actual solve() functions for each strategy to see their respective
	time and space complexities.

	'''
	def find_initial_bssf(self, initial_strategy):

		#This is used to increment the totals for the solution as needed. I am assuming that we don't increment total solutions with the initial solution, 
		#but we do increment all of the other totals variables (cost, count, max, and pruned).
		result = None
		#This is where we go through all the implemented algorithms that can be used to find an initial bssf and find the one that is assigned for this instance.
		if initial_strategy == InitialStrategies.GREEDY:
			#access the greedy algorithm implemented in the outer class
			result = self.greedy()

		elif initial_strategy == InitialStrategies.RESTRICTED_BRANCH_AND_BOUND:
			#create a new branch and bound solver instance and find a solution with more restricted parameters

			#These are the values that can be fine-tuned for finding the initial BSSF if the initial strategy is RESTRICTED_BRANCH_AND_BOUND
			#These are used when the instance of BrandAndBoundSolver in TSPSolver uses BranchAndBoundSolver to find the initial bssf. This uses more restrictive
			#parameters to find a solution faster. These are passed into init inside of the find_initial_bssf() function if the initial_strategy is 
			# RESTRICTED_BRANCH_AND_BOUND
			RESTRICTED_MAX_CHILD_STATES = 2
			RESTRICTED_MAX_QUEUE_SIZE = 20
			RESTRICTED_DEPTH_IMPORTANCE = 1
			RESTRICTED_BOUND_IMPORTANCE = .8

			solver = self.BranchAndBoundSolver(self, None, RESTRICTED_MAX_CHILD_STATES, RESTRICTED_MAX_QUEUE_SIZE, \
				RESTRICTED_DEPTH_IMPORTANCE, RESTRICTED_BOUND_IMPORTANCE)

			result = solver.solve()
			
		elif initial_strategy == InitialStrategies.NOISY_GREEDY:
			N_CHOICES = 3

			solver = self.GreedySolver(self, N_CHOICES)
			result = solver.solve()
		else:
			print("Error in find_initial_bssf!")
			return math.inf
		

		#return the result so that values can be incremented
		return result


	###---SOLVER CLASSES---###
	
	'''
	Because we are using so many different ways of solving and each way uses different data structures and techniques, it is more organized to use inner-classes
	'''
	class LocalSearchSolver(object):
		def __init__(self, solver_instance):
			self.outer = solver_instance

		def solve(self, time_allowance=60.0):
			pass

	class GreedySolver(object):
		def __init__(self, solver_instance, n=1):
			self.outer = solver_instance
			#This is the number of best children that we randomly choose from to follow
			self.n = n
			self.shared_utils = SharedUtils()
		
		
		def solve(self, time_allowance=60.0):
			np.random.seed(int(time.time()))
			time_allowance = math.inf
			self.start_city_ind = 0
			self.cities = self.outer._scenario.getCities()
			self.n_cities = len(self.cities)
			self.bssf = None
			self.bssf_dist = math.inf

			#reset the totals variables needed for solving this problem
			self.max = 0
			self.count = 0
			self.pruned = 0
			# set total to 0 because this is a self.count of the number of child states and not the total number of states
			self.total = 0

			self.start_time = time.time()
			self.edges = self.shared_utils.get_edges(self.cities)
			result = self.explore(time_allowance)
			time_used = result["time"]

			#This allows us to loop through all the cities because we go through all the start indices
			while time_used < time_allowance and self.start_city_ind < self.n_cities - 1:
				self.start_city_ind += 1
				time_used = time.time() - self.start_time
				result = self.explore(time_allowance - time_used)
				time_used += result['time']
			
			return result

		
		def explore(self, time_allowance):

			curr_city = self.start_city_ind
			depth = 0
			path = []
			solution = None

			# while the queue is not empty and we are in the time bound and there is a solution:
			while time.time() - self.start_time < time_allowance:

				curr_edges = self.edges[curr_city]
				
				#if we are at the last city and there is an edge to the start city, we need to take it
				if depth == self.n_cities - 1:
					#If it's not a solution, the cost will be inf
					p2 = []
					for p in path:
						p2.append(self.cities[p])

					solution = TSPSolution(p2)
					if solution.cost < self.bssf_dist:
						self.bssf_dist = solution.cost
						self.bssf = solution 
					break
				

				#At this point, we know that we won't go to cities that have already been visited  or the start city unless we're in the right depth

				sorted_edges = sorted(curr_edges, key=curr_edges.get)

				#append the current city to the path
				path.append(curr_city)

				#filter sorted edges
				sorted_edges = [e for e in sorted_edges if e not in path]

				#If there are no outgoing edges, return 
				if len(sorted_edges) == 0:
					self.end_time = time.time()
					return self.shared_utils.create_results(self)

				#now set a new curr_city
				curr_city = np.random.choice(sorted_edges[:self.n])
				depth += 1


			self.end_time = time.time()

			#Set all the needed stats in the result object
			return self.shared_utils.create_results(self)

	'''
	This class is what is used for the BranchAndBound algorithm. It has hyper-parameters that can be tuned to find the balance between quickly finding a solution
	and finding a more optimal one. 
	'''
	class BranchAndBoundSolver(object):

		#These default values are the values that can be fine tuned to improve the not initial bssf pruning
		#These are used when TSPSolver calls branchAndBound() because nothing is passed into init()
		DEFAULT_MAX_CHILD_STATES = 5
		DEFAULT_MAX_QUEUE_SIZE = math.inf
		DEFAULT_DEPTH_IMPORTANCE = 1
		DEFAULT_BOUND_IMPORTANCE = .5

		#The greedy algorithm seems to work best for the initial algorithm because it's really fast. However, I didn't spend much time tuning the RESTRICTED* 
		#parameters
		DEFAULT_INITIAL_STRATEGY = InitialStrategies.GREEDY

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

			self.cities = None

			#PartialPathUtils contains needed functions for the partial path algorithm
			self.shared_utils = SharedUtils(self.calculate_priority_key)


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
		This function calculates the key that is used to determines how child states are sorted

		Time complexity: O(1)
		Space complexity: O(1)
		'''
		def calculate_priority_key(self, state):
			state.priority_key = ((self.bound_importance * state.lower_bound) + 1) /((self.depth_importance * state.depth) + 1)


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
			self.bssf = None
			self.bssf_dist = math.inf

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
			start_state = self.shared_utils.create_start_state(self.cities, self.start_city_ind)
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
				result = self.outer.find_initial_bssf(self.initial_strategy)
				if result != None and result != math.inf:
					self.bssf_dist = result['cost']
					self.bssf = result['soln']
					self.total += result['total']
					self.pruned += result['pruned']
					if result['max'] > self.max:
						self.max = result['max']

			# while the queue is not empty and we are in the time bound and there is a solution:
			while not self.priority_queue.is_empty() and time.time() - self.start_time < time_allowance and \
				start_state != None:

				#take the state at the top of the queue
				state = self.priority_queue.delete_min()

				#If it isn't small enough, we can just skip it and increment pruned
				if state.lower_bound > self.bssf_dist:
					self.pruned += 1
					continue

				#expand that state into child states
			#	This function returns a sorted list of all of the child states 
				child_states = self.shared_utils.create_child_states(state, self.cities, self.start_city_ind, self.test_states)

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
					solution = self.shared_utils.check_if_solution(child_states[i], self.bssf_dist, self.cities)
					if solution != None:
						self.bssf = solution
						self.bssf_dist = solution.cost
						self.count += 1
						#If this is being used to find an initial bssf we return on the first solution found
						if self.initial_strategy == None: 
							self.end_time = time.time()
							return self.shared_utils.create_results(self)


					#else if child_state.lower_bound < bssf: 
					elif not child_states[i].is_solution and child_states[i].lower_bound < self.bssf_dist:
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
			results = self.shared_utils.create_results(self)
			
			if is_test:
				return results, self.test_states

			#otherwise just return results
			return results


