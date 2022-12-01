import unittest
import os
import sys
import json

root_os_location = os.path.join(os.getcwd())
sys.path.insert(0, root_os_location)

# Import in the code with the actual implementation
from TSPSolver import *
from TSPClasses import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

SCALE = 1.0
SEED = 20
DIFF = "Hard (Deterministic)"
NPOINTS = 5

def newPoints():	

    data_range = { 'x':[-1.5*SCALE,1.5*SCALE], \
                        'y':[-SCALE,SCALE] }

    seed = SEED
    random.seed( seed )

    ptlist = []
    xr = data_range['x']
    yr = data_range['y']
    npoints = NPOINTS
    while len(ptlist) < npoints:
        x = random.uniform(0.0,1.0)
        y = random.uniform(0.0,1.0)
        if True:
            xval = xr[0] + (xr[1]-xr[0])*x
            yval = yr[0] + (yr[1]-yr[0])*y
            ptlist.append( QPointF(xval,yval) )
    return ptlist

def fill_state_data(state_name, test_data):
    state_data = {}
    state_data['matrix'] = {}
    for row in test_data[state_name]['matrix']:
        state_data['matrix'][int(row)] = {}
        for col in test_data[state_name]['matrix'][row]:
            state_data['matrix'][int(row)][int(col)] = test_data[state_name]['matrix'][row][col]
    state_data['lower bound'] = test_data[state_name]['lower bound']
    state_data['current index'] = test_data[state_name]['current index']
    state_data['priority key'] = test_data[state_name]['priority key']
    return state_data


def prepare_data(test_data):
    initial_state_data = fill_state_data("initial state", test_data)
    child_state_data = fill_state_data("child state", test_data)
    return initial_state_data, child_state_data

def matrices_equal(m1, m2):
    if len(m1) != len(m2):
        return False
    for row in m1:
        if row not in m2:
            return False
        if len(m1[row]) != len(m2[row]):
            return False
        for col in m1[row]:
            if col not in m2[row]:
                return False
            if m1[row][col] != m2[row][col]:
                return False
    return True


class TestBranchAndBound(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        points = newPoints()
        diff = "Hard (Deterministic)"
        scenario = Scenario( city_locations=points, difficulty=diff, rand_seed=SEED )

        #For testing purposes I want to change the tuning variables to be unrestrictive
        #These are static so we can just change them here with out an instance being created
        
        cls.solver = TSPSolver(None)
        cls.solver.setupWithScenario(scenario)
        cls.initial_state_data = None
        cls.child_state_data = None

        try:
            path = os.path.join(root_os_location, 'tests', 'bb_test_data.json')
            with open(path) as f:
                cls.test_data = json.load(f)
        except:
            print("Error loading json file for testing")
        cls.initial_state_data, cls.child_state_data = prepare_data(cls.test_data)

        #We set our own unrestrictive variables by creating our own inner class instance and than passing in our outer class 
        #instance. This is not how you should use the object, but it is how we test it
        cls.bb_solver = TSPSolver.BranchAndBoundSolver(cls.solver, InitialStrategies.NO_INITIAL, \
            math.inf, math.inf, 1.0, 1.0)
        cls.result, cls.states = cls.bb_solver.solve(is_test=True)
        cls.c_state = None
        for state in cls.states[1:]:
            #find the child state at curr_index == 0
            if state.curr_ind == 0:
                cls.c_state = state
                break

    def testNormalSolve(self):
        result = self.solver.branchAndBound()
        self.assertIsNotNone(result['cost'])
    
    def testTestSolve(self):
        #We indicate that this is a test so that it will return states as well
        #For this test, we don't want an initial bssf because it is easier to test this way
        self.assertIsNotNone(self.result['cost'], "error in result")
        self.assertTrue(len(self.states) > 0, "error in states returned")
    
    def testParentReduction(self):
        self.assertEqual(self.initial_state_data['lower bound'], self.states[0].lower_bound, "incorrect bound")
        self.assertEqual(self.initial_state_data['current index'], self.states[0].curr_ind, "incorrect index")
        self.assertTrue(matrices_equal(self.initial_state_data['matrix'], self.states[0].matrix), "incorrect matrix")
    
    def testChildReduction(self):
        self.assertIsNotNone(self.c_state, "no state with correct index found")
        self.assertEqual(self.child_state_data['lower bound'], self.c_state.lower_bound)
        self.assertTrue(matrices_equal(self.child_state_data['matrix'], self.c_state.matrix), "incorrect matrix")
    
    def testPriorityKey(self):
        self.assertEqual(self.initial_state_data["priority key"], self.states[0].priority_key, "incorrect initial priority key")
        self.assertEqual(self.child_state_data["priority key"], self.c_state.priority_key, "incorrect child priority key")

if __name__=="__main__":
    unittest.main()