import unittest
import os
import sys

root_os_location = os.path.join(os.getcwd())
sys.path.insert(0, root_os_location)

from tests.test_branch_and_bound import TestBranchAndBound
from tests.test_priority_queue import TestPriorityQueue

if __name__=="__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestBranchAndBound)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestPriorityQueue)
    alltests = unittest.TestSuite([suite1, suite2])
    runner = unittest.TextTestRunner()
    runner.run(alltests)
