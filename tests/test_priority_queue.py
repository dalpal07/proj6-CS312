import unittest
import os
import sys

root_os_location = os.path.join(os.getcwd())
sys.path.insert(0, root_os_location)

from PriorityQueue import HeapPriorityQueue

class TestPriorityQueue(unittest.TestCase):
    #This is called at instantiation
    @classmethod
    def setUpClass(cls):
        #We create all the data that we want to insert
        #The priority queue takes tuples of size 3 that has the (key, object_id, object)
        #The queue prioritizes smaller keys
        d1 = (1, 11, 21)
        d2 = (2, 12, 22)
        d3 = (3, 13, 23)
        d4 = (4, 14, 24)
        d5 = (5, 15, 25)
        d6 = (6, 16, 26)
        d7 = (7, 17, 27)
        d8 = (8, 18, 28)
        d9 = (9, 19, 29)
        cls.data = [d1, d9, d5, d3, d2, d4, d7, d8, d6]
    
    def setUp(self):
        self.queue = HeapPriorityQueue()

        for e in self.data:
            self.queue.insert(e)
    
    #TODO: write pass and fail for each public method
    
    def testInsert(self):
        self.assertEqual(len(self.data), self.queue.size())
    
    def testDeleteMin(self):
        min = self.queue.delete_min()
        self.assertEqual(min, 21, "first wrong")
        min = self.queue.delete_min()
        self.assertEqual(min, 22, "second wrong")
        min = self.queue.delete_min()
        min = self.queue.delete_min()
        min = self.queue.delete_min()
        min = self.queue.delete_min()
        min = self.queue.delete_min()
        min = self.queue.delete_min()
        min = self.queue.delete_min()
        self.assertEqual(min, 29, "last wrong")
    
    def testInitialLevels(self):
        self.assertEqual(21, self.queue.heap[0][2], "root not correct")

        test = False
        el = self.queue.heap[1][2]
        if el == 22 or el == 23:
            test = True
        self.assertTrue(test, "error in first level")

        test = False
        el = self.queue.heap[3][2]
        if el == 24 or el == 25 or el == 26 or el == 27:
            test = True
        self.assertTrue(test, "error in second level")
    
        test = False
        el = self.queue.heap[7][2]
        if el == 28 or el == 29:
            test = True
        self.assertTrue(test, "error in third level")
    
    def testLevelsAfterDeletion(self):
        self.queue.delete_min()
        self.assertEqual(22, self.queue.heap[0][2], "root not correct")

        self.assertTrue(self.queue.heap[0][2] < self.queue.heap[1][2], "first level not correct")
        self.assertTrue(self.queue.heap[0][2] < self.queue.heap[2][2], "first level not correct")

        self.assertTrue(self.queue.heap[1][2] < self.queue.heap[3][2], "second level not correct")
        self.assertTrue(self.queue.heap[1][2] < self.queue.heap[4][2], "second level not correct")
        self.assertTrue(self.queue.heap[2][2] < self.queue.heap[5][2], "second level not correct")
        self.assertTrue(self.queue.heap[2][2] < self.queue.heap[6][2], "second level not correct")

        self.assertTrue(self.queue.heap[3][2] < self.queue.heap[7][2], "third level not correct")
    
    def testDeleteLeaf(self):
        self.queue.delete_leaf()
        self.assertEqual(8, self.queue.size(), "error in size after delete_leaf()")

        self.assertEqual(21, self.queue.heap[0][2], "root not correct")

        test = False
        el = self.queue.heap[1][2]
        if el == 22 or el == 23:
            test = True
        self.assertTrue(test, "error in first level")

        test = False
        el = self.queue.heap[3][2]
        if el == 24 or el == 25 or el == 26 or el == 27:
            test = True
        self.assertTrue(test, "error in second level")
    
        test = False
        el = self.queue.heap[7][2]
        if el == 28 or el == 29:
            test = True
        self.assertTrue(test, "error in third level")
    



if __name__=="__main__":
    unittest.main()