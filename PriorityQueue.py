from math import ceil

"""
Description: This is just an interface for the different types of priority queue implementations. 
"""
class PriorityQueue:
    def __init__(self):
        self.key_vals = {}

    #Should return the minimum key in the queue and pop it. 
    def delete_min(self):
        pass

    #Should decrease the value for a key and put that key in the correct positon in the queue
    def decrease_key(self, key):
        pass

    #Creates the queue given a dictionary with initial key value pairs.
    def make_queue(self, key_vals):
        pass

    #Returns whether or not the queue is empty. 
    def is_empty(self):
        pass
        

class HeapPriorityQueue(PriorityQueue):

    def __init__(self):
        super().__init__()
        self.heap = []
        self.positions = {}

    '''
    Description: This deletes the min by popping of the node id in position 0 of our array implementation of a heap. 
    It then takes the last element in our heap and puts it in the first position and then sifts it down into place. 

    Time complexity: O(log(n))
        - I am assuming the pop off the front of the list is O(1) time (assuming python uses a linked list structure). 
        - Moving the last element in the array to the front takes O(1) time
        - We sift the element down at most logn times. 
    Space complexity: O(1)
        - No notable extra space is allocated for this function. 
    '''
    def delete_min(self):
        #The object we care about is at index 2 of the data tuple
        object_to_return = self.heap[0][2] #we will return this later=
        object_id_to_return = self.heap[0][1]
        del self.key_vals[object_id_to_return]
        del self.positions[object_id_to_return]

        #Delete the largest value in the heap
        pos = 0
        # node_id = self.heap.pop(-1)
        data = self.heap.pop(-1)
        #The object being stored is kept at index 2 of the data tuple
        object_id = data[1]

        #If the length of the heap is 0, we can just return
        if len(self.heap) == 0:
            return object_to_return

        key_val = self.key_vals[object_id]
        #Place the last node in the heap at the front
        self.heap[pos] = data
        self.positions[object_id] = pos

        while True:
            #now sift down
            child_one_pos = (pos * 2) + 1
            child_two_pos = (pos * 2) + 2

            #If child_one_pos is out of range we are done sifting 
            if child_one_pos >= len(self.heap) - 1:
                break

            #We know that there is a left child so we instantiate the values
            child_one_object_id = self.heap[child_one_pos][1]
            child_one_key_val =self.key_vals[child_one_object_id]

            #If child_two pos is out of range, we aren't necessarily done because we know child one isn't
            #We have at most one more move at this point and that would be a left sift so we check if we
            #need to do it and then break
            if child_two_pos >= len(self.heap) - 1:
                if key_val > child_one_key_val:
                    pos = self.sift(pos, object_id, child_one_pos, child_one_object_id)  
                break

            #We know there is a left child and we aren't at any base case so we set right child values
            child_two_object_id = self.heap[child_two_pos][1]
            child_two_key_val = self.key_vals[child_two_object_id]

            # We have three posibilities:
            # It's greater than both children
            # It's greater than the left child only
            # It's greater than the right child only

            #As we update values, we want to update the current node position. we dont care about updating the 
            #child info at this point becasue it will be taken care of the next iteration. We also don't need
            # to update the node_id or the val because those will stay constant throught the sifting process
            if key_val > child_one_key_val and key_val > child_two_key_val:
                #We want to make sure that we sift with the min child
                if child_one_key_val <= child_two_key_val:
                    pos = self.sift(pos, object_id, child_one_pos, child_one_object_id)
                else:
                    pos = self.sift(pos, object_id, child_two_pos, child_two_object_id)
            elif key_val > child_one_key_val:
                pos = self.sift(pos, object_id, child_one_pos, child_one_object_id)
            elif key_val > child_two_key_val:
                pos = self.sift(pos, object_id, child_two_pos, child_two_object_id)
            else:
                break
        
        return object_to_return
    
    '''
    Description: this is a function that swithches the elements in the heap and the positions pointers and returns the new position of 
    the element we are sifting

    Time complexity: O(1) 
        - The time this takes doesn't change depending on the size of the input. 
    Space complexity: O(1)
        - The amount of space doesn't change depending on the input
    '''
    def sift(self, pos, object_id, child_pos, child_object_id):
        #TODO: I can do this by only passing in the object ids

        #Swap the parent and child data tuples
        temp = self.heap[pos]
        self.heap[pos] = self.heap[child_pos]
        self.heap[child_pos] = temp
        #Swap the positions pointers
        self.positions[child_object_id] = pos
        self.positions[object_id] = child_pos
        return child_pos 
    

    '''
    Description: This function sumply bubbles the node up if it needs it to satisfy the conditions of the heap. 

    Time complexity: O(log(n))
        - The node moves up at most logn spots in the worst case.
    Space complexity: O(1)
        - Only a constant amount of space is needed to perform this function. 
    '''
    def bubble(self, object_id):
        pos = self.positions[object_id]
        val = self.key_vals[object_id]
        parent_pos = ceil(pos / 2) - 1

        if parent_pos < 0:
            return
        
        #The heap holds a data tuple where the object id is stored at index 1
        parent_object_id = self.heap[parent_pos][1]
        #Now we get the parent priority key value
        parent_key_val = self.key_vals[parent_object_id]
        while val < parent_key_val:
            #we swap the data tuples
            temp = self.heap[parent_pos]
            self.heap[parent_pos] = self.heap[pos]
            self.heap[pos] = temp
            #We swat the positions pointers of the data tuples
            temp = self.positions[object_id]
            self.positions[object_id] = parent_pos
            self.positions[parent_object_id] = pos

            #update local variables for the next iteration of the while loop
            #node_id, pos, val stays the same but parent info changes
            pos = parent_pos
            parent_pos = ceil(pos / 2) - 1
            parent_object_id = self.heap[parent_pos][1]
            parent_key_val = self.key_vals[parent_object_id]
    
    '''
    Description: This is honestly just here to fulfill the spec requirements and is completely unnecesary because I do all the inserting
    in make_queue. This function just adds a node to the end of the queue and then bubbles up. 

    Time complexity: O(log(n))
        - we append to the end of the heap and then bubble up at most log|v| spots
    Space complexity: O(1)
        - The input is constant and the amount of space allocated for that input is constant. 
    '''
    def insert(self, data):
        #extract needed values from data tuple
        priority_key = data[0]
        object_id = data[1]

        #add the entire data tuple to the actual heap because we will need to access the object_id's of parents and children
        self.heap.append(data)
        #add the position of the object to the positions dictionary using the object_id as the dictionary key
        self.positions[object_id] = len(self.heap) - 1
        #add the key value of the object to the key_vals dictionary using the object_id as the dictionary key
        self.key_vals[object_id] = priority_key
        #because it's added to the bottom of the queue, bubble it up as needed
        self.bubble(object_id)
    
    '''
    Description: This function simply checks if the heap is empty and returns true if it is and false otherwise. 

    Space complexity: O(1)
    Time complexity: O(1)
    '''
    def is_empty(self):
        if len(self.heap) == 0:
            return True
        return False
    
    '''
    Description: Isn't it obvious?

    Time complexity: O(1)
    Space complexity: O(1)
    '''
    def size(self):
        return len(self.heap)

    '''
    Description: Deletes a leaf node from the bottom level

    Time complexity: O(1)
    Space complexity: O(1) 
    '''
    def delete_leaf(self):
        object_id = self.heap[-1][1]
        del self.positions[object_id]
        del self.key_vals[object_id]
        del self.heap[-1]



###-------- UNUSED FUNCTIONS FOR LATER USE. IMPLEMENT LATER IF WANTED--------###

    '''
    Description: This function takes all of the keys (node ids) from the table and puts them into the heap. It reserves the first spot
    for the key with a value of 0, but the rest of the ordering doesn't matter because the values of every other key are infinity. 
    We assign the correct positions of the keys in the heap in the postions dictionary. 

    Time complexity: O(|V|) 
        - We loop through all of the keys in key_vals table and put them in the heap. 
        - Determining if the value is 0 or infinity is a constant time operation. 
        - Updating the positions dictionary is a constant time operation. 
    Space complexity: O(|V|)
        - We need a heap of size n and a positions dictionary of size n. 
    
    DON'T NEED THIS FUNCTION FOR THIS LAB! FIX LATER
    '''
    # def make_queue(self, key_vals):
    #     #We know that there will be an value with 0 at the beginning
    #     self.heap.append(0)
    #     self.key_vals = key_vals
    #     counter = 1
    #     for key in key_vals:
    #         if key_vals[key] == 0:
    #             self.heap[0] = key
    #             self.positions[key] = 0
    #         else:
    #             self.heap.append(key)
    #             self.positions[key] = counter
    #             counter += 1
    
    '''
    Description: This function changes the value of a node in the distance table and then bubbles the node up if it needs to be moves. 

    Time complexity: O(log|V|)
        - This is the time complexity of bubble()
    Space complexity: O(1)
        - This is the space complexity of bubble() 
    
    DON'T NEED THIS FUNCTION FOR THIS LAB! FIX LATER
    '''
    # def decrease_key(self, key):
    #     #key is already decreased so all I have to do is bubble the key up
    #     self.bubble(key)