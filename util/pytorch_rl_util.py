import numpy as np


class Node(object):
    """Generic Node class. Used in the implementation of a prioritised replay buffer"""
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def update_key_and_value(self, new_key, new_value):
        self.update_key(new_key)
        self.update_value(new_value)

    def update_key(self, new_key):
        self.key = new_key

    def update_value(self, new_value):
        self.value = new_value

    def __eq__(self, other):
        return self.key == other.key and self.value == other.value


class Deque(object):
    """Generic deque object"""
    def __init__(self, max_size, dimension_of_value_attribute):

        self.max_size = max_size
        self.dimension_of_value_attribute = dimension_of_value_attribute
        self.deque = self.initialise_deque()
        self.deque_index_to_overwrite_next = 0
        self.reached_max_capacity = False
        self.number_experiences_in_deque = 0

    def initialise_deque(self):
        """Initialises a queue of Nodes of length self.max_size"""
        deque = np.array([Node(0, tuple([None for _ in range(self.dimension_of_value_attribute)])) for _ in range(self.max_size)])
        return deque

    def add_element_to_deque(self, new_key, new_value):
        """Adds an element to the deque and then updates the index of the next element to be overwritten and also the
        amount of elements in the deque"""
        self.update_deque_node_key_and_value(self.deque_index_to_overwrite_next, new_key, new_value)
        self.update_number_experiences_in_deque()
        self.update_deque_index_to_overwrite_next()

    def update_deque_node_key_and_value(self, index, new_key, new_value):
        self.update_deque_node_key(index, new_key)
        self.update_deque_node_value(index, new_value)

    def update_deque_node_key(self, index, new_key):
        self.deque[index].update_key(new_key)

    def update_deque_node_value(self, index, new_value):
        self.deque[index].update_value(new_value)

    def update_deque_index_to_overwrite_next(self):
        """Updates the deque index that we should write over next. When the buffer gets full we begin writing over
         older experiences"""
        if self.deque_index_to_overwrite_next < self.max_size - 1:
            self.deque_index_to_overwrite_next += 1
        else:
            self.reached_max_capacity = True
            self.deque_index_to_overwrite_next = 0

    def update_number_experiences_in_deque(self):
        """Keeps track of how many experiences there are in the buffer"""
        if not self.reached_max_capacity:
            self.number_experiences_in_deque += 1


class MaxHeap(object):
    """Generic max heap object"""
    def __init__(self, max_size, dimension_of_value_attribute, default_key_to_use):

        self.max_size = max_size
        self.dimension_of_value_attribute = dimension_of_value_attribute
        self.default_key_to_use = default_key_to_use
        self.heap = self.initialise_heap()

    def initialise_heap(self):
        """Initialises a heap of Nodes of length self.max_size * 4 + 1"""
        heap = np.array([Node(self.default_key_to_use, tuple([None for _ in range(self.dimension_of_value_attribute)])) for _ in range(self.max_size * 4 + 1)])

        # We don't use the 0th element in a heap so we want it to have infinite value so it is never swapped with a lower node
        heap[0] = Node(float("inf"), (None, None, None, None, None))
        return heap

    def update_element_and_reorganise_heap(self, heap_index_for_change, new_element):
        self.update_heap_element(heap_index_for_change, new_element)
        self.reorganise_heap(heap_index_for_change)

    def update_heap_element(self, heap_index, new_element):
        self.heap[heap_index] = new_element

    def reorganise_heap(self, heap_index_changed):
        """This reorganises the heap after a new value is added so as to keep the max value at the top of the heap which
        is index position 1 in the array self.heap"""

        node_key = self.heap[heap_index_changed].key
        parent_index = int(heap_index_changed / 2)

        if node_key > self.heap[parent_index].key:
            self.swap_heap_elements(heap_index_changed, parent_index)
            self.reorganise_heap(parent_index)

        else:
            biggest_child_index = self.calculate_index_of_biggest_child(heap_index_changed)
            if node_key < self.heap[biggest_child_index].key:
                self.swap_heap_elements(heap_index_changed, biggest_child_index)
                self.reorganise_heap(biggest_child_index)

    def swap_heap_elements(self, index1, index2):
        """Swaps the position of two heap elements"""
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]

    def calculate_index_of_biggest_child(self, heap_index_changed):
        """Calculates the heap index of the node's child with the biggest td_error value"""
        left_child = self.heap[int(heap_index_changed * 2)]
        right_child = self.heap[int(heap_index_changed * 2) + 1]

        if left_child.key > right_child.key:
            biggest_child_index = heap_index_changed * 2
        else:
            biggest_child_index = heap_index_changed * 2 + 1

        return biggest_child_index

    def give_max_key(self):
        """Returns the maximum td error currently in the heap. Because it is a max heap this is the top element of the heap"""
        return self.heap[1].key
