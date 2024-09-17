# Imports
from __future__ import annotations  # Needed for typing Node class

import warnings
from typing import Any

import binarytree
import heapdict


# Node class (do not change)
class Node:
    def __init__(self, data: Any = None, next: None | Node = None):
        self.data = data
        self.next = next


# Add your implementations below


class Stack:
    """Last In First Out (LIFO)"""
    
    def __init__(self):
        """Initialize stack object, with head attribute"""
        self.head = None

    def push(self, data: Any) -> None:
        """Add new node with data to stack"""
        node = Node()
        node.data = data
        if self.head != None:
            node.next = self.head
            self.head = node
        else:
            self.head = node
            node.next = None    

    def peek(self) -> Node | None:
        """Return data from node on top of stack, without changing stack"""
        return self.head.data

    def pop(self) -> Node:
        """Remove last added node and return its data"""
        if self.head == None:
            raise(IndexError)
        data = self.head.data
        self.head = self.head.next
        return data


class Queue:
    """First In First Out (FIFO)"""
    
    def __init__(self):
        """Initialize queue object with head and tail"""
        self.head = None
        self.tail = None

    def enqueue(self, data: Any) -> None:
        """Add node with data to queue"""
        node = Node()
        node.data = data
        if (self.head == None) and (self.tail == None):
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        


    def peek(self) -> Node | None:
        """Return data from head of queue without changing the queue"""
        return self.head.data

    def dequeue(self) -> Node:
        """Remove node from head of queue and return its data"""
        if self.head == None:
            raise(IndexError)
        data = self.head.data
        self.head = self.head.next
        return data


class EmergencyRoomQueue:
    ##### https://www.geeksforgeeks.org/priority-queue-using-queue-and-heapdict-module-in-python/ ###
    def __init__(self):
        """Initialize emergency room queue, use heapdict as property 'queue'"""
        self.queue = heapdict.heapdict()

    def add_patient_with_priority(self, patient_name: str, priority: int) -> None:
        """Add patient name and priority to queue

        # Arguments:
        patient_name:   String with patient name
        priority:       Integer. Higher priority corresponds to lower-value number.
        """
        # node = Node()
        # node.patient_name = patient_name
        # node.priority = priority
        self.queue[patient_name] = priority

    def update_patient_priority(self, patient_name: str, new_priority: int) -> None:
        """Update the priority of a patient which is already in the queue

        # Arguments:
        patient_name:   String, name of patient in queue
        new_priority:   Integer, updated priority for patient

        """
        self.queue[patient_name] = new_priority

    def get_next_patient(self) -> str:
        """Remove highest-priority patient from queue and return patient name

        # Returns:
        patient_name    String, name of patient with highest priority
        """
        if (list(self.queue.items()) == []) or (self.queue == None):
            raise(IndexError)
        return self.queue.popitem()[0]


class BinarySearchTree:
    ### https://pypi.org/project/binarytree/ ###
    def __init__(self, root: binarytree.Node | None = None):
        """Initialize binary search tree

        # Inputs:
        root:    (optional) An instance of binarytree.Node which is the root of the tree

        # Notes:
        If a root is supplied, validate that the tree meets the requirements
        of a binary search tree (see property binarytree.Node.is_bst ). If not, raise
        ValueError.
        """
        if not binarytree._is_bst(root):
            raise(ValueError)
        self.root = root

    def insert(self, value: float | int) -> None:
        """Insert a new node into the tree (binarytree.Node object)

        # Inputs:
        value:    Value of new node

        # Notes:
        The method should issue a warning if the value already exists in the tree.
        See https://docs.python.org/3/library/warnings.html#warnings.warn
        In the case of duplicate values, leave the tree unchanged.
        """
        if self.root == None:
            node = binarytree.Node(value=value)
            self.root = node
        else:
            flag = False
            current = self.root
            while not flag:
                if value > current.value:
                    if current.right is not None:
                        current = current.right
                    else:
                        node = binarytree.Node(value=value)
                        current.right = node
                        flag = True
                elif value < current.value:
                    if current.left is not None:
                        current = current.left
                    else:
                        node = binarytree.Node(value=value)
                        current.left = node
                        flag = True
                else:
                    raise(warnings.warn('Value already exists in tree (duplicate)'))
                    flag = True
                
        

    def __str__(self) -> str | None:
        """Return string representation of tree (helper function for debugging)"""
        if self.root is not None:
            return str(self.root)
