# Imports
import binarytree
import data_structures as ds
import heapdict
import pytest


# Stack
def test_stack_init():  # 1p
    stack = ds.Stack()
    assert stack.head is None


def test_stack_push():  # 2p
    stack = ds.Stack()
    stack.push("A")
    stack.push("B")
    stack.push("C")
    assert stack.head.data == "C"
    assert stack.head.next.data == "B"
    assert stack.head.next.next.data == "A"


def test_stack_peek():  # 1p
    stack = ds.Stack()
    stack.push("A")
    stack.push("B")
    assert stack.peek() == "B"


def test_stack_pop():  # 2p
    stack = ds.Stack()
    stack.push("A")
    stack.push("B")
    stack.push("C")
    assert stack.pop() == "C"
    assert stack.pop() == "B"
    assert stack.pop() == "A"


def test_stack_pop_empty():  # 1p
    stack = ds.Stack()
    with pytest.raises(IndexError):
        stack.pop()


# Queue
def test_queue_init():  # 1p
    queue = ds.Queue()
    assert queue.head is None
    assert queue.tail is None


def test_queue_enqueue():  # 2p
    queue = ds.Queue()
    queue.enqueue("A")
    queue.enqueue("B")
    queue.enqueue("C")
    assert queue.head.data == "A"
    assert queue.tail.data == "C"


def test_queue_peek():  # 1p
    queue = ds.Queue()
    queue.enqueue("A")
    queue.enqueue("B")
    assert queue.peek() == "A"


def test_queue_dequeue():  # 2p
    queue = ds.Queue()
    queue.enqueue("A")
    queue.enqueue("B")
    queue.enqueue("C")
    assert queue.dequeue() == "A"
    assert queue.dequeue() == "B"
    assert queue.dequeue() == "C"


def test_queue_dequeue_empty():  # 1p
    queue1 = ds.Queue()
    with pytest.raises(IndexError):
        queue1.dequeue()
    queue2 = ds.Queue()
    queue2.enqueue("X")
    _ = queue2.dequeue()
    with pytest.raises(IndexError):
        queue2.dequeue()


# Priority queue
def test_erqueue_init():  # 1p
    erqueue = ds.EmergencyRoomQueue()
    assert isinstance(erqueue.queue, heapdict.heapdict)


def test_erqueue_add_patient():  # 1p
    erqueue = ds.EmergencyRoomQueue()
    erqueue.add_patient_with_priority("Bob", 3)
    assert erqueue.queue["Bob"] == 3


def test_erqueue_update_patient_priority():  # 1p
    erqueue = ds.EmergencyRoomQueue()
    erqueue.add_patient_with_priority("Bob", 3)
    erqueue.update_patient_priority("Bob", 2)
    assert erqueue.queue["Bob"] == 2


def test_erqueue_get_patient():  # 1p
    erqueue = ds.EmergencyRoomQueue()
    erqueue.add_patient_with_priority("Bob", 3)
    erqueue.add_patient_with_priority("Shabana", 2)
    erqueue.add_patient_with_priority("Thu", 5)
    assert erqueue.get_next_patient() == "Shabana"


def test_erqueue_add_update_get_patients():  # 1p
    erqueue = ds.EmergencyRoomQueue()
    erqueue.add_patient_with_priority("Bob", 3)
    erqueue.add_patient_with_priority("Shabana", 2)
    erqueue.update_patient_priority("Bob", 1)
    erqueue.add_patient_with_priority("Thu", 5)
    erqueue.update_patient_priority("Shabana", 8)
    assert erqueue.get_next_patient() == "Bob"
    assert erqueue.get_next_patient() == "Thu"
    assert erqueue.get_next_patient() == "Shabana"


def test_erqueue_get_from_empty_queue():  # 1p
    erqueue = ds.EmergencyRoomQueue()
    with pytest.raises(IndexError):
        erqueue.get_next_patient()


# Binary search tree
def test_bst_init():  # 4
    bst1 = ds.BinarySearchTree()
    assert bst1.root is None
    node1 = binarytree.Node(3)
    node2 = binarytree.Node(7)
    bst2 = ds.BinarySearchTree(binarytree.Node(5, left=node1, right=node2))
    assert bst2.root.value == 5
    with pytest.raises(ValueError):
        _ = ds.BinarySearchTree(
            binarytree.Node(5, left=node2, right=node1)
        )  # Supplied node is not root of BST


def test_bst_insert():  # 4
    numbers = [7, 3, 8, 9, 5, 1, 4]
    bst = ds.BinarySearchTree()
    for number in numbers:
        bst.insert(number)
    assert bst.root.value == 7
    assert bst.root.left.value == 3
    assert bst.root.left.left.value == 1
    assert bst.root.left.right.value == 5
    assert bst.root.left.right.left.value == 4
    assert bst.root.right.value == 8
    assert bst.root.right.right.value == 9
