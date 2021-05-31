import numpy as np


class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        return self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)


def test_stack():
    test_items = ['dog', 4, True, 'cat', 'book']
    stk = Stack()

    for item in test_items:
        print('Push item: {}'.format(item))
        stk.push(item)

    print('Is empty? {}, Size: {}'.format(stk.is_empty(), stk.size()))

    while stk.size():
        print('Pop item: {}'.format(stk.pop()))


def brackets_detector(strs):
    start_s = '({['
    end_s = ')}]'

    balance = True
    stk = Stack()
    for s in strs:
        if s in start_s:
            stk.push(s)
        else:
            if stk.size():
                a = stk.pop()
                if start_s.index(a) != end_s.index(s):
                    balance = False
                    break
            else:
                balance = False
                break

    if stk.size():
        balance = False

    print('Input string: {} is balance: {}'.format(strs, str(balance)))
    return balance


def decimal_to_binary(decimal_num):
    bin_str = ''

    stk = Stack()
    while decimal_num > 0:
        resd = decimal_num % 2
        stk.push(resd)
        decimal_num = decimal_num // 2

    while not stk.is_empty():
        bin_str += str(stk.pop())

    print('Decimal num: {}, Binary: {}'.format(decimal_num, bin_str))
    return bin_str


def decimal_to_any_scale(decimal_num, base):
    digits = '0123456789ABCDEF'
    bin_str = ''
    decimal_num_1 = decimal_num

    stk = Stack()
    while decimal_num > 0:
        resd = decimal_num % base
        stk.push(resd)
        decimal_num = decimal_num // base

    while not stk.is_empty():
        bin_str += str(digits[stk.pop()])

    print('Decimal num: {}, Scale-{}: {}'.format(decimal_num_1, base, bin_str))
    return bin_str


class Queue():
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.insert(0, item)

    def pop(self):
        return self.items.pop()

    def is_empty(self):
        return self.items == []

    def size(self):
        return len(self.items)


def test_queue():
    test_items = ['dog', 4, True, 'cat', 'book']

    queue = Queue()

    for item in test_items:
        print('Push item: {}'.format(item))
        queue.push(item)

    print('Is empty? {}, Size: {}'.format(queue.is_empty(), queue.size()))

    while queue.size():
        print('Pop item: {}'.format(queue.pop()))


def transfer_potato():
    # 击鼓传花
    persons = ['Bill', 'David', 'Susan', 'Jane', 'Kent', 'Brad']

    pass_times = 7

    queue = Queue()
    for p in persons:
        queue.push(p)

    for i in range(pass_times):
        queue.push(queue.pop())

    unlucky_baby =  queue.pop()
    print('Unlucky baby is: {}'.format(unlucky_baby))
    return unlucky_baby


class BiQueue():
    def __init__(self):
        self.queue = []

    def is_empty(self):
        return self.queue == []

    def push_front(self, item):
        self.queue.append(item)

    def pop_front(self):
        return self.queue.pop(0)

    def push_back(self, item):
        self.queue.insert(0, item)

    def pop_back(self):
        return self.queue.pop()

    def size(self):
        return len(self.queue)


def test_biqueue():
    test_items = ['dog', 4, True, 'cat', 'book']

    queue = BiQueue()

    for item in test_items:
        direct = np.random.choice([0, 1])

        if direct:
            queue.push_front(item)
            print('Bi-queue: {}'.format(queue.queue))
        else:
            queue.push_back(item)
            print('Bi-queue: {}'.format(queue.queue))


def pal_checker(strs):
    # 回文检测
    queue = BiQueue()
    for s in strs:
        queue.push_front(s)
    print('Queue is: {}'.format(queue.queue))

    is_pal = True
    while queue.size() > 1 and not queue.is_empty():
        front = queue.pop_front()
        back = queue.pop_back()
        if back != front:
            is_pal = False
            break
    print('IS pal: {}'.format(str(is_pal)))
    return is_pal


class LinkedListNode():
    def __init__(self, item):
        self.item = item
        self.next = None
        self.pre = None

    def get_data(self):
        return self.item

    def get_next(self):
        return self.next

    def get_pre(self):
        return self.pre

    def set_data(self, d):
        self.item = d

    def set_next(self, n):
        self.next = n

    def set_pre(self, p):
        self.pre = p


def test_linked_list():
    head = LinkedListNode(None)
    node_1 = LinkedListNode('2')
    node_2 = LinkedListNode('hello')
    node_3 = LinkedListNode('4')

    head.set_next(node_1)
    node_1.set_next(node_2)
    node_2.set_next(node_3)

    # traversing
    def traversing(head):
        start = head
        data = [start.get_data()]
        while start.get_next():
            data.append(start.get_next().get_data())
            start = start.get_next()
        print(data)
    traversing(head)

    # insert
    def insert(node, item):
        if node is None:
            print('ERROR: Node is None')
            return -1
        else:
            new_node = LinkedListNode(item)
            new_node.set_next(node.get_next())
            node.set_next(new_node)

    insert(head, 'new')
    traversing(head)
    insert(node_2, 'new_1')
    traversing(head)
    insert(node_3, 'new_2')
    traversing(head)


if __name__ == '__main__':
    # Stack
    print('\n====>>>> Testing Stack \n')
    test_stack()
    brackets_detector('(()(())())')
    brackets_detector('(()())())')
    brackets_detector('(()(())()')
    brackets_detector('{{([][])()}}')
    brackets_detector('{{([]])()}}')
    brackets_detector('{{([][]))}}')
    decimal_to_binary(233)
    decimal_to_any_scale(233, 8)


    # Queue
    print('\n====>>>> Testing Queue \n')
    test_queue()
    transfer_potato()
    test_biqueue()
    pal_checker('admda')


    # Linked List
    test_linked_list()


    import time
    print(type(time.time()))