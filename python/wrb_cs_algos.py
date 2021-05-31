import os
import time
import heapq
import numpy as np
from collections import OrderedDict


class BinarySearch(object):
    def __init__(self):
        print('')

    '''
    KOKO 偷香蕉
    n 堆香蕉 piles[i],  H 小时内吃完所有的香蕉，每小时的最小速度 k 是多少？每小时最多吃完一整堆"
    类似问题：D天内 运送完 传送带上的所有物品，最小的运载力？
    '''

    def eating_banana_force(self, piles, time_limits):
        '''
        using force searching
        :param piles:
        :param time_limits:
        :return:
        '''
        min_speed = 1
        max_speed = np.max(piles)

        for speed in range(min_speed, max_speed + 1):
            if self.can_finish(piles, time_limits, speed):
                return speed

    def eating_banana_bs(self, piles, time_limits):
        '''
        using binary searching
        :param piles:
        :param time_limits:
        :return:
        '''
        min_speed = 1
        max_speed = np.max(piles)

        left = min_speed
        right = max_speed + 1

        while (left < right):
            mid = left + (right - left) // 2
            if self.can_finish(piles, time_limits, mid):
                right = mid
            else:
                left = mid + 1
        return left

    def ship_bs(self, weights, day_limits):
        min_cap = max(weights)
        max_cap = np.sum(weights)

        left = min_cap
        right = max_cap + 1

        while (left < right):
            mid = left + (right - left) // 2
            if self.can_ship(weights, day_limits, mid):
                right = mid
            else:
                left = mid + 1
        return left

    @staticmethod
    def can_eat(piles, time_limits, speed):
        can = 0
        hours_consuption = 0
        for p in piles:
            hours_consuption += np.ceil(p / speed)
        if hours_consuption <= time_limits:
            can = 1
        return can

    @staticmethod
    def can_ship(wights, day_limits, ship_cap):
        can = 0
        days_consuption = 0

        sum_tmp = 0
        for w in wights:
            sum_tmp += w
            if sum_tmp > ship_cap:
                days_consuption += 1
                sum_tmp = 0
        if days_consuption <= day_limits:
            can = 1
        return can


class LRUCache:
    '''
    LRU 缓存淘汰算法就是一种常用策略。LRU 的全称是 Least Recently Used，
    也就是说我们认为最近使用过的数据应该是是「有用的」，很久都没用过的数据应该是无用的，内存满了就优先删那些很久没用过的数据。
    '''

    def __init__(self, capacity: int):
        self.capacity = capacity  # cache的容量
        self.visited = OrderedDict()  # python内置的OrderDict具备排序的功能

    def get(self, key: int) -> int:
        if key not in self.visited:
            return -1
        self.visited.move_to_end(key)  # 最近访问的放到链表最后，维护好顺序
        return self.visited[key]

    def put(self, key: int, value: int) -> None:
        if key not in self.visited and len(self.visited) == self.capacity:
            # last=False时，按照FIFO顺序弹出键值对
            # 因为我们将最近访问的放到最后，所以最远访问的就是最前的，也就是最first的，故要用FIFO顺序
            self.visited.popitem(last=False)
            self.visited[key] = value
            self.visited.move_to_end(key)  # 最近访问的放到链表最后，维护好顺序


class SubSeqAndPalindrome(object):
    '''
    is sub-seq or get longest palind
    '''

    def __init__(self):
        print('')

    def get_longest_palindrome(self, s):
        res = []
        for i in range(len(s) - 1):
            s1 = self.get_palindrome(s, i, i)
            s2 = self.get_palindrome(s, i, i + 1)
            res.append(s1)
            res.append(s2)
        return res

    @staticmethod
    def get_palindrome(s, left, right):
        ps = ''

        while right < len(s) and left >= 0:
            if s[left] == s[right]:
                # centre to border
                right += 1
                left -= 1
            else:
                break

        s_tmp = s[left + 1:right]
        # ps = s_tmp
        if len(s_tmp) > 1:
            ps = s_tmp

        return ps

    @staticmethod
    def is_subseq(s, t):
        idx_s = 0
        idx_t = 0

        while idx_s < len(s) and idx_t < len(t):
            if s[idx_s] == t[idx_t]:
                idx_s += 1
                # idx_t = idx_s
            idx_t += 1

        return idx_s == len(s)


class Brackets(object):
    def __init__(self):
        self.brackets_left = ['(', '{', '[']
        self.brackets_left_to_right = {'(': ')',
                                       '{': '}',
                                       '[': ']'}

    def is_valid(self, s):
        stack = []
        for c in s:
            if c in self.brackets_left:
                stack.append(c)
            else:
                if len(stack) > 0 and c == self.brackets_left_to_right[stack[-1]]:
                    stack.pop()

        return len(stack) == 0


class Primes(object):
    def __init__(self):
        print()

    def get_primes(self, N):
        ps = []
        for i in range(1, N + 1):
            if self.is_primes(i):
                ps.append(i)
        return ps

    @staticmethod
    def is_primes(N):
        flag = True
        for i in range(2, int(np.floor(np.sqrt(N))) + 1):
            if N % i == 0:
                flag = False
        return flag


class SortedAlgos(object):
    def __init__(self):
        print('fuck-algo-sorting')

    """
    Suppose a sorted array is rotated at some pivot unknown to you beforehand.
    (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
    """

    @staticmethod
    def find_element_in_semi_sorted_list_1(arr_in, v):
        st_v = arr_in[0]
        ed_v = arr_in[-1]

        if v < st_v:
            for i in range(len(arr_in) - 1, 0, -1):
                if v == arr_in[i]:
                    return i
        else:
            for i in range(len(arr_in)):
                if v == arr_in[i]:
                    return i

    @staticmethod
    def find_element_in_semi_sorted_list_2(arr_in, v):
        st_id = 0
        ed_id = len(arr_in) - 1

        while st_id < ed_id:
            mid_id = (st_id + ed_id) // 2 + 1

            if v == arr_in[mid_id]:
                return mid_id

            # 如果前半段有序
            if arr_in[st_id] < arr_in[mid_id]:
                # 判断target是否在前半段，如果在，则继续遍历前半段，如果不在，则继续遍历后半段
                if arr_in[st_id] <= v < arr_in[mid_id]:
                    ed_id = mid_id - 1
                else:
                    st_id = mid_id + 1
            else:  # 如果后半段有序
                if arr_in[mid_id] < v <= arr_in[ed_id]:
                    st_id = mid_id + 1
                else:
                    ed_id = mid_id - 1

        return -1

    @staticmethod
    def find_element_binary_search(arr_in, v):
        st_id = 0
        ed_id = len(arr_in) - 1

        while st_id < ed_id:
            mid_id = (st_id + ed_id) // 2 + 1

            if v == arr_in[mid_id]:
                return mid_id

            if arr_in[st_id] <= v < arr_in[mid_id]:
                ed_id = mid_id - 1
            else:
                st_id = mid_id + 1

        return -1


class LinkedListNode(object):
    def __init__(self, val):
        self.val = val
        self.next = None


class LinkedList(object):
    """
    206：reverse linked list
    24: swap nodes in pairs, Given 1->2->3->4, you should return the list as 2->1->4->3.
    141: linked list cycle, solution: set / fast-slow pointer
    """

    def __init__(self):
        print('fuck linked list!!!')

    @staticmethod
    def reverse_list(head):
        cur = head
        prev = None

        while cur:
            cur.next = prev
            prev = cur
            cur = cur.next

        return prev

    def swap_list(self, head):
        # recursion stop
        if head is None or head.next is None:
            return head

        new_node = head.next
        head.next = self.swap_list(head.next.next)
        new_node.next = head

        return new_node

    @staticmethod
    def has_cycle(head):
        has_cyc = False
        fast = slow = head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                has_cyc = True
                break
        return has_cyc

    @staticmethod
    def demo():
        l4 = LinkedListNode(4)
        l3 = LinkedListNode(3)
        l2 = LinkedListNode(2)
        l1 = LinkedListNode(1)
        l0 = LinkedListNode(0)

        l0.next = l1
        l1.next = l2
        l2.next = l3
        l3.next = l4


class StackAndQueue(object):
    """
    20: is valid brackets
    232, 255: using stack to be queue, using queue to be stack

    """

    def __init__(self):
        print('fuck stack and queue!!!')

    def is_valid_brackets(ss):
        brackets_left = ['(', '{', '[']
        brackets_left_to_right = {'(': ')', '{': '}', '[': ']'}
        stack = []
        for c in ss:
            if c in brackets_left:
                stack.append(c)
            else:
                if len(stack) > 0 and c == brackets_left_to_right[stack[-1]]:
                    stack.pop()

        return len(stack) == 0


class Heap(object):
    """
    priority queue
    703: top_k, solution: sort / heep
    239: sliding widow max, Input: nums = [1,3,-1,-3,5,3,6,7], and k = 3, Output: [3,3,5,5,6,7]
    """

    def __init__(self):
        print('fuck heap!!!')


class TopKHeap(object):
    def __init__(self, k, nums_list):
        self.k = k
        self.pool = heapq.nlargest(k, nums_list)[::-1]

    def insert_element(self, v):
        if len(self.pool) < self.k:
            heapq.heappush(self.pool, v)
        elif v > self.pool[0]:
            heapq.heapreplace(self.pool, v)
        return self.pool[0]


class TopKSort(object):
    def __init__(self, k, nums_list):
        self.k = k
        self.nums_list = nums_list

    def insert_elelment(self, v):
        self.nums_list.append(v)
        self.nums_list.sort(reverse=True)
        return self.nums_list[self.k - 1]


def sliding_window_max(nums, k):
    if not nums: return []
    window, res = [], []

    for i, x in enumerate(nums):
        if i >= k and window[0] <= i - k:
            window.pop()
        while window and nums[window[-1]] <= x:
            window.pop()
        window.append(i)
        if i > k - 1:
            res.append(nums[window[0]])
    return res


"""
Hash and set
242: valid anagram, s = "anagram", t = "nagaram", return true / s = "rat", t = "car", return false.
1: two sum,  nums = [2, 7, 11, 15], target = 9, return  [0, 1]
15: three sum, nums=[-1, 0, 1, 2, -1, -4], target=0
18: four sum
"""


def is_anagram(s, t):
    return sorted(s) == sorted(t)


def is_anagram_2(s, t):
    dict_s = {}
    dict_t = {}
    for item in s:
        dict_s[item] = dict_s.get(item, 0) + 1

    for item in t:
        dict_t[item] = dict_t.get(item, 0) + 1

    return dict_s == dict_t


def two_sum(nums, tag):
    # O(N)
    dict_t = {}
    for i in range(len(nums)):
        x = nums[i]
        if tag - x in dict_t.keys():
            return [dict_t[tag - x], i]
        dict_t[x] = i


"""
Tree and Graph
98: validate binary search tree, using in-order traverse, should be sorted
235/236： lowest common ancestor
pre-order: root-left-right
in-order: left-root-right
post-order: left-right-root
"""


class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def is_valid_BST(root):
    inorder_list = inorder_traverse(root)
    return inorder_list == list(sorted(set(inorder_list)))


def inorder_traverse(root):
    if root is None:
        return []
    return inorder_traverse(root.left) + [root.val] + inorder_traverse(root.right)


def preorder_traverse(root):
    if root is None:
        return []
    return [root.val] + preorder_traverse(root.left) + preorder_traverse(root.right)


def postorder_traverse(root):
    if root is None:
        return []
    return preorder_traverse(root.left) + preorder_traverse(root.right) + [root.val]


def is_valid_BST_2(root, min, max):
    if root is None:
        return True
    if min and root.val <= min: return False
    if max and root.val >= max: return False

    return is_valid_BST_2(root.left, min, root.val) and is_valid_BST_2(root.right, root.val, max)


def find_nearst_common_ancestor(root, p, q):
    if root is None or root == p or root == q: return root
    left = find_nearst_common_ancestor(root.left, p, q)
    right = find_nearst_common_ancestor(root.right, p, q)

    if left is None:
        return right
    if right is None:
        return left
    return root


def find_nearst_common_ancestor_in_BST(root, p, q):
    if p.val < root.val > q.val:
        return find_nearst_common_ancestor_in_BST(root.left, p, q)
    if p.val > root.val < q.val:
        return find_nearst_common_ancestor_in_BST(root.right, p, q)
    return root


def find_nearst_common_ancestor_in_BST_2(root, p, q):
    while root:
        if p.val < root.val > q.val:
            root = root.left
        elif p.val > root.val < q.val:
            root = root.right
        else:
            return root


"""
recursion, divide&conquer

factorial: n!
fibonacci: 1, 1, 2, 3, 5, 8, 13, 21, 34, ..., f(n) = f(n-1) + f(n-2)
50: power, x^n
169: majority, 众数， count(x) > n/2, using map
"""


def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)


def fibonacci(n):
    if n == 0 or n == 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def power(x, n):
    if n == 0:
        return 1
    if n < 0:
        return 1 / power(x, -n)

    if n % 2: # n is odd
        return x * power(x, n-1)

    return power(x*x, n/2) # n is even


def find_majority(nums):
    return 0



"""
greedy

"""





if __name__ == '__main__':
    # bs = BinarySearch()
    # bs.eating_banana()

    # ssap = SubSeqAndPalindrome()
    # print(ssap.get_longest_palindrome('babad'))
    # print(ssap.is_subseq(s="abc", t="ahbgdc"))
    # print(ssap.is_subseq(s="axc", t="ahbgdc"))

    # brk = Brackets()
    # print(brk.is_valid('({[]})'))

    # primes = Primes()
    # print(primes.get_primes(12))

    # sta = SortedAlgos()
    # print(sta.find_element_in_semi_sorted_list_1([4, 5, 6, 7, 0, 1, 2], 0))
    # print(sta.find_element_in_semi_sorted_list_2([4, 5, 6, 7, 0, 1, 2], 0))
    # print(sta.find_element_binary_search([0, 1, 2, 4, 5, 6, 7], 2))

    print(sliding_window_max(nums=[1, 3, -1, -3, 5, 3, 6, 7], k=3))
    print(is_anagram(s="anagram", t="nagaram"), is_anagram(s="cat", t="rat"))
    print(is_anagram_2(s="anagram", t="nagaram"), is_anagram(s="cat", t="rat"))
    print(two_sum(nums=[2, 7, 11, 15], tag=9))
    print(factorial(6))
    print(fibonacci(6))
    print(power(2, 11))
