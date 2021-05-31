# %%
"""
LRU: Least Recently Used，也就是说我们认为最近使用过的数据应该是是「有用的」，很久都没用过的数据应该是无用的，
内存满了就优先删那些很久没用过的数据。
根本的难点在于记录最久被使用的键值对，这就设计到排序的问题，在python中，天生具备排序功能的字典就是OrderDict。
注意到，记录最久未被使用的键值对的充要条件是将每一次put/get的键值对都定义为最近访问，那么最久未被使用的键值对自然就会排到最后。
如果你深入python OrderDict的底层实现，就会知道它的本质是个双向链表+字典。
它内置支持了
1. move_to_end来重排链表顺序，它可以让我们将最近访问的键值对放到最后面
2. popitem来弹出键值对，它既可以弹出最近的，也可以弹出最远的，弹出最远的就是我们要的操作。
"""
from collections import OrderedDict


class LRUCache:
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


# %%
# # test LRU
# lru_cache = LRUCache(capacity=5)
# lru_cache.get()