import time
import numpy as np

"""
要在 D 天内运输完所有货物，货物不可分割，如何确定运输的最小载重呢（下文称为 cap）？

示例 1：
输入：weights = [1,2,3,4,5,6,7,8,9,10], D = 5
输出：15
解释：
船舶最低载重 15 就能够在 5 天内送达所有包裹，如下所示：
第 1 天：1, 2, 3, 4, 5
第 2 天：6, 7
第 3 天：8
第 4 天：9
第 5 天：10

示例 2：
输入：weights = [3,2,2,4,1,4], D = 3
输出：6
解释：
船舶最低载重 6 就能够在 3 天内送达所有包裹，如下所示：
第 1 天：3, 2
第 2 天：2, 4
第 3 天：1, 4
示例 3：

输入：weights = [1,2,3,1,1], D = 4
输出：3
解释：
第 1 天：1
第 2 天：2
第 3 天：3
第 4 天：1, 1
"""


def get_min_ship_capacity(weights, day_limits):
    min_cap = np.max(weights)  # len(weights) days can finish
    max_cap = np.sum(weights)  # 1 day can finish

    for cap in range(min_cap, max_cap+1):
        if can_finish(weights, cap, day_limits):
            return cap


def get_min_ship_capacity_binary_search(weights, day_limits):
    min_cap = np.max(weights)  # len(weights) days can finish
    max_cap = np.sum(weights)  # 1 day can finish

    left = min_cap
    right = max_cap + 1
    while left < right:
        mid = left + (right - left) // 2
        if can_finish(weights, mid, day_limits):
            right = mid
        else:
            left = mid + 1
    return left


def can_finish(weights, cap, day_limits):
    can = False
    days = 0
    weight = 0
    for w in weights:
        weight += w
        if weight > cap:
            days += 1
            weight = 0
    if days <= day_limits:
        can = True
    return can


if __name__ == '__main__':
    weights = [1,2,3,1,1]
    D = 4

    ts = time.time()
    min_ship_cap = get_min_ship_capacity(weights, day_limits=D)
    print('can finish in {} days with min_ship_cap {}, time consuption {}'.format(D, min_ship_cap, time.time()-ts))

    ts = time.time()
    min_ship_cap = get_min_ship_capacity_binary_search(weights, day_limits=D)
    print('can finish in {} days with min_ship_cap {}, time consuption {}'.format(D, min_ship_cap, time.time()-ts))