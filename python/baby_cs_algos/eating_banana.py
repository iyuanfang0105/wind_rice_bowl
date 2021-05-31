import time
import numpy as np

"""
n 堆香蕉 piles[i],  H小时内吃完所有的香蕉，每小时的最小速度是多少， 每小时最多吃完一整堆？
"""


def get_min_speed_original(piles, time_limits):
    min_speed = 1
    max_speed = np.max(piles)

    for speed in range(min_speed, max_speed+1):
        if can_finish(piles, time_limits, speed):
            return speed


def get_min_speed_binary_search(piles, time_limits):
    min_speed = 1
    max_speed = np.max(piles)

    # for speed in range(min_speed, max_speed+1):
    #     if can_finish(piles, time_limits, speed):
    #         return speed
    left = min_speed
    right = max_speed + 1
    while(left < right):
        mid = left + (right - left) // 2
        if can_finish(piles, time_limits, mid):
            right = mid
        else:
            left = mid + 1
    return left


def can_finish(piles, time_limits, speed):
    can = 0
    hours_consuption = 0
    for p in piles:
        hours_consuption += np.ceil(p / speed)
    if hours_consuption <= time_limits:
        can = 1
    return can


if __name__ == '__main__':
    piles = [3, 6, 7, 11]
    hours_limits = 8
    ts = time.time()
    sp = get_min_speed_original(piles, hours_limits)
    print('result of get_min_speed_original: {}, time consumption: {}'.format(sp, time.time() - ts ))

    ts = time.time()
    sp = get_min_speed_binary_search(piles, hours_limits)
    print('result of get_min_speed_binary_search: {}, time consumption: {}'.format(sp, time.time() - ts ))

    piles = [30, 11, 23, 4, 20]
    hours_limits = 5

    ts = time.time()
    sp = get_min_speed_original(piles, hours_limits)
    print('result of get_min_speed_original: {}, time consumption: {}'.format(sp, time.time() - ts ))

    ts = time.time()
    sp = get_min_speed_binary_search(piles, hours_limits)
    print('result of get_min_speed_binary_search: {}, time consumption: {}'.format(sp, time.time() - ts ))

