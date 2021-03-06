"""
你和你的朋友面前有一堆石子，你们轮流拿，一次至少拿一颗，最多拿三颗，谁拿走最后一颗石子谁获胜。
或者说如何保证胜，只要每次保证剩下的总数为4的总数
"""


def can_win(n):
    return n % 4 != 0

"""
你和你的朋友面前有一排石头堆，用一个数组 piles 表示，piles[i] 表示第 i 堆石子有多少个。你们轮流拿石头，
一次拿一堆，但是只能拿走最左边或者最右边的石头堆。所有石头被拿完后，谁拥有的石头多，谁获胜。
石头的堆的数量为偶数，所以你们两人拿走的堆数一定是相同的。
石头的总数为奇数，也就是你们最后不可能拥有相同多的石头，一定有胜负之分。

[2, 1, 9, 5]
[2, 9, 1, 5]

方法：先对比 奇数堆和偶数堆 的总和 大小，比如 奇数堆的总和大，那就保证一直取奇数堆的

"""

if __name__ == '__main__':
    print('can win: {}'.format(can_win(4)))
