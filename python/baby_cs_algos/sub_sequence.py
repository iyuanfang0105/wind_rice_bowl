"""
如何判定字符串 s 是否是字符串 t 的子序列（可以假定 s 长度比较小，且 t 的长度非常大）。举两个例子：

s = "abc", t = "ahbgdc", return true.

s = "axc", t = "ahbgdc", return false.
"""


def is_subsequence(ss, target):
    i = 0
    j = 0

    while i < len(ss) and j < len(target):
        if ss[i] == target[j]:
            i += 1
        j += 1

    return i == len(ss)


if __name__ == '__main__':
    s = "abc"
    t = "ahbgdc"
    print('\"{}\" is the subsequence of \"{}\": {}'.format(s, t, is_subsequence(s, t)))

    s = "axc"
    t = "ahbgdc"
    print('\"{}\" is the subsequence of \"{}\": {}'.format(s, t, is_subsequence(s, t)))