"""
回文串是对称的，所以正着读和倒着读应该是一样的，这一特点是解决回文串问题的关键。
核心思想是从中心向两端扩展
思路：双指针
"""


def get_palindrome_strings(ss):
    pd_strs = []

    if len(ss) <= 2:
        pd_strs.append(ss)

    for idx in range(len(ss)-1):
        # 以idx为中心
        pd_s_1 = get_palindrome_string(ss, idx, idx)
        # 以idx, idx+1为中心
        pd_s_2 = get_palindrome_string(ss, idx, idx+1)
        if len(pd_s_1) > 0:
            pd_strs.append(pd_s_1)
        if len(pd_s_2) > 0:
            pd_strs.append(pd_s_2)

    return pd_strs


def get_palindrome_string(ss, left_center, right_center):
    while left_center>=0 and right_center<len(ss) and ss[left_center]==ss[right_center]:
        left_center -= 1
        right_center += 1

    pd_str = ss[left_center+1:right_center]
    if len(pd_str) >= 2:
        return pd_str
    else:
        return ""


if __name__ == '__main__':
    ss = ["babad", "aacxycaa", "abxxbababad"]
    for s in ss:
        palindrome = get_palindrome_strings(s)
        print("palindrome in \"{}\": {}".format(s, palindrome))
