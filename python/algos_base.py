from turtle import *

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

    def traversing(self):
        out_str = ''
        while self.size():
            out_str += self.pop()
            # out_str.join(str(self.pop()) + ' ')
        print(out_str)


def decimal_to_any_scale(num, base):
    digits = '0123456789ABCDEF'
    if num < base:
        return digits[num]
    else:
        return decimal_to_any_scale(num//base, base) + digits[num % base]


def show_recursive_spiral():
    # plot a spiral
    my_turtle = Turtle()
    my_window = my_turtle.getscreen()

    def draw_spiral(my_turtle, line_len):
        if line_len > 0:
            my_turtle.forward(line_len)
            my_turtle.right(90)
            line_len = line_len - 5
            draw_spiral(my_turtle, line_len)

    draw_spiral(my_turtle, 200)
    my_window.exitonclick()


def show_recursive_tree():
    # plot a tree
    my_turtle = Turtle()
    my_window = my_turtle.getscreen()

    my_turtle.left(90)
    my_turtle.up()
    my_turtle.backward(300)
    my_turtle.down()
    my_turtle.color('green')

    def tree(branch_len, turtle):
        if branch_len > 5:
            turtle.forward(branch_len)
            turtle.right(20)
            tree(branch_len-15, turtle)
            turtle.left(40)
            tree(branch_len-10, turtle)
            turtle.right(20)
            turtle.backward(branch_len)

    tree(110, my_turtle)
    my_window.exitonclick()


if __name__ == '__main__':
    print(decimal_to_any_scale(10, 2))
    # show_recursive_spiral()
    show_recursive_tree()
