import sys
import math


class IntervalNumber:
    def __init__(self, l=0, u=0):
        self.l = l
        self.u = u

    def __repr__(self):
        return "[" + str(self.l) + ", " + str(self.u) + "]"

    def __neg__(self):
        return IntervalNumber(-self.u, -self.l)

    def __add__(self, other):
        if type(self).__name__ == type(other).__name__:
            return IntervalNumber(self.l + other.l, self.u + other.u)
        else:
            return IntervalNumber(self.l + other, self.u + other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if type(self).__name__ == type(other).__name__:
            print("An IntervalNumber can not multiply another IntervalNumber, please check code.")
            sys.exit(0)
        if other >= 0:
            return IntervalNumber(self.l * other, self.u * other)
        else:
            return IntervalNumber(self.u * other, self.l * other)

    def __rmul__(self, other):
        return self * other


def interval_max(x, y):
    if type(x).__name__ == "IntervalNumber":
        if x.u <= y:
            return IntervalNumber(y, y)
        elif x.l >= y:
            return x
        else:
            return IntervalNumber(y, x.u)
    elif type(x[0][0]).__name__ == "IntervalNumber":
        for i in range(0, len(x[0])):
            x[0][i] = interval_max(x[0][i], y)
        return x
    else:
        print("Incorrect use of interval_max, please check code.")
        sys.exit(0)


def my_sigmoid(x):
    return 1/(1+math.exp(-x))


def interval_sigmoid(x):
    if type(x).__name__ == "IntervalNumber":
        return IntervalNumber(my_sigmoid(x.l), my_sigmoid(x.u))
    elif type(x[0][0]).__name__ == "IntervalNumber":
        for i in range(0, len(x[0])):
            x[0][i] = interval_sigmoid(x[0][i])
        return x
    else:
        print("Incorrect use of interval_sigmoid, please check code.")
        sys.exit(0)


def my_tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))


def interval_tanh(x):
    if type(x).__name__ == "IntervalNumber":
        return IntervalNumber(my_tanh(x.l), my_tanh(x.u))
    elif type(x[0][0]).__name__ == "IntervalNumber":
        for i in range(0, len(x[0])):
            x[0][i] = interval_tanh(x[0][i])
        return x
    else:
        print("Incorrect use of interval_tanh, please check code.")
        sys.exit(0)


def interval_cos_sin(x):
    if type(x).__name__ == "IntervalNumber":
        return IntervalNumber(-1, 1)
    elif type(x[0][0]).__name__ == "IntervalNumber":
        for i in range(0, len(x[0])):
            x[0][i] = interval_cos_sin(x[0][i])
        return x
    else:
        print("Incorrect use of interval_tanh, please check code.")
        sys.exit(0)


def inf(x):
    return x.l


def sup(x):
    return x.u



