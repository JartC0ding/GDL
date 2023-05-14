from math import *
from typing import *
import numpy as np


class GDL:
    # start point has to have the result as the last element in the tuple
    def __init__(self, number_of_variables: int, degree: int, function: Callable, gradient: Callable, starting_point: Tuple[float, ...]) -> None:
        self.num_vars = number_of_variables
        self.degree = degree
        self.f = function
        self.nabla = gradient
        self.sp = starting_point

    def desc(self) -> Tuple[float, ...]:
        return self.__gda(False)

    def asc(self) -> Tuple[float, ...]:
        return self.__gda(True)

    # returns a tuple of dx scalars and it's roots
    def __derivative_roots(self, xx: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
        sol = []
        xx = xx[::-1]

        for i in range(len(xx)-1):
            a = len(xx)-1-i
            sol.append((xx[a]*a))

        return (sol, np.roots(sol))

    def __derivative_at(self, xx: tuple[float, ...], x: float) -> float:
        sol = 0
        xx = xx[::-1]

        for i in range(len(xx)-1):
            a = len(xx)-1-i
            sol += ((xx[a]*a)*(x**(a-1)))

        return sol

    # true for asc false for desc
    def __gda(self, asc_desc: bool) -> Tuple[float, ...]:
        # repeat until the gradient is zero
        while self.nabla(self.sp[0:(len(self.sp)-1)]) != 0:
            # get degree+1 points on the line formed by the gradient on f
            # 1. start to get the first point then extend all the coordinates of that point by nabla
            # 2. repeat this process degree times
            points = [self.sp]
            start = self.sp[:len(self.sp)-1]
            for _ in range(self.degree):
                start = list(
                    map(lambda x: x[0]+x[1], zip(self.nabla(self.sp[0:(len(self.sp)-1)]), start)))
                x = start
                x.append(self.f(start))
                points.append(x)

            # transform points to 2d coordinates
            # 1. for each point, take the magnitude of all but the last dimensional part of the coordinate
            points = list(map(lambda x: (
                sqrt(sum(list(map(lambda x: x**2, x[:len(x)-1])))), x[len(x)-1]), points))

            # transform the resulting points into a system of linear equations and solve for a resulting function
            xs = []
            ys = []
            for i in points:
                k = []
                for j in range(self.degree):
                    k.append(i[0]**(self.degree-j))
                k.append(0)

                xs.append(k)
                ys.append(i[1])

            scalars = list(map(lambda x: round(x, 4), np.linalg.lstsq(
                np.array(xs), np.array(ys), rcond=None)[0]))  # round each scalar value

            # find the local maxima or minima (depending on asc_desc)
            if asc_desc:
                # maxima
                dx, roots = self.__derivative_roots(scalars)

                # dx^2 at roots has to be < 0, else return inf
                maxima = []
                for i in roots:
                    x = self.__derivative_at(dx, i)
                    if (x < 0):
                        maxima.append(i)

                if len(maxima) == 0:
                    return inf

                # solve the equation for the actual function that solves for the maxima
            else:
                # minima
                dx, roots = self.__derivative_roots(scalars)

                # dx^2 at roots has to be > 0, else return inf
                minima = []
                for i in roots:
                    x = self.__derivative_at(dx, i)
                    if (x > 0):
                        minima.append(i)

                if len(minima) == 0:
                    return inf

                # choose the smallest point by evaluating the scalars at the point of the minima
                # with the smallest point solve the equation with the original function for the smallest point
                # if there are multiple solutions choose the one on the gradient line

            # update that to the start point
        return self.sp


def f(x):
    return (x[0]**2) + (x[1]**2)


def nabla_f(x):
    return ((2*x[0] + x[1]**2), (x[0]**2 + 2*x[1]))


def test():
    gdl = GDL(2, 2, f, nabla_f, (1, 1, f((1, 1))))
    print(gdl.asc()) # should be inf as x^2+y^2 has no highest point as x -> inf


test()
