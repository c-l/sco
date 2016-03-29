import numpy as np, numdifftools as nd

"""
Utility classes to represent expresions. Each expression defines an eval, grad,
hess, and convexify method.
"""


class Expr(object):

    """
    by default, expressions are defined by black box functions
    """

    def __init__(self, f):
        self.f = f
        self._grad = nd.Jacobian(f)

    def eval(self, x):
        return self.f(x)

    def grad(self, x):
        return self._grad(x)

    def hess(self, x):
        raise NotImplementedError

    def convexify(self, x, degree=1):
        """
        Returns an convex approximation of the expression at x where degree 1 is
        an affine approximation and degree 2 is a quadratic approximation
        """

        if degree == 1:
            A = self.grad(x)
            b = self.eval(x) - A.dot(x)
            return AffExpr(A, b)
        else:
            raise NotImplementedError


class AffExpr(Expr):

    def __init__(self, A, b):
        """
        expr is Ax + b
        """
        self.A = A
        self.b = b
        self.x_shape = (A.shape[1], 1)

    def eval(self, x):
        return self.A.dot(x) + self.b

    def grad(self, x):
        return self.A.T

    def hess(self, x):
        return 0.0


class QuadExpr(AffExpr):

    def __init__(self, Q, A, b):
        """
        expr is x'Qx + Ax + b
        """
        assert A.shape[0] == 1, 'Can only define scalar quadrative expressions'
        super(QuadExpr, self).__init__(A, b)
        self.Q = Q

    def eval(self, x):
        return x.T.dot(self.Q.dot(x)) + self.A.dot(x) + self.b

    def grad(self, x):
        assert x.shape == self.x_shape
        return self.Q.dot(x) + self.Q.T.dot(x) + self.A.T

    def hess(self, x):
        return self.Q.copy()


class BoundExpr(object):
    """
    Bound expression

    Bound expression is composed of an Expr and a Variable
    """

    def __init__(self, expr, var):
        self.expr = expr
        self.var = var

    def grb_expr(self):
        """
        Returns the Gurobi expression
        """
        return self.expr.eval(self.var.get_grb_vars())

    def eval(self):
        """
        Returns the current value of the bound expression
        """
        return self.expr.eval(self.var.get_value())

    def convexify(self, degree=1):
        """
        Returns a convexified BoundExpr at the variable's current value.
        """
        convexified_expr = self.expr.convexify(self.var.get_value(), degree)
        return BoundExpr(convexified_expr, self.var)


class TFExpr(Expr):

    """
    TODO

    wrapper around exprs defined by a tensorflow graph. Leverages
    automated differentition.
    """
    raise NotImplementedError
