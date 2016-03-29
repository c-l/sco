DEFAULT_TOL=1e-8

class Cntr(object):
    """
    Constraint class
    Constraint is composed of a bound expression and a value.

    """

    def __init__(self, bound_expr, val):
        self.bound_expr = bound_expr
        self.val = val

    def test(self, tol=DEFAULT_TOL):
        """
        Tests whether constraint holds true.
        """
        raise NotImplementedError

    def convexify(self):
        """
        Returns a BoundExpr which represents the penalty term associated with
        this constraint.
        """
        raise NotImplementedError

class EqCntr(Cntr):
    """
    Equality constraint
    """

    def test(self, x, tol=DEFAULT_TOL):
        """
        Tests whether the bound expression (self.bound_expr) is equal to val
        within tolerance tol.
        """
        raise NotImplementedError

    def convexify(self):
        """
        Returns a BoundExpr which represents the l1 penalty term associated with
        this constraint.
        """
        raise NotImplementedError

class LEqCntr(Cntr)
    """
    Less than equal to constraint
    """

    def test(self, x, tol=DEFAULT_TOL):
        """
        Tests whether the bound expression (self.bound_expr) is less than or
        equal to val within tolerance tol.
        """
        raise NotImplementedError

    def convexify(self):
        """
        Returns a BoundExpr which represents the hinge penalty term associated
        with this constraint.
        """
        raise NotImplementedError
