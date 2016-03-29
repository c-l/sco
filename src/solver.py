class Solver(object):
    """
    SCO Solver
    """

    def __init__(self):
        pass

    def solve(self, prob, method=None):
        """
        Given a sco (sequential convex optimization) problem instance, solve
        using specified method to find a solution
        """
        raise NotImplementedError

    def _penalty_sqp(self, prob):
        """
        Uses Penalty Sequential Quadratic Programming to solve the problem
        instance.
        """

        pass

    def _min_merit_fn(self, prob, penalty_coeff, trust_region_size):
        """
        Minimize merit function for penalty sqp
        """
        pass
