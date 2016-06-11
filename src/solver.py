class Solver(object):
    """
    SCO Solver
    """

    def __init__(self):
        pass

    def solve(self, prob, method=None):
        """
        Given a sco (sequential convex optimization) problem instance, solve
        using specified method to find a solution.
        """
        raise NotImplementedError

    def _penalty_sqp(self, prob):
        """
        Uses Penalty Sequential Quadratic Programming to solve the problem
        instance.
        """
        raise NotImplementedError

    def _min_merit_fn(self, prob, penalty_coeff, trust_region_size):
        """
        Minimize merit function for penalty sqp
        """
        raise NotImplementedError

    def _is_converged(self, trust_region_size):
        """
        Returns true if prob has converged
        """
        raise NotImplementedError

    def _bad_model(self, approx_merit_improve):
        """
        Returns true if the approx_merit_improve is too low which suggests that
        either the convexification is wrong to the zeroth order or there are
        numerical problems.
        """
        raise NotImplementedError

    def _shrink_trust_region(self, exact_merit_improve, approx_merit_improve):
        """
        Returns true if the trust region should shrink (exact merit improve is negative or the merit improve ratio is too low)
        """
        raise NotImplementedError

    def _x_converged(self, trust_region_size):
        """
        Returns true if the variable values has converged (trust_region size is
        smaller than the minimum trust box size)
        """
        raise NotImplementedError

    def _y_converged(self, approx_merit_improve):
        """
        Returns true if the approx_merit has converged (approx_merit <
        min_approx_merit_improve)
        """
        raise NotImplementedError
