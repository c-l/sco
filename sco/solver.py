import time
from ipdb import set_trace as st

class Solver(object):
    """
    SCO Solver
    """

    def __init__(self):
        """
        values taken from Pieter Abbeel's CS287 hw3 q2 penalty_sqp.m file
        """
        self.improve_ratio_threshold = .25
        self.min_trust_region_size = 1e-4
        self.min_approx_improve = 1e-4
        self.min_cnt_improve = 1e-2
        self.max_iter = 50
        self.trust_shrink_ratio = .1
        self.trust_expand_ratio = 1.5
        self.cnt_tolerance = 1e-4
        self.max_merit_coeff_increases = 1
        self.merit_coeff_increase_ratio = 10
        self.initial_trust_region_size = 1
        self.initial_penalty_coeff = 1e3

    def solve(self, prob, method=None, tol=None):
        """
        Returns whether solve succeeded.

        Given a sco (sequential convex optimization) problem instance, solve
        using specified method to find a solution. If the specified method
        doesn't exist, an exception is thrown.
        """
        if tol is not None:
            self.min_trust_region_size = tol
            self.min_approx_improve = tol
            self.cnt_tolerance = tol

        if method is "penalty_sqp":
            return self._penalty_sqp(prob, early_converge=False)
        elif method is "penalty_sqp_early_converge":
            return self._penalty_sqp(prob, early_converge=True)
        else:
            raise Exception("This method is not supported.")


    def _penalty_sqp(self, prob, early_converge=False):
        """
        Return True, None is the penalty sqp method succeeds.
        Uses Penalty Sequential Quadratic Programming to solve the problem
        instance.
        If early_converge is true, then it returns (False, violated bound
        expression).
        """
        start = time.time()
        trust_region_size = self.initial_trust_region_size
        penalty_coeff = self.initial_penalty_coeff

        prob.find_closest_feasible_point()

        for i in range(self.max_merit_coeff_increases):
            success, violated_bexpr = self._min_merit_fn(prob, penalty_coeff, trust_region_size, early_converge)
            print '\n'
            if violated_bexpr is not None:
                return False

            if prob.get_max_cnt_violation() > self.cnt_tolerance:
                penalty_coeff = penalty_coeff*self.merit_coeff_increase_ratio
                trust_region_size = self.initial_trust_region_size
            else:
                end = time.time()
                print "sqp time: ", end-start
                return success
        end = time.time()
        print "sqp time: ", end-start
        return False


    def _min_merit_fn(self, prob, penalty_coeff, trust_region_size, early_converge=False):
        """
        Returns true if the merit function is minimized successfully.
        Minimize merit function for penalty sqp
        """
        sqp_iter = 1

        while True:
            print("  sqp_iter: {0}".format(sqp_iter))

            prob.convexify()
            merit = prob.get_value(penalty_coeff)
            cnt_merits = prob.get_cnt_values(penalty_coeff)
            shared_cnt_merits = prob.get_shared_cnt_values(penalty_coeff)
            prob.save()

            while True:
                print("    trust region size: {0}".format(trust_region_size))

                prob.add_trust_region(trust_region_size)
                prob.optimize(penalty_coeff)

                model_merit = prob.get_approx_value(penalty_coeff)
                model_cnt_merits = prob.get_cnt_approx_values(penalty_coeff)
                model_shared_cnt_merits = prob.get_shared_cnt_approx_values(penalty_coeff)
                new_merit = prob.get_value(penalty_coeff)
                new_cnt_merits = prob.get_cnt_values(penalty_coeff)
                new_shared_cnt_merits = prob.get_shared_cnt_values(penalty_coeff)

                all_cnts_merits = (cnt_merits, model_cnt_merits, new_cnt_merits)
                all_shared_cnts_merits = (shared_cnt_merits, model_shared_cnt_merits, new_shared_cnt_merits)

                approx_merit_improve = merit - model_merit
                exact_merit_improve = merit - new_merit
                merit_improve_ratio = exact_merit_improve / approx_merit_improve

                print("      merit: {0}. model_merit: {1}. new_merit: {2}".format(merit, model_merit, new_merit))
                print("      approx_merit_improve: {0}. exact_merit_improve: {1}. merit_improve_ratio: {2}".format(approx_merit_improve, exact_merit_improve, merit_improve_ratio))

                if self._bad_model(approx_merit_improve):
                    print("Approximate merit function got worse ({0})".format(approx_merit_improve))
                    print("Either convexification is wrong to zeroth order, or you're in numerical trouble.")
                    prob.restore()
                    return False, None

                if self._y_converged(approx_merit_improve):
                    print("Converged: y tolerance")
                    prob.restore()
                    return True, None

                if early_converge:
                    cnt_converged, bexpr = self._violated_cnt_converged(all_cnts_merits, all_shared_cnts_merits)
                    if cnt_converged:
                        prob.restore()
                        return False, bexpr

                if self._shrink_trust_region(exact_merit_improve, merit_improve_ratio):
                    prob.restore()
                    print("Shrinking trust region")
                    trust_region_size = trust_region_size * self.trust_shrink_ratio
                else:
                    print("Growing trust region")
                    trust_region_size = trust_region_size * self.trust_expand_ratio
                    break #from trust region loop

                if self._x_converged(trust_region_size):
                    print("Converged: x tolerance")
                    return True, None

            sqp_iter = sqp_iter + 1

    def _bad_model(self, approx_merit_improve):
        """
        Returns true if the approx_merit_improve is too low which suggests that
        either the convexification is wrong to the zeroth order or there are
        numerical problems.
        """
        return approx_merit_improve < -1e-5

    def _shrink_trust_region(self, exact_merit_improve, merit_improve_ratio):
        """
        Returns true if the trust region should shrink (exact merit improve is negative or the merit improve ratio is too low)
        """
        return (exact_merit_improve < 0) or \
            (merit_improve_ratio < self.improve_ratio_threshold)

    def _violated_cnt_converged(self, all_cnt_merits, all_shared_cnts_merits):
        """
        Returns a tuple (True, violated constraint) if a violated constraint has
        converged. Otherwise it returns (False, None)
        """
        cnt_merits, model_cnt_merits, new_cnt_merits = all_cnt_merits
        shared_cnt_merits, model_shared_cnt_merits, new_shared_cnt_merits = all_shared_cnts_merits

        for bexpr, merit in cnt_merits.iteritems():
            assert bexpr in model_cnt_merits
            assert bexpr in new_cnt_merits
            assert bexpr in shared_cnt_merits
            assert bexpr in model_shared_cnt_merits
            assert bexpr in new_shared_cnt_merits

            model_merit = model_cnt_merits[bexpr]
            new_merit = new_cnt_merits[bexpr]

            shared_merit = shared_cnt_merits[bexpr]
            model_shared_merit = model_shared_cnt_merits[bexpr]
            new_shared_merit = new_shared_cnt_merits[bexpr]
            # if constraint is not satisfied
            if new_merit > self.cnt_tolerance:
                approx_merit_improve = merit - model_merit
                exact_merit_improve = merit - new_merit
                approx_shared_merit_improve = shared_merit - model_shared_merit
                exact_shared_merit_improve = shared_merit - new_shared_merit
                print ""
                print "approx_merit_improve: {0}".format(approx_merit_improve)
                print "exact_merit_improve: {0}".format(exact_merit_improve)
                print "approx_shared_merit_improve: {0}".format(approx_shared_merit_improve)
                print "exact_shared_merit_improve: {0}".format(exact_shared_merit_improve)
                if (abs(approx_merit_improve) < self.min_cnt_improve and
                    abs(exact_merit_improve) < self.min_cnt_improve and
                    abs(approx_shared_merit_improve) < self.min_cnt_improve):
                    return True, bexpr
        return False, None

    def _x_converged(self, trust_region_size):
        """
        Returns true if the variable values has converged (trust_region size is
        smaller than the minimum trust region size)
        """
        return trust_region_size < self.min_trust_region_size

    def _y_converged(self, approx_merit_improve):
        """
        Returns true if the approx_merit has converged (approx_merit_improve <
        min_approx_merit_improve)
        """
        return approx_merit_improve < self.min_approx_improve
