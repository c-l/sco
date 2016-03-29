import gurobipy as grb

class Prob(object):
    """
    Sequential convex programming problem.
    """

    def __init__(self):
        self._model = grb.Model()
        self._quad_obj = 0.0
        self._penalty_obj = 0.0
        self.obj_exprs = []
        self.cnt_exprs = []
        self.vars = []
        self._convexified_obj_exprs = []

    def add_obj_expr(self, expr, var):
        """
        Adds an expression (expr) to the objective.

        If the expression (expr) is quadratic, the expression is directly added
        to the objective (self._quad_obj). Otherwise, a bound expression
        (BoundExpr) is created and added to self.obj_exprs to keep track of the
        objective terms that need to be convexified.

        var is added to self.vars so that a trust region can be added to var.
        """
        raise NotImplementedError

    # adds an Expr instance to prob's constraints
    def add_cnt_expr(self, expr, var):
        """
        Adds an expression (expr) to the constraints.

        If the expression is linear, the constraint is directly added to model.
        Otherwise, a bound expression (BoundExpr) is created and added to
        self.cnt_exprs to keep track of all the constraints that need to be
        convexified.

        var is added to self.vars so that a trust region can be added to var.
        """
        raise NotImplementedError

    # calls gurobi optimizer on the current QP approximation
    def optimize(self):
        """
        Calls the Gurobi optimizer on the current QP approximation
        """
        raise NotImplementedError

    def add_trust_region(self, trust_region_size):
        """
        Adds the trust region by changing the lower bound and upper bound of
        every variable in self.vars
        """
        raise NotImplementedError

    def convexify(self, penalty_coeff):
        """
        Convexifies the optimization problem by computing the penalty objective
        and saving it in self._penalty_obj. The penalty objective is obtained by
        summing up the convexified constraint Gurobi expressions. The
        convexified constraint Gurobi expressions are saved in
        self._convexified_obj_exprs.
        """
        raise NotImplementedError


    def get_value(self, penalty_coeff):
        """
        Returns the value of the penalty objective
        """
        raise NotImplementedError

    def get_approx_value(self, penalty_coeff):
        """
        Returns the value the penalty QP approximation by summing up the values
        of self._convexified_obj_exprs on the current value of the variables.
        """
        raise NotImplementedError

    def save(self):
        """
        Saves the problem's current state by saving the values of all the
        variables.
        """
        for var in self.vars
            var.save()

    def restore(self):
        """
        Restores the problem's state to the problem's saved state
        """
        for var in self.vars
            var.restore()
