import gurobipy as grb
from expr import *
from ipdb import set_trace as st

class Prob(object):
    """
    Sequential convex programming problem. A solution is found using the l1
    penalty method. In the this class, cvx (convex) refers to quadratic
    objectives and linear constraints.
    """

    def __init__(self):
        """
        _model: Gurobi model associated with this problem
        _vars: variables in this problem
        _obj_exprs: list of expressions in the objective, the sum of the
            expressions represent the objective
        _cnt_exprs: list of constraint expressions

        The l1 penalty objective is the sum of the quadratic approximation
        of the objective (_quad_obj_exprs) and the penalty terms created
        by approximating the constraints (_penalty_exprs)
        _quad_obj_exprs: list of expressions in the quadratic approximation of
            the objective, the sum of these expressions represents the current
            quadratic approximation of the objective
        _penalty_exprs: list of penalty term expressions
        """
        self._model = grb.Model()
        self._vars = set()

        self._cvx_obj_exprs = []
        self._noncvx_obj_exprs = []
        self._approx_obj_exprs = []

        # convex constraints are added directly to the model so there's no
        # need for a _cvx_cnt_exprs variable
        self._noncvx_cnt_exprs = []

        self._penalty_exprs = []
        self._grb_penalty_cnts = [] # hinge and abs value constraints

    def add_obj_expr(self, bound_expr):
        """
        Adds a bound expression (bound_expr) to the objective. If the objective
        is quadratic, is it added directly to the model. Otherwise, it is added
        by appending bound_expr to self._obj_exprs.

        bound_expr's var is added to self._vars so that a trust region can be
        added to var.
        """
        expr = bound_expr.expr
        if isinstance(expr, AffExpr) or isinstance(expr, QuadExpr):
            self._cvx_obj_exprs.append(bound_expr)
        else:
            self._noncvx_obj_exprs.append(bound_expr)
        self._vars.add(bound_expr.var)

    def add_cnt_expr(self, bound_expr):
        """
        Adds a bound expression (bound_expr) to the problem's constraints.
        If the constraint is linear, it is added directly to the model.
        Otherwise, the constraint is added by appending bound_expr to
        self._cnt_exprs.

        bound_expr's var is added to self._vars so that a trust region can be
        added to var.
        """
        comp_expr = bound_expr.expr
        expr = comp_expr.expr
        var = bound_expr.var
        assert isinstance(comp_expr, CompExpr)
        if isinstance(expr, AffExpr):
            # adding constraint directly into model
            grb_expr = self._aff_expr_to_grb_expr(expr, var)
            if isinstance(comp_expr, EqExpr):
                self._add_np_array_grb_cnt(grb_expr, GRB.EQUAL, comp_expr.val)
            elif isinstance(comp_expr, LEqExpr):
                self._add_np_array_grb_cnt(grb_expr, GRB.LESS_EQUAL, comp_expr.val)
        else:
            self._noncvx_cnt_exprs.append(bound_expr)
        self._vars.add(var)

    def _add_np_array_grb_cnt(self, grb_exprs, sense, val):
        for index, grb_expr in np.ndenumerate(grb_exprs):
            self._model.addConstr(grb_expr, sense, val[index])

    def _expr_to_grb_expr(self, bound_expr):
        expr = bound_expr.expr
        var = bound_expr.var

        if isinstance(expr, AffExpr):
            return self._aff_expr_to_grb_expr(expr, var)
        elif isinstance(expr, QuadExpr):
            return self._quad_expr_to_grb_expr(expr, var)

    def _quad_expr_to_grb_expr(self, quad_expr, var):
        x = var.get_grb_vars()
        return x.T.dot(quad_expr.Q.dot(x)) + quad_expr.A.dot(x) + quad_expr.b

    def _aff_expr_to_grb_expr(self, aff_expr, var):
        grb_var = var.get_grb_vars()
        return aff_expr.A.dot(grb_var) + aff_expr.b

    def optimize(self):
        """
        Calls the Gurobi optimizer on the current QP approximation

        Temporary Gurobi constraints and variables from the previous optimize
        call are deleted.

        The Gurobi objective is computed by translating all the expressions
        in the quadratically approximated objective (self._quad_obj_expr) and
        in the penalty approximation of the constraints (self._penalty_exprs)
        to Gurobi expressions, and summing them. The temporary constraints and
        variables created from the translation process are saved so that they
        can be deleted later.

        The Gurobi constraints are the linear constraints which have already
        been added to the model when constraints were added to this problem.
        """
        self._model.update()
        obj_exprs = [self._expr_to_grb_expr(bound_expr) for bound_expr in self._cvx_obj_exprs]
        for obj_expr in obj_exprs:
            assert obj_expr.shape == (1,1)
        obj_exprs = [obj_expr[0,0] for obj_expr in obj_exprs]
        obj = grb.quicksum(obj_exprs)
        self._model.setObjective(obj)
        self._model.optimize()
        self._update_vars()

        raise NotImplementedError

    def _add_expr_to_grb_obj(self, bound_expr):
        """
        Adds a BoundExpr to self's Gurobi model's objective.
        """
        expr = bound_expr.expr
        var = bound_expr.var

        if isinstance(expr, AffExpr):
            self._add_aff_expr_to_grb_model(expr, var)
        elif isinstance(expr, QuadExpr):
            self._add_quad_expr_to_grb_model(expr, var)
        elif isinstance(expr, HingeExpr):
            self._add_hinge_expr_to_grb_model(expr, var)
        elif isinstance(expr, AbsExpr):
            self._add_abs_expr_to_grb_model(expr, var)
        elif isinstance(expr, CompExpr):
            raise Exception("Comparison Expressions can't be added to the \
                Gurobi model")

    def _add_aff_expr_to_grb_model(self, aff_expr, var):
        """
        Adds an affine expression with var into self's Gurobi model's
        objective.
        """
        raise NotImplementedError

    def _add_quad_expr_to_grb_model(self, quad_expr, var):
        """
        Adds a quadratic expression with var into self's Gurobi model's
        objective.
        """
        raise NotImplementedError

    def _add_hinge_expr_to_grb_model(self, hinge_expr, var):
        """
        Adds a hinge expression with var into self's Gurobi model's
        objective by creating additional Gurobi variables and constraints
        to represent the hinge. The created Gurobi variables and
        constraints are saved so that they can be removed later.
        """
        raise NotImplementedError

    def _add_abs_expr_to_grb_model(self, abs_expr, var):
        """
        Adds an absolute value expression with var into self's Gurobi
        model's objective by creating additional Gurobi variables and
        constraints to represent the absolute value. The created Gurobi
        variables and constraints are saved so that they can be removed
        later.
        """
        raise NotImplementedError

    def add_trust_region(self, trust_region_size):
        """
        Adds the trust region for every variable
        """
        for var in self._vars:
            var.add_trust_region(trust_region_size)

    def convexify(self, penalty_coeff):
        """
        Convexifies the optimization problem by computing a QP approximation
        A quadratic approximation of the objective (self._obj_exprs) is saved
        in self._quad_obj_exprs.
        The penalty approximation of the constraints (self._cnt_exprs) is saved
        in self._penalty_exprs
        """
        raise NotImplementedError


    def get_value(self, penalty_coeff):
        """
        Returns the current value of the penalty objective.
        The penalty objective is computed by summing up all the values of the
        objective expressions self._obj_exprs and the penalty values of
        constraint expressions self._cnt_exprs.
        """
        raise NotImplementedError

    def get_approx_value(self, penalty_coeff):
        """
        Returns the current value of the penalty QP approximation by summing
        up the expression values in the convexified objective self._penalty_obj
        and expressions values of penalty terms for the constraints in
        self._convexified_obj_exprs. Note that this approximate value is
        computed with respect to when the last convexification was performed.

        TODO: check that get_approx_value returns the same value as Gurobi's
        model objVal.
        """
        raise NotImplementedError

    def _update_vars(self):
        """
        Updates the variables values
        """
        for var in self._vars:
            var.update()

    def save(self):
        """
        Saves the problem's current state by saving the values of all the
        variables.
        """
        for var in self._vars:
            var.save()

    def restore(self):
        """
        Restores the problem's state to the problem's saved state
        """
        for var in self._vars:
            var.restore()
