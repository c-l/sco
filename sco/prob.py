import gurobipy as grb
GRB = grb.GRB
from expr import *
from ipdb import set_trace as st


class Prob(object):
    """
    Sequential convex programming problem with a scalar objective. A solution is
    found using the l1 penalty method.
    """

    def __init__(self, grb_model, callback=None):
        """
        _model: Gurobi model associated with this problem
        _vars: variables in this problem

        _quad_obj_exprs: list of quadratic expressions in the objective
        _nonquad_obj_exprs: list of non-quadratic expressions in the objective
        _approx_obj_exprs: list of approximation of the non-quadratic
            expressions in the objective

        _nonlin_cnt_exprs: list of non-linear constraint expressions
        _penalty_exprs: list of penalty term expressions (approximations of the
            non-linear constraint expressions in _nonlin_cnt_exprs)

        _grb_penalty_cnts: list of Gurobi constraints that are generated when
            adding the hinge and absolute value terms from the penalty terms.
        _pgm: Positive Gurobi variable manager provides a lazy way of generating
            positive Gurobi variables so that there are less model updates.

        _cnt_to_cvx: dictionary that maps constraint expressions to its
            corresponding convex expression

        _bexpr_to_grb_expr: dictionary that caches quadratic bound expressions
            with their corresponding Gurobi expression
        """
        self._model = grb_model
        self._model.params.OutputFlag = 0 # silences Gurobi output
        self._vars = set()
        if callback is not None:
            self._callback = callback
        else:
            def do_nothing():
                pass
            self._callback = do_nothing

        self._quad_obj_exprs = []
        self._nonquad_obj_exprs = []
        self._approx_obj_exprs = []

        # linear constraints are added directly to the model so there's no
        # need for a _lin_cnt_exprs variable
        self._nonlin_cnt_exprs = []

        self._penalty_exprs = []
        self._grb_penalty_cnts = [] # hinge and abs value constraints
        self._pgm = PosGRBVarManager(self._model)

        self._cnt_to_cvx = {}

        self._bexpr_to_grb_expr = {}

        # TODO: handle non-quadratic objectives for computing param value
        self._grb_var_to_quad_obj_bexprs = {}
        self._grb_var_to_nonlin_cnt_bexprs = {}
        self._grb_var_to_penalty_cnt_bexprs = {}

    def add_obj_expr(self, bound_expr):
        """
        Adds a bound expression (bound_expr) to the objective. If the objective
        is quadratic, is it added to _quad_obj_exprs. Otherwise, it is added
        to self._nonquad_obj_exprs.

        bound_expr's var is added to self._vars so that a trust region can be
        added to var.
        """
        expr = bound_expr.expr
        if isinstance(expr, AffExpr) or isinstance(expr, QuadExpr):
            self._quad_obj_exprs.append(bound_expr)
            self._add_bexpr_to_grb_var_to_obj_map(bound_expr)
        else:
            self._nonquad_obj_exprs.append(bound_expr)
        self._vars.add(bound_expr.var)

    def add_cnt_expr(self, bound_expr):
        """
        Adds a bound expression (bound_expr) to the problem's constraints.
        If the constraint is linear, it is added directly to the model.
        Otherwise, the constraint is added by appending bound_expr to
        self._nonlin_cnt_exprs.

        bound_expr's var is added to self._vars so that a trust region can be
        added to var.
        """
        comp_expr = bound_expr.expr
        expr = comp_expr.expr
        var = bound_expr.var
        assert isinstance(comp_expr, CompExpr)
        if isinstance(expr, AffExpr):
            # adding constraint directly into model
            grb_expr, grb_cnt = self._aff_expr_to_grb_expr(expr, var)
            if isinstance(comp_expr, EqExpr):
                self._add_np_array_grb_cnt(grb_expr, GRB.EQUAL, comp_expr.val)
            elif isinstance(comp_expr, LEqExpr):
                self._add_np_array_grb_cnt(grb_expr, GRB.LESS_EQUAL, comp_expr.val)
        else:
            self._nonlin_cnt_exprs.append(bound_expr)
            self._add_bexpr_to_grb_var_to_nonlin_cnt_map(bound_expr)
        self._vars.add(var)

    def _add_bexpr_to_grb_var_map(self, bexpr, mapping):
        var = bexpr.var
        grb_vars = var.get_grb_vars()
        for grb_var in grb_vars.flat:
            if grb_var not in mapping:
                mapping[grb_var] = [bexpr]
            else:
                mapping[grb_var].append(bexpr)

    def _add_bexpr_to_grb_var_to_obj_map(self, bexpr):
        self._add_bexpr_to_grb_var_map(bexpr, self._grb_var_to_quad_obj_bexprs)

    def _add_bexpr_to_grb_var_to_nonlin_cnt_map(self, bexpr):
        self._add_bexpr_to_grb_var_map(bexpr, self._grb_var_to_nonlin_cnt_bexprs)

    def _add_bexpr_to_grb_var_penalty_cnt_map(self, bexpr):
        self._add_bexpr_to_grb_var_map(bexpr, self._grb_var_to_penalty_cnt_bexprs)

    def _add_np_array_grb_cnt(self, grb_exprs, sense, val):
        """
        Adds a numpy array of Gurobi constraints to the model and returns
        the constraints.
        """
        cnts = []
        for index, grb_expr in np.ndenumerate(grb_exprs):
            cnts.append(self._model.addConstr(grb_expr, sense, val[index]))
        return cnts

    def _expr_to_grb_expr(self, bound_expr):
        """
        Translates AffExpr, QuadExpr, HingeExpr and AbsExpr to Gurobi
        expressions and returns the corresponding Gurobi expressions and
        constraints. If there are no Gurobi constraints, an empty list is
        returned. Otherwise, this method raises an exception.
        """
        expr = bound_expr.expr
        var = bound_expr.var

        if isinstance(expr, AffExpr):
            return self._aff_expr_to_grb_expr(expr, var)
        elif isinstance(expr, QuadExpr):
            if bound_expr in self._bexpr_to_grb_expr:
                return self._bexpr_to_grb_expr[bound_expr], []
            else:
                grb_expr, cnts = self._quad_expr_to_grb_expr(expr, var)
                self._bexpr_to_grb_expr[bound_expr] = grb_expr
                return grb_expr, cnts
        elif isinstance(expr, HingeExpr):
            return self._hinge_expr_to_grb_expr(expr, var)
        elif isinstance(expr, AbsExpr):
            return self._abs_expr_to_grb_expr(expr, var)
        elif isinstance(expr, CompExpr):
            raise Exception("Comparison Expressions cannot be converted to \
                a Gurobi expression. Use add_cnt_expr instead")
        else:
            raise Exception("This type of Expression cannot be converted to\
                a Gurobi expression.")

    def _aff_expr_to_grb_expr(self, aff_expr, var):
        grb_var = var.get_grb_vars()
        return aff_expr.A.dot(grb_var) + aff_expr.b, []

    def _quad_expr_to_grb_expr(self, quad_expr, var):
        x = var.get_grb_vars()
        grb_expr = grb.QuadExpr()
        Q = quad_expr.Q
        rows, cols = x.shape
        assert cols == 1
        for i in range(rows):
            for j in range(rows):
                if Q[i][j] != 0:
                    grb_expr += Q[i][j]*x[i,0]*x[j,0]

        grb_expr = np.array([[0.5*grb_expr]])
        grb_expr = grb_expr + quad_expr.A.dot(x)
        grb_expr = grb_expr + quad_expr.b
        return grb_expr, []

    def _hinge_expr_to_grb_expr(self, hinge_expr, var):
        aff_expr = hinge_expr.expr
        assert isinstance(aff_expr, AffExpr)
        grb_expr, _ = self._aff_expr_to_grb_expr(aff_expr, var)
        grb_hinge = self._pgm.get_array(grb_expr.shape)
        cnts = self._add_np_array_grb_cnt(grb_expr, GRB.LESS_EQUAL, grb_hinge)
        return grb_hinge, cnts

    def _abs_expr_to_grb_expr(self, abs_expr, var):
        aff_expr = abs_expr.expr
        assert isinstance(aff_expr, AffExpr)
        grb_expr, _ = self._aff_expr_to_grb_expr(aff_expr, var)
        pos = self._pgm.get_array(grb_expr.shape)
        neg = self._pgm.get_array(grb_expr.shape)
        cnts = self._add_np_array_grb_cnt(grb_expr, GRB.EQUAL, pos-neg)
        return pos+neg, cnts

    def find_closest_feasible_point(self):
        """
        Finds the closest point (l2 norm) to the initialization that satisfies
        the linear constraints.
        """
        self._del_old_grb_cnts()
        self._model.update()

        obj = grb.QuadExpr()
        for var in self._vars:
            g_var = var.get_grb_vars()
            val = var.get_value()
            if val is not None:
                assert g_var.shape == val.shape
                for i in np.ndindex(g_var.shape):
                    if not np.isnan(val[i]):
                        obj += g_var[i]*g_var[i] - 2*val[i]*g_var[i] + val[i]*val[i]

        grb_exprs = []
        for bound_expr in self._quad_obj_exprs:
            grb_expr, grb_cnts = self._expr_to_grb_expr(bound_expr)
            self._grb_penalty_cnts.extend(grb_cnts)
            grb_exprs.extend(grb_expr.flatten().tolist())

        obj += grb.quicksum(grb_exprs)

        self._model.setObjective(obj)
        self._model.optimize()
        if self._model.status != 2:
            return False
        
        # if self._model.status == 3:
        #     self._model.optimize()
        # if self._model.status == 4:
        #     self._model.computeIIS()
        #     self._model.write('infeasible.ilp')
        #     raise Exception('Failed to satisfy linear equalities. Infeasible Constraint set written to infeasible.ilp')
        # elif self._model.status != 2:
        #     self._model.write('infeasible.lp')
        #     raise Exception('Failed to satisfy linear equalities. Infeasible Constraint set written to infeasible.lp')
        self._update_vars()
        self._callback()
        return True

    def optimize(self, penalty_coeff=0.0):
        """
        Calls the Gurobi optimizer on the current QP approximation with a given
        penalty coefficient.

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
        self._del_old_grb_cnts()
        self._model.update()

        grb_exprs = []
        for bound_expr in self._quad_obj_exprs + self._approx_obj_exprs:
            grb_expr, grb_cnts = self._expr_to_grb_expr(bound_expr)
            self._grb_penalty_cnts.extend(grb_cnts)
            grb_exprs.extend(grb_expr.flatten().tolist())

        for bound_expr in self._penalty_exprs:
            grb_expr, grb_cnts = self._expr_to_grb_expr(bound_expr)
            grb_expr *= penalty_coeff
            self._grb_penalty_cnts.extend(grb_cnts)
            grb_exprs.extend(grb_expr.flatten().tolist())

        obj = grb.quicksum(grb_exprs)
        self._model.setObjective(obj)
        self._model.optimize()
        self._update_vars()
        self._callback()

    def _del_old_grb_cnts(self):
        for cnt in self._grb_penalty_cnts:
            self._model.remove(cnt)

    def add_trust_region(self, trust_region_size):
        """
        Adds the trust region for every variable
        """
        for var in self._vars:
            var.add_trust_region(trust_region_size)

    def convexify(self):
        """
        Convexifies the optimization problem by computing a QP approximation
        A quadratic approximation of the non-quadratic objective terms
        (self._nonquad_obj_exprs) is saved in self._approx_obj_exprs.
        The penalty approximation of the non-linear constraints
        (self._nonlin_cnt_exprs) is saved in self._penalty_exprs
        The map between constraint expressions to its convexified expressions
        is updated.
        """
        self._approx_obj_exprs = [bexpr.convexify(degree = 2) \
            for bexpr in self._nonquad_obj_exprs]
        self._penalty_exprs = []
        self._grb_var_to_penalty_cnt_bexprs = {}
        for bexpr in self._nonlin_cnt_exprs:
            cvx_bexpr = bexpr.convexify(degree = 1)
            self._add_bexpr_to_grb_var_penalty_cnt_map(cvx_bexpr)
            self._cnt_to_cvx[bexpr] = cvx_bexpr
            self._penalty_exprs.append(cvx_bexpr)

    def get_value(self, penalty_coeff):
        """
        Returns the current value of the penalty objective.
        The penalty objective is computed by summing up all the values of the
        quadratic objective expressions (self._quad_obj_exprs), the
        non-quadratic objective expressions and the penalty coeff multiplied
        by the constraint violations (computed using _nonlin_cnt_exprs)
        """
        value = 0.0
        for bound_expr in self._quad_obj_exprs + self._nonquad_obj_exprs:
            value += np.sum(bound_expr.eval())
        for bound_expr in self._nonlin_cnt_exprs:
            value += penalty_coeff*self._get_cnt_bexpr_value(bound_expr)
        return value

    def _compute_cnt_violation(self, bexpr):
        comp_expr = bexpr.expr
        var_val = bexpr.var.get_value()
        if isinstance(comp_expr, EqExpr):
            return np.absolute(comp_expr.expr.eval(var_val) - comp_expr.val)
        elif isinstance(comp_expr, LEqExpr):
            v = comp_expr.expr.eval(var_val) - comp_expr.val
            zeros = np.zeros(v.shape)
            return np.maximum(v, zeros)

    def get_max_cnt_violation(self):
        """
        Returns the the maximum amount a non-linear constraint is violated.
        Linear constraints are assumed to be satisfied because they are added
        directly to the model and QP solvers can deal with them.
        """
        max_vio = 0.0
        for bound_expr in self._nonlin_cnt_exprs:
            cnt_vio = self._compute_cnt_violation(bound_expr)
            cnt_max_vio = np.amax(cnt_vio)
            max_vio = np.maximum(max_vio, cnt_max_vio)
        return max_vio

    def get_approx_value(self, penalty_coeff):
        """
        Returns the current value of the penalty QP approximation by summing
        up the expression values for the quadratic objective terms
        (_quad_obj_exprs), the quadratic approximation of the non-quadratic
        terms (_approx_obj_exprs) and the penalty terms (_penalty_exprs).
        Note that this approximate value is computed with respect to when the
        last convexification was performed.
        """
        value = 0.0
        for bound_expr in self._quad_obj_exprs + self._approx_obj_exprs:
            value += np.sum(bound_expr.eval())
        for bound_expr in self._penalty_exprs:
            value += penalty_coeff*self._get_cnt_approx_bexpr_value(bound_expr)
        return value

    def _get_cnt_bexpr_value(self, bexpr):
        cnt_vio = self._compute_cnt_violation(bexpr)
        return np.sum(cnt_vio)

    def _get_cnt_approx_bexpr_value(self, cvx_bexpr):
        return np.sum(cvx_bexpr.eval())

    def get_cnt_values(self, penalty_coeff):
        """
        Returns a dictionary which maps the bound expressions of non-linear
        constraints to its value (how much it's violated)
        """
        bexpr_to_val = {}
        for bound_expr in self._nonlin_cnt_exprs:
            bexpr_to_val[bound_expr] = penalty_coeff*self._get_cnt_bexpr_value(bound_expr)
        return bexpr_to_val

    def get_cnt_approx_values(self, penalty_coeff):
        """
        Returns a dictionary which maps the bound expressions of non-linear
        constraints to its approximate value (how much it's violated)
        """
        bexpr_to_val = {}
        for bexpr, cvx_bexpr in self._cnt_to_cvx.iteritems():
            bexpr_to_val[bexpr] = penalty_coeff*self._get_cnt_approx_bexpr_value(cvx_bexpr)
        return bexpr_to_val

    def get_shared_cnt_values(self, penalty_coeff):
        bexpr_to_val = {}
        for bound_expr in self._nonlin_cnt_exprs:
            val = 0.
            for grb_var in bound_expr.var.get_grb_vars().flat:
                # add in value of all the non-linear constraints which share the
                # same Gurobi variable
                nonlin_cnt_bexprs = self._grb_var_to_nonlin_cnt_bexprs[grb_var]
                for be in nonlin_cnt_bexprs:
                    val += penalty_coeff*self._get_cnt_bexpr_value(be)

                # add in value of all the objectives which share the same
                # Gurobi variable
                if grb_var in self._grb_var_to_quad_obj_bexprs:
                    obj_bexprs = self._grb_var_to_quad_obj_bexprs[grb_var]
                    for be in obj_bexprs:
                        val += np.sum(be.eval())
            bexpr_to_val[bound_expr] = val
        return bexpr_to_val

    def get_shared_cnt_approx_values(self, penalty_coeff):
        bexpr_to_val = {}
        for bound_expr in self._nonlin_cnt_exprs:
            val = 0.
            for grb_var in bound_expr.var.get_grb_vars().flat:
                # add in value of all the linear convexification of the
                # non-linear constraints which share the same Gurobi variable
                nonlin_cnt_bexprs = self._grb_var_to_nonlin_cnt_bexprs[grb_var]
                penalty_cnt_bexprs = [self._cnt_to_cvx[be] for be in nonlin_cnt_bexprs]
                for be in penalty_cnt_bexprs:
                    val += penalty_coeff*self._get_cnt_approx_bexpr_value(be)

                # add in value of all the objectives which share the same
                # Gurobi variable
                if grb_var in self._grb_var_to_quad_obj_bexprs:
                    obj_bexprs = self._grb_var_to_quad_obj_bexprs[grb_var]
                    for be in obj_bexprs:
                        val += np.sum(be.eval())
            bexpr_to_val[bound_expr] = val
        return bexpr_to_val

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

class PosGRBVarManager(object):
    """
    Manages positive Gurobi variables. The purpose of the manager is to create
    many Gurobi variables at once to decrease the number of Gurobi model update
    because model updates take a long time.
    """
    INIT_NUM = 1000
    INC_NUM = 1000

    def __init__(self, model, init_num=INC_NUM, inc_num=INC_NUM):
        self._index = 0
        self._model = model
        self._grb_vars = []
        self._add_grb_vars(init_num)
        self._inc_num = inc_num

    def _add_grb_vars(self, num=None):
        """
        Creates a batch of positive Gurobi variables so that the model is
        updated less often.
        """
        if num is None:
            num = self._inc_num
        new_grb_vars = [self._model.addVar(lb=0.0, ub=GRB.INFINITY) \
                            for i in range(num)]
        self._grb_vars.extend(new_grb_vars)
        self._model.update()

    def next(self):
        """
        Returns one positive Gurobi variable.
        """
        if self._index == len(self._grb_vars)-1:
            self._add_grb_vars()
        self._index += 1
        return self._grb_vars[self._index-1]

    def get_array(self, shape):
        """
        Returns a numpy array of unused positive Gurobi variables.
        """
        a = np.empty(shape, dtype=object)
        for x in np.nditer(a, op_flags=['readwrite'], flags=['refs_ok']):
            x[...] = self.next()
        return a

    def reset():
        self._index = 0
