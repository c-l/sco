import unittest
import gurobipy as grb
GRB = grb.GRB

import numpy as np
from prob import Prob, PosGRBVarManager
from expr import *
from variable import Variable


from ipdb import set_trace as st

f = lambda x: np.array([[x]])

def test_grb_var_pos(ut, grb_var):
    ut.assertTrue(grb_var.lb == 0.0 and grb_var.ub == GRB.INFINITY)

class TestProb(unittest.TestCase):

    def test_add_obj_expr_quad(self):
        quad = QuadExpr(np.eye(1), -2*np.ones((1,1)), np.zeros((1,1)))
        aff = AffExpr(-2*np.ones((1,1)), np.zeros((1,1)))
        prob = Prob()
        model = prob._model
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

        bexpr_quad = BoundExpr(quad, var)
        bexpr_aff = BoundExpr(aff, var)
        prob.add_obj_expr(bexpr_quad)
        prob.add_obj_expr(bexpr_aff)

        self.assertTrue(bexpr_aff in prob._quad_obj_exprs)
        self.assertTrue(bexpr_quad in prob._quad_obj_exprs)
        self.assertTrue(var in prob._vars)

    def test_add_obj_expr_nonquad(self):
        expr = Expr(f)
        prob = Prob()
        model = prob._model
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

        bexpr = BoundExpr(expr, var)
        prob.add_obj_expr(bexpr)

        self.assertTrue(bexpr not in prob._quad_obj_exprs)
        self.assertTrue(bexpr in prob._nonquad_obj_exprs)
        self.assertTrue(var in prob._vars)


    def test_optimize_just_quad_obj(self):
        quad = QuadExpr(np.eye(1), -2*np.ones((1,1)), np.zeros((1,1)))
        aff = AffExpr(-2*np.ones((1,1)), np.zeros((1,1)))
        prob = Prob()
        model = prob._model
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

        bexpr_quad = BoundExpr(quad, var)
        bexpr_aff = BoundExpr(aff, var)
        prob.add_obj_expr(bexpr_quad)
        prob.add_obj_expr(bexpr_aff)

        self.assertTrue(bexpr_aff in prob._quad_obj_exprs)
        self.assertTrue(bexpr_quad in prob._quad_obj_exprs)
        self.assertTrue(var in prob._vars)

        prob.optimize()
        self.assertTrue(np.allclose(var.get_value(), np.array([[2.0]])))

    def test_add_cnt_expr_eq_aff(self):
        aff = AffExpr(np.ones((1,1)), np.zeros((1,1)))
        comp = EqExpr(aff, np.array([[2]]))
        prob = Prob()
        model = prob._model
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

        bexpr = BoundExpr(comp, var)
        prob.add_cnt_expr(bexpr)

        prob.optimize()
        self.assertTrue(np.allclose(var.get_value(), np.array([[2]])))

    def test_add_cnt_leq_aff(self):
        quad = QuadExpr(np.eye(1), -2*np.ones((1,1)), np.zeros((1,1)))

        aff = AffExpr(np.ones((1,1)), np.zeros((1,1)))
        comp = LEqExpr(aff, np.array([[-4]]))
        prob = Prob()
        model = prob._model
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

        bexpr_quad = BoundExpr(quad, var)
        prob.add_obj_expr(bexpr_quad)

        bexpr = BoundExpr(comp, var)
        prob.add_cnt_expr(bexpr)

        prob.optimize()
        self.assertTrue(np.allclose(var.get_value(), np.array([[-4]])))

    def test_hinge_expr_to_grb_expr1(self):
        """
        min max(0, x+1) st. x == -4
        """
        aff = AffExpr(np.ones((1,1)), np.ones((1,1)))
        hinge = HingeExpr(aff)
        prob = Prob()
        model = prob._model

        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

        hinge_grb_expr, hinge_grb_cnt = prob._hinge_expr_to_grb_expr(hinge, var)
        model.update()
        obj = hinge_grb_expr[0,0]
        prob._model.setObjective(obj)

        aff = AffExpr(np.ones((1,1)), np.zeros((1,1)))
        comp = EqExpr(aff, np.array([[-4]]))
        bound_expr = BoundExpr(comp, var)
        prob.add_cnt_expr(bound_expr)

        prob._model.optimize()
        var.update()
        self.assertTrue(np.allclose(var.get_value(), np.array([[-4]])))
        self.assertTrue(np.allclose(obj.X, 0.0))

    def test_hinge_expr_to_grb_expr2(self):
        """
        min max(0, x+1) st. x == 1
        """
        aff = AffExpr(np.ones((1,1)), np.ones((1,1)))
        hinge = HingeExpr(aff)
        prob = Prob()
        model = prob._model

        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

        hinge_grb_expr, hinge_grb_cnt = prob._hinge_expr_to_grb_expr(hinge, var)
        model.update()
        obj = hinge_grb_expr[0,0]
        prob._model.setObjective(obj)

        aff = AffExpr(np.ones((1,1)), np.zeros((1,1)))
        comp = EqExpr(aff, np.array([[1.0]]))
        bound_expr = BoundExpr(comp, var)
        prob.add_cnt_expr(bound_expr)

        prob._model.optimize()
        var.update()
        self.assertTrue(np.allclose(var.get_value(), np.array([[1.0]])))
        self.assertTrue(np.allclose(obj.X, 2.0))

    def test_abs_expr_to_grb_expr(self):
        """
        min |x + 1| s.t. x <= -4
        """
        aff = AffExpr(np.ones((1,1)), np.ones((1,1)))
        abs_expr = AbsExpr(aff)

        aff = AffExpr(np.ones((1,1)), np.zeros((1,1)))
        comp = LEqExpr(aff, np.array([[-4]]))

        prob = Prob()
        model = prob._model


        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

        abs_grb_expr, abs_grb_cnt = prob._abs_expr_to_grb_expr(abs_expr, var)
        model.update()
        prob._model.setObjective(abs_grb_expr[0,0])

        bexpr = BoundExpr(comp, var)
        prob.add_cnt_expr(bexpr)

        prob._model.optimize()
        var.update()
        self.assertTrue(np.allclose(var.get_value(), np.array([[-4]])))
        pos = abs_grb_expr[0,0].getVar(0).X
        neg = abs_grb_expr[0,0].getVar(1).X
        self.assertTrue(np.allclose(pos, 0.0))
        self.assertTrue(np.allclose(neg, 3.0))

    def test_convexify_eq(self):
        prob = Prob()
        model = prob._model
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

        model.update()
        grb_cnt = model.addConstr(grb_var, GRB.EQUAL, 0)
        model.optimize()
        var.update()

        e = Expr(f)
        eq = EqExpr(e, np.array([[4]]))
        bexpr = BoundExpr(eq, var)
        prob.add_cnt_expr(bexpr)

        prob.convexify()
        self.assertTrue(len(prob._penalty_exprs) == 1)
        self.assertTrue(isinstance(prob._penalty_exprs[0].expr, AbsExpr))

    def test_convexify_leq(self):
        prob = Prob()
        model = prob._model
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

        model.update()
        grb_cnt = model.addConstr(grb_var, GRB.EQUAL, 0)
        model.optimize()
        var.update()

        e = Expr(f)
        eq = LEqExpr(e, np.array([[4]]))
        bexpr = BoundExpr(eq, var)
        prob.add_cnt_expr(bexpr)

        prob.convexify()
        self.assertTrue(len(prob._penalty_exprs) == 1)
        self.assertTrue(isinstance(prob._penalty_exprs[0].expr, HingeExpr))

    def test_get_value(self):
        """
        min x^2 st. x == 4
        when convexified,
        min x^2 + penalty_coeff*|x-4|
        when penalty_coeff == 1, solution is x = 0.5 and the value is 3.75
        (according to Wolfram Alpha)

        when penalty_coeff == 2, solution is x = 1.0 and the value is 7.0
        (according to Wolfram Alpha)
        """
        quad = QuadExpr(np.eye(1), np.zeros((1,1)), np.zeros((1,1)))
        e = Expr(f)
        eq = EqExpr(e, np.array([[4]]))

        prob = Prob()
        model = prob._model

        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

        obj = BoundExpr(quad, var)
        prob.add_obj_expr(obj)
        bexpr = BoundExpr(eq, var)
        prob.add_cnt_expr(bexpr)

        prob.optimize() # needed to set an initial value
        prob.convexify()
        prob.optimize(penalty_coeff=1.0)
        self.assertTrue(np.allclose(var.get_value(), np.array([[0.5]])))
        self.assertTrue(np.allclose(prob.get_value(1.0), np.array([[3.75]])))

        prob.optimize(penalty_coeff=2.0)
        self.assertTrue(np.allclose(var.get_value(), np.array([[1.0]])))
        self.assertTrue(np.allclose(prob.get_value(2.0), np.array([[7]])))

    def test_get_approx_value(self):
        """
        min x^2 st. x == 4
        when convexified,
        min x^2 + penalty_coeff*|x-4|
        when penalty_coeff == 1, solution is x = 0.5 and the value is 3.75
        (according to Wolfram Alpha)

        when penalty_coeff == 2, solution is x = 1.0 and the value is 7.0
        (according to Wolfram Alpha)
        """
        quad = QuadExpr(np.eye(1), np.zeros((1,1)), np.zeros((1,1)))
        e = Expr(f)
        eq = EqExpr(e, np.array([[4]]))

        prob = Prob()
        model = prob._model

        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

        obj = BoundExpr(quad, var)
        prob.add_obj_expr(obj)
        bexpr = BoundExpr(eq, var)
        prob.add_cnt_expr(bexpr)

        prob.optimize() # needed to set an initial value
        prob.convexify()
        prob.optimize(penalty_coeff=1.0)
        self.assertTrue(np.allclose(var.get_value(), np.array([[0.5]])))
        self.assertTrue(np.allclose(prob.get_approx_value(1.0), np.array([[3.75]])))

        prob.optimize(penalty_coeff=2.0)
        self.assertTrue(np.allclose(var.get_value(), np.array([[1.0]])))
        self.assertTrue(np.allclose(prob.get_approx_value(2.0), np.array([[7]])))

    def test_pos_grb_var_manager(self):
        prob = Prob()
        model = prob._model
        init_num = 1
        inc_num = 10
        shape = (2,7)
        pos_grb_manager = PosGRBVarManager(model, init_num=init_num, inc_num=inc_num)
        pgm = pos_grb_manager
        PosGRBVarManager.INC_NUM = 2
        self.assertTrue(len(pgm._grb_vars) == init_num)
        var = pgm.next()
        test_grb_var_pos(self, var)
        self.assertTrue(len(pgm._grb_vars) == inc_num+init_num)
        a = pgm.get_array(shape)
        for x in a.flatten():
            test_grb_var_pos(self, x)

        self.assertTrue(a.shape == shape)
        self.assertTrue(len(pgm._grb_vars) == init_num+2*inc_num)

if __name__ == '__main__':
    unittest.main()
