import unittest
import gurobipy as grb
GRB = grb.GRB

import numpy as np
from prob import Prob
from expr import *
from variable import Variable


from ipdb import set_trace as st

f = lambda x: np.array([[x]])


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

        self.assertTrue(bexpr_aff in prob._cvx_obj_exprs)
        self.assertTrue(bexpr_quad in prob._cvx_obj_exprs)
        self.assertTrue(var in prob._vars)

    def test_add_obj_expr_noncvx(self):
        expr = Expr(f)
        prob = Prob()
        model = prob._model
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x')
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

        bexpr = BoundExpr(expr, var)
        prob.add_obj_expr(bexpr)

        self.assertTrue(bexpr not in prob._cvx_obj_exprs)
        self.assertTrue(bexpr in prob._noncvx_obj_exprs)
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

        self.assertTrue(bexpr_aff in prob._cvx_obj_exprs)
        self.assertTrue(bexpr_quad in prob._cvx_obj_exprs)
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


if __name__ == '__main__':
    unittest.main()
