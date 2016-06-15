import unittest
import gurobipy as grb
GRB = grb.GRB

from solver import Solver
from prob import Prob
from variable import Variable
from expr import *

import numpy as np
from ipdb import set_trace as st

solv = Solver()
"""
values taken from Pieter Abbeel's CS287 hw3 q2 penalty_sqp.m file
"""
solv.improve_ratio_threshold = .25
solv.min_trust_region_size = 1e-5
solv.min_approx_improve = 1e-8
solv.max_iter = 50
solv.trust_shrink_ratio = .1
solv.trust_expand_ratio = 1.5
solv.cnt_tolerance = 1e-4
solv.max_merit_coeff_increases = 5
solv.merit_coeff_increase_ratio = 10
solv.initial_trust_region_size = 1
solv.initial_penalty_coeff = 1.

zerofunc = lambda x: np.array([[0.0]])
neginffunc = lambda x: np.array([[-1e5]])
N = 2

def test_prob(ut, x0, x_true, f=zerofunc, g=neginffunc, h=zerofunc,
    Q=np.zeros((N,N)), q=np.zeros((1,N)), A_ineq=np.zeros((1,N)),
    b_ineq=np.zeros((1,1)), A_eq=np.zeros((1,1)), b_eq=np.zeros((1,1))):

    prob = Prob()
    model = prob._model

    grb_var1 = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x1')
    grb_var2 = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name='x2')
    grb_vars = np.array([[grb_var1], [grb_var2]])
    var = Variable(grb_vars, value=x0)

    quad_obj = BoundExpr(QuadExpr(Q, q, np.zeros((1,1))), var)
    prob.add_obj_expr(quad_obj)
    nonquad_obj = BoundExpr(Expr(f), var)
    prob.add_obj_expr(nonquad_obj)

    cnts = []
    nonlin_ineq = LEqExpr(Expr(g), np.zeros(g(np.zeros((2,1))).shape))
    nonlin_ineq = BoundExpr(nonlin_ineq, var)
    cnts.append(nonlin_ineq)
    nonlin_eq = EqExpr(Expr(h), np.zeros(g(np.zeros((2,1))).shape))
    nonlin_eq = BoundExpr(nonlin_eq, var)
    cnts.append(nonlin_eq)

    for cnt in cnts:
        prob.add_cnt_expr(cnt)

    solv.solve(prob, method='penalty_sqp')
    x_sol = var.get_value()
    if not np.allclose(x_sol, x_true, atol=1e-4): st()
    ut.assertTrue(np.allclose(x_sol, x_true, atol=1e-4))


class TestSolver(unittest.TestCase):

    def test_prob0(self):
        x0 = np.array([[1.0],[1.0]])
        f = lambda x: np.array([[x[0,0]**2+x[1,0]**2]])
        g = lambda x: np.array([[3 - x[0,0] - x[1,0]]])
        x_true = np.array([[1.5],[1.5]])
        test_prob(self, x0, x_true, f=f, g=g)

    def test_prob1(self):
        x0 = np.array([[-2.0],[1.0]])
        f = lambda x: np.array([[(x[1,0]-x[0,0]**2)**2 + (1-x[0,0])**2]])
        g = lambda x: np.array([[-1.5 - x[1,0]]])
        x_true = np.array([[1.0],[1.0]])
        test_prob(self, x0, x_true, f=f, g=g)

    def test_prob2(self):
        x0 = np.array([[10.0],[1.0]])
        f = lambda x: np.array([[x[1,0] + 1e-5 + (x[1,0]-x[0,0])**2]])
        h = lambda x: np.array([[-x[1,0]]])
        x_true = np.array([[0.0],[0.0]])
        test_prob(self, x0, x_true, f=f, h=h)

    def test_prob3(self):
        x0 = np.array([[10.0],[1.0]])
        f = lambda x: np.array([[(1-x[0,0])**2]])
        h = lambda x: np.array([[10*(x[1,0]-x[0,0]**2)]])
        x_true = np.array([[1.0],[1.0]])
        test_prob(self, x0, x_true, f=f, h=h)

    def test_prob4(self):
        x0 = np.array([[2.0],[2.0]])
        f = lambda x: np.array([[np.log(1+x[0,0]**2)-x[1,0]]])
        h = lambda x: np.array([[(1+x[0,0]**2)**2+x[1,0]**2-4]])
        x_true = np.array([[0.0],[np.sqrt(3)]])
        test_prob(self, x0, x_true, f=f, h=h)


if __name__ == '__main__':
    unittest.main()
