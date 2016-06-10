import unittest
from expr import Expr, AffExpr, QuadExpr
from expr import AbsExpr, HingeExpr
from expr import CompExpr, EqExpr, LEqExpr
from expr import BoundExpr

from variable import Variable

import numpy as np

from ipdb import set_trace as st

fs = [(lambda x: np.array([x]), lambda x: np.array([1]),
        lambda x: np.array([0])),
      (lambda x: np.array([x**2]), lambda x: np.array([2*x]),
        lambda x: np.array([2])),
      (lambda x: np.array([x**3]), lambda x: np.array([3*x**2]),
        lambda x: np.array([6*x]))]
xs = [1., 2., -1., 0.]
N = 10
d = 10

def test_expr_val_grad(ut, e, x, y, y_prime):
    y = np.array(y)
    y_prime = np.array(y_prime)
    y_e = np.array(e.eval(x))
    ut.assertTrue(np.allclose(y_e, y))
    y_prime_e = np.array(e.grad(x))
    ut.assertTrue(np.allclose(y_prime_e, y_prime))

def test_expr_val_grad_hess(ut, e, x, y, y_prime, y_d_prime):
    y = np.array(y)
    y_prime = np.array(y_prime)
    y_e = np.array(e.eval(x))
    ut.assertTrue(np.allclose(y_e, y))
    y_prime_e = np.array(e.grad(x))
    ut.assertTrue(np.allclose(y_prime_e, y_prime))
    y_d_prime_e = np.array(e.hess(x))
    ut.assertTrue(np.allclose(y_d_prime_e, y_d_prime))

class TestExpr(unittest.TestCase):

    def test_expr_eval_grad(self):
        for f, fder, fhess in fs:
            e = Expr(f)
            for x in xs:
                y = f(x)
                y_prime = fder(x)
                y_d_prime = fhess(x)
                test_expr_val_grad_hess(self, e, x, y, y_prime, y_d_prime)

    def test_convexify_deg_1(self):
        for f, fder, _ in fs:
            e = Expr(f)
            for x in xs:
                y = f(x)
                y_prime = fder(x)

                aff_e = e.convexify(x, degree=1)
                self.assertIsInstance(aff_e, AffExpr)
                A = aff_e.A
                b = aff_e.b
                self.assertTrue(np.allclose(A, fder(x)))
                self.assertTrue(np.allclose(b, f(x) - A.dot(x)))
                self.assertTrue(np.allclose(aff_e.eval(x), f(x)))

    def test_convexify_deg_2(self):
        for f, fder, fhess in fs:
            e = Expr(f)
            for x in xs:
                x = np.array([x])
                y = f(x)
                y_prime = fder(x)
                y_d_prime = fhess(x)

                quad_e = e.convexify(x, degree=2)
                self.assertIsInstance(quad_e, QuadExpr)
                Q = quad_e.Q
                A = quad_e.A
                b = quad_e.b
                self.assertTrue(np.allclose(Q, y_d_prime))
                self.assertTrue(np.allclose(A, \
                    y_prime -2*np.transpose(x).dot(y_d_prime)))
                self.assertTrue(np.allclose(b, \
                    np.array(np.transpose(x).dot(y_d_prime)).dot(x)\
                                - y_prime.dot(x) + y))
                self.assertTrue(np.allclose(quad_e.eval(x), y))


class TestAffExpr(unittest.TestCase):

    def test_aff_expr_eval_grad_hess(self):
        for _ in range(N):
            A = np.random.rand(d, d)
            b = np.random.rand(d, 1)
            x = np.random.rand(d, 1)
            y = A.dot(x) + b
            y_prime = A.T
            e = AffExpr(A, b)
            test_expr_val_grad(self, e, x, y, y_prime)

            hess = np.zeros((d,d))
            self.assertTrue(np.allclose(e.hess(b), hess))
            self.assertTrue(np.allclose(e.hess(np.ones((d,1))), hess))
            self.assertTrue(np.allclose(e.hess(np.zeros((d,1))), hess))
            self.assertTrue(np.allclose(e.hess(x), hess))

class TestQuadExpr(unittest.TestCase):

    def test_quad_expr_eval_grad_hess(self):
        for _ in range(N):
            A = np.random.rand(1, d)
            b = np.random.rand(1)
            Q = np.random.rand(d, d)
            x = np.random.rand(d, 1)
            y = x.T.dot(Q.dot(x)) + A.dot(x) + b
            y_prime = Q.T.dot(x) + Q.dot(x) + A.T
            e = QuadExpr(Q, A, b)

            test_expr_val_grad(self, e, x, y, y_prime)
            hess = Q
            self.assertTrue(np.allclose(e.hess(b), hess))
            self.assertTrue(np.allclose(e.hess(np.ones((d,1))), hess))
            self.assertTrue(np.allclose(e.hess(np.zeros((d,1))), hess))
            self.assertTrue(np.allclose(e.hess(x), hess))

class TestAbsExpr(unittest.TestCase):

    def test_abs_expr_eval(self):
        for _ in range(N):
            A = np.random.rand(d, d) - 0.5*np.ones((d,d))
            b = np.random.rand(d, 1) - 0.5*np.ones((d,1))
            x = np.random.rand(d, 1) - 0.5*np.ones((d,1))
            y = A.dot(x) + b
            y_prime = A.T
            e = AffExpr(A, b)
            abs_e = AbsExpr(e)
            self.assertTrue(np.allclose(np.absolute(e.eval(x)), abs_e.eval(x)))

class TestHingeExpr(unittest.TestCase):

    def test_hinge_expr_eval(self):
        for _ in range(N):
            A = np.random.rand(d, d) - 0.5*np.ones((d,d))
            b = np.random.rand(d, 1) - 0.5*np.ones((d,1))
            x = np.random.rand(d, 1) - 0.5*np.ones((d,1))
            y = A.dot(x) + b
            y_prime = A.T
            e = AffExpr(A, b)
            hinge_e = HingeExpr(e)
            zeros = np.zeros((1,1))
            self.assertTrue(np.allclose(np.maximum(e.eval(x), zeros), hinge_e.eval(x)))

class TestCompExpr(unittest.TestCase):

    def test_comp_expr(self):
        f, fder, _ = fs[0]
        e = Expr(f)
        val = np.array([0])
        comp_e = CompExpr(e, val)
        self.assertEqual(comp_e.expr, e)
        self.assertTrue(np.allclose(comp_e.val, val))

        # check to ensure that modifying val won't modifying the comp expr
        val[0] = 1
        self.assertTrue(not np.allclose(comp_e.val, val))

        with self.assertRaises(NotImplementedError) as nie:
            comp_e.eval(0)
        with self.assertRaises(NotImplementedError) as nie:
            comp_e.convexify(0)
        with self.assertRaises(Exception) as e:
            comp_e.grad(0)

class TestEqExpr(unittest.TestCase):

    def test_eq_expr_eval(self):
        for _ in range(N):
            A = np.random.rand(d, d) - 0.5*np.ones((d,d))
            b = np.random.rand(d, 1) - 0.5*np.ones((d,1))
            x = np.random.rand(d, 1) - 0.5*np.ones((d,1))
            y = A.dot(x) + b
            y_prime = A.T
            e = AffExpr(A, b)

            val = e.eval(x)
            eq_e = EqExpr(e, val)
            self.assertTrue(eq_e.eval(x, tol=0.0))
            self.assertTrue(eq_e.eval(x, tol=0.01))

            val = e.eval(x) + 0.1
            eq_e = EqExpr(e, val)
            self.assertFalse(eq_e.eval(x, tol=0.01))
            self.assertTrue(eq_e.eval(x, tol=0.1))

    def test_eq_expr_convexify(self):
        for f, fder, _ in fs:
            e = Expr(f)
            for x in xs:
                y = f(x)
                y_prime = fder(x)

                eq_e = EqExpr(e, np.array([1.0]))
                abs_e = eq_e.convexify(x)
                self.assertIsInstance(abs_e, AbsExpr)

                aff_e = abs_e.expr
                A = aff_e.A
                b = aff_e.b
                self.assertTrue(np.allclose(A, fder(x)))
                self.assertTrue(np.allclose(b, f(x) - A.dot(x) - 1.0))

                self.assertTrue(np.allclose(abs_e.eval(x), np.absolute(y-1.0)))
                x2 = x+1.0
                self.assertTrue(np.allclose(abs_e.eval(x2), np.absolute(A.dot(x2)+b)))

class TestLEqExpr(unittest.TestCase):

    def test_leq_expr_eval(self):
        for _ in range(N):
            A = np.random.rand(d, d) - 0.5*np.ones((d,d))
            b = np.random.rand(d, 1) - 0.5*np.ones((d,1))
            x = np.random.rand(d, 1) - 0.5*np.ones((d,1))
            y = A.dot(x) + b
            y_prime = A.T
            e = AffExpr(A, b)

            val = e.eval(x)
            leq_e = LEqExpr(e, val)
            self.assertTrue(leq_e.eval(x, tol=0.0))
            self.assertTrue(leq_e.eval(x, tol=0.01))

            val = e.eval(x) + 0.1
            leq_e = LEqExpr(e, val)
            self.assertTrue(leq_e.eval(x, tol=0.01))
            self.assertTrue(leq_e.eval(x, tol=0.1))

            val = e.eval(x) - 0.1
            leq_e = LEqExpr(e, val)
            self.assertFalse(leq_e.eval(x, tol=0.01))
            self.assertTrue(leq_e.eval(x, tol=0.1+1e-8))

    def test_leq_expr_convexify(self):
        for f, fder, _ in fs:
            e = Expr(f)
            for x in xs:
                y = f(x)
                y_prime = fder(x)

                leq_e = LEqExpr(e, np.array([1.0]))
                hinge_e = leq_e.convexify(x)
                self.assertIsInstance(hinge_e, HingeExpr)

                aff_e = hinge_e.expr
                A = aff_e.A
                b = aff_e.b
                self.assertTrue(np.allclose(A, fder(x)))
                self.assertTrue(np.allclose(b, f(x) - A.dot(x) - 1.0))

                self.assertTrue(np.allclose(hinge_e.eval(x), np.maximum(y-1.0, np.zeros(y.shape))))

                x2 = x+1.0
                self.assertTrue(np.allclose(hinge_e.eval(x2), np.maximum(A.dot(x2)+b, np.zeros(y.shape))))

class TestBoundExpr(unittest.TestCase):

    def test_bound_expr(self):
        b_e = BoundExpr(1,2)
        self.assertEqual(b_e.expr, 1)
        self.assertEqual(b_e.var, 2)

    def test_bound_expr_eval_convexify(self):
        for f, fder, _ in fs:
            e = Expr(f)
            for x in xs:
                y = f(x)

                dummy_grb_vars = np.array([1])
                value = np.array([x])
                v = Variable(dummy_grb_vars, value)

                b_e = BoundExpr(e, v)
                self.assertTrue(np.allclose(b_e.eval(), e.eval(x)))

                cvx_b_e = b_e.convexify()
                self.assertIsInstance(cvx_b_e, BoundExpr)
                self.assertEqual(cvx_b_e.var, v)

                cvx_e = b_e.expr.convexify(value)
                self.assertTrue(np.allclose(cvx_e.A, cvx_b_e.expr.A))
                self.assertTrue(np.allclose(cvx_e.b, cvx_b_e.expr.b))

if __name__ == '__main__':
    unittest.main()
