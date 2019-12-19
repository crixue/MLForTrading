import AbstractOptimalHoldings
import cvxpy as cvx
import numpy as np

class OptimalHoldings(AbstractOptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert (len(alpha_vector.columns) == 1)
        return cvx.Minimize(-alpha_vector.T.values[0]*weights)


    def _get_constraints(self, weights, factor_betas, risk):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            Factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """
        assert (len(factor_betas.shape) == 2)
        return [risk <= np.sqrt(self.risk_cap),
                factor_betas.T * weights <= self.weights_max,
                factor_betas.T * weights >= self.weights_min,
                sum(weights) == 0,
                sum(cvx.abs(weights)) <= 1,
                weights <= self.weights_max,
                weights >= self.weights_min]

    def __init__(self, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55, weights_min=-0.55):
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min

if __name__ == '__main__':
    opt = OptimalHoldings2()
    print()