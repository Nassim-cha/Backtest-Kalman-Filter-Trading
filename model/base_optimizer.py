import numpy as np


class BaseOptimizer:
    def __init__(self, a, l, q, max_iter=100, tol=1e-7, perturb_scale=0.05, restart_attempts=5):
        self.a = a
        self.l = l
        self.q = q
        self.max_iter = max_iter
        self.tol = tol
        self.perturb_scale = perturb_scale
        self.restart_attempts = restart_attempts
        self.best_LL = -np.inf
        self.best_params = np.array([a, l, q])
        self.n_steps = None  # pre-allocate

    def compute_gradient_hessian(self, params, *args):
        """Compute the gradient and Hessian of the log-likelihood function."""
        epsilon = 1e-5  # Small perturbation for finite differences
        grad = np.zeros_like(params)
        hess = np.zeros((len(params), len(params)))

        # Compute log-likelihood at the current parameter values
        base_LL = self.log_likelihood(params, *args)

        # Compute gradient using finite differences
        for i in range(len(params)):
            params_perturbed = params.copy()
            params_perturbed[i] += epsilon
            perturbed_LL = self.log_likelihood(params_perturbed, *args)
            grad[i] = (perturbed_LL - base_LL) / epsilon

        # Compute Hessian using finite differences
        for i in range(len(params)):
            for j in range(len(params)):
                params_perturbed = params.copy()
                params_perturbed[i] += epsilon
                params_perturbed[j] += epsilon
                perturbed_LL = self.log_likelihood(params_perturbed, *args)
                hess[i, j] = (perturbed_LL - base_LL - grad[i] * epsilon - grad[j] * epsilon) / (epsilon ** 2)

        return grad, hess

    def newton_raphson_update(self, params, *args):
        """Update parameters using the Newton-Raphson method."""
        grad, hess = self.compute_gradient_hessian(params, *args)
        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            print("Hessian is not invertible. Applying random perturbation.")
            return self.random_perturbation(params, *args)

        params_new = params - np.dot(hess_inv, grad)
        return self.clip_params(params_new)

    def clip_params(self, params):
        """Clip parameters to ensure they remain within valid bounds."""
        params[0] = np.clip(params[0], 0.5, 0.999999)
        params[1] = np.clip(params[1], -0.999999, 0.999999)
        params[2] = np.clip(params[2], 0, 0.999999)
        return params

    def random_perturbation(self, params, *args):
        """Apply random perturbation to parameters."""
        while True:
            params[0] = np.clip(params[0] + np.random.uniform(-self.perturb_scale, self.perturb_scale), 0.5, 0.999999)
            params[1] = np.clip(params[1] + np.random.uniform(-self.perturb_scale, self.perturb_scale), -0.999999, 0.999999)
            params[2] = np.clip(params[2] + np.random.uniform(-self.perturb_scale, self.perturb_scale), 0, 0.999999)

            if self.is_valid_params(params, *args):
                return params

    def em_algorithm(self, *args):
        """Generalized EM algorithm to optimize parameters."""
        params = np.array([self.a, self.l, self.q])

        for iteration in range(self.max_iter):
            # E-step
            LL = self.log_likelihood(params, *args)

            if LL == -1e10:
                params = self.random_perturbation(params, *args)
                continue

            # Check for convergence
            LL_diff = np.abs(LL - self.best_LL)
            if LL_diff < self.tol:
                self.best_LL = LL
                self.best_params = params.copy()
                continue

            # M-step
            params_new = self.newton_raphson_update(params, *args)
            if self.is_valid_params(params_new, *args):
                params = params_new
                self.best_LL = LL
                self.best_params = params.copy()

        print(f"EM algorithm completed. Best parameters: {self.best_params}, Best log-likelihood: {self.best_LL}")
        return self.best_params, self.best_LL

    def log_likelihood(self, params, *args):
        """Abstract method: Calculate the log-likelihood."""
        raise NotImplementedError

    def is_valid_params(self, params, *args):
        """Abstract method: Check if parameters are valid."""
        raise NotImplementedError
