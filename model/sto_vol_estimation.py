import numpy as np
from numba import njit
from utils import generate_sp, calc_weights, f_vectorized
from generate import DataGeneration
import matplotlib.pyplot as plt


class KalmanFilterVolEstimation:
    def __init__(
            self,
            a,
            l,
            q,
            init_log_vol,
            init_var,
            n_steps,
            returns_series,
            alpha=1.6,
            beta=2,
            kappa=1.75
    ):
        self.a = a  # Mean reversion speed
        self.l = l  # Long-term mean
        self.q = q  # Volatility of process
        self.alpha = alpha  # UKF tuning parameter
        self.beta = beta  # UKF tuning parameter
        self.kappa = kappa  # UKF tuning parameter
        self.init_log_vol = init_log_vol  # Initial log correlation
        self.init_var = init_var  # Initial variance
        self.n_steps = n_steps  # Number of time steps
        self.returns_series = returns_series
        self.state_estimation, self.var_setimation, self.LL, self.forecasts = self.calc_log_likelihood_numba()

    def calc_log_likelihood_numba(self):
        state_estimation, var_estimation, LL, forecasts = calculate_loglikelihood(
            self.n_steps,
            self.returns_series,
            self.init_log_vol,
            self.q,
            self.init_var,
            self.a,
            self.l,
            self.alpha,
            self.kappa,
            self.beta
        )
        return state_estimation, var_estimation, LL, forecasts

    def sto_vol_estimation(self):
        """ Compute the estimated correlation using exp of the state estimation. """
        return np.exp(self.state_estimation)

    def calc_eps_t(self):
        return self.returns_series / self.sto_vol_estimation()


@njit
def vectorized_update_step(X2, r, wm2, wc2):
    """
    Perform the vectorized update step in the Kalman filter.

    X2: 1D array of sigma points
    first, second: Scalars or arrays representing the first and second measurements
    wm2: Weights for the measurement update
    """

    # Vectorized calculation of eta assuming X2 is 1D
    eta = r / np.exp(X2)  # Assuming `h` is based on this formula

    # Calculate the normal density for each eta value
    prob_eta = normal_pdf(eta)

    # Vectorized calculation of h_x
    h_x = prob_eta * np.abs(eta)

    # Calculate normalization constant Z
    Z = np.sum(wm2 * h_x)

    # Error handling for Z
    if Z <= 0 or Z < 1e-10:
        return np.nan, np.nan, -1e10  # Error: Z is either too small or non-positive

    # Calculate the mean using the sigma points and h_x
    mean = np.sum((wm2 * X2 * h_x) / Z)

    # Calculate the variance using the updated sigma points
    var = np.sum(wm2 * ((h_x / Z) * (X2 - mean) ** 2))

    return mean, var, Z

#@njit
def calculate_loglikelihood(n_steps, returns_series, init_log_vol, q, init_var, a, l, alpha, kappa, beta):
    """ Main function to estimate state and variance using UKF, in a Numba-compatible manner. """
    L = 2

    # Initialize the augmented covariance and state
    aug_cov, aug_x = initialize_ukf_state(init_var, init_log_vol)

    # Calculate UKF weights
    wm, wc, lambda1 = calc_weights(L, alpha, beta, kappa)
    wm2, wc2, lambda2 = calc_weights(L/2, alpha, beta, kappa)

    state_estimation = np.zeros(n_steps)
    var_estimation = np.zeros(n_steps)

    X_mean=0

    LL = 0

    for t in range(n_steps):
        ### PREDICTION STEP
        X1 = generate_sp(aug_x, aug_cov, lambda1)

        # Apply the vectorized `f` function directly to the arrays `X1[:, 0]` and `X1[:, 1]`
        X = f_vectorized(X1[:, 0], X1[:, 1], a, l, q)

        # Predicting state mean and covariance
        X_mean = np.dot(X.T, wm)  # X.T @ wm works in Numba

        X_diff = X - X_mean
        # Element-wise multiplication for the covariance without using np.diag
        P = np.dot(X_diff.T * wc, X_diff)  # Equivalent to X_diff.T @ np.diag(wc) @ X_diff

        ### UPDATE STEP
        X2 = generate_sp2(P, L, lambda1, X_mean)

        mean, var, Z = vectorized_update_step(X2, returns_series[t], wm2, wc2)

        # If an error occurred (mean, var, or Z is NaN), return error values
        if np.isnan(mean) or np.isnan(var) or np.isnan(Z):
            return None, None, -1e10, None

        state_estimation[t] = mean
        var_estimation[t] = var

        LL += np.log(np.abs(Z))

        # Initialize the augmented covariance and state
        aug_cov, aug_x = initialize_ukf_state(var, mean)

    return state_estimation, var_estimation, LL, X_mean


@njit
def initialize_ukf_state(init_var, init_log_correl):
    """ Initialize the augmented covariance and state vector. """
    aug_cov = np.zeros((2, 2))  # Initialize a 2x2 matrix of zeros
    aug_cov[0, 0] = init_var
    aug_cov[1, 1] = 1.0  # Set the value for the second diagonal entry
    aug_x = np.zeros(2)
    aug_x[0] = init_log_correl
    return aug_cov, aug_x


@njit
def normal_pdf(x):
    """ Calculate the probability density function of a standard normal distribution. """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)

@njit
def generate_sp2(P, L, lambda_, X_mean):
    """
    Generate sigma points for the update step using covariance matrix P and mean X_mean.

    P: Covariance matrix (assumed to be already Cholesky-decomposed or sqrt of covariance matrix)
    L: Dimension of the state vector (here assumed to be 1, scalar case)
    lambda_: Scaling parameter for sigma points
    X_mean: Predicted state mean (float scalar)
    """
    # Compute the scaling factor
    phi = np.sqrt(L + lambda_)

    P_sqrt = np.sqrt(P)

    # Initialize the sigma points array
    X2 = np.zeros(3)  # 3 sigma points for L = 1

    # First sigma point is the mean
    X2[0] = X_mean

    # Generate sigma points based on square root of P and phi
    X2[1] = X_mean + phi * P_sqrt
    X2[2] = X_mean - phi * P_sqrt

    return X2

if __name__ == "__main__":
    A_vect = np.array([0.975, 0.925])
    L_vect = np.array([0.5, 0.25])
    Q_vect = np.array([0.05, 0.125])
    corr_matrix = np.array([[1, 0.5], [0.5, 1]])
    A_rho_vect = np.array([0.975])
    L_rho_vect = np.array([0.5])
    Q_rho_vect = np.array([0.125])
    n_steps = 500
    generate = DataGeneration(
        A_vect,
        L_vect,
        Q_vect,
        corr_matrix,
        A_rho_vect,
        L_rho_vect,
        Q_rho_vect,
        n_steps
    )
    returns = generate.returns[:, 0]
    a = 0.975
    l = 0.5
    q = 0.05
    init_log_vol = l
    init_var = q
    n_steps = len(returns)
    vol_estimate_1 = KalmanFilterVolEstimation(a, l, q, init_log_vol, init_var, n_steps, returns)
    est_vol = vol_estimate_1.state_estimation
    gen_vol = generate.X_vect[:, 0]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(est_vol, label='Estimated Volatility', linestyle='-', marker=None)
    plt.plot(gen_vol, label='Generated Volatility', linestyle='--', marker=None)

    # Add title, labels, legend, and grid
    plt.title("Estimated vs Generated Volatility")
    plt.xlabel("Time Steps")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()


