from generate import DataGeneration
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from utils import generate_sp, calc_weights, f_vectorized


class KalmanFilterCorrelAndVolEstimation:
    def __init__(
            self,
            returns,
            A_vect,
            L_vect,
            Q_vect,
            corr_matrix,
            A_rho_vect,
            L_rho_vect,
            Q_rho_vect,
            alpha=1.6,
            beta=2,
            kappa=1.75
    ):
        self.returns = returns
        self.A_vect = A_vect
        self.L_vect = L_vect
        self.Q_vect = Q_vect
        self.corr_matrix = corr_matrix
        self.A_rho_vect = A_rho_vect
        self.L_rho_vect = L_rho_vect
        self.Q_rho_vect = Q_rho_vect
        self.n_steps = returns.shape[0]
        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta
        self.n_x = len(self.A_vect)
        self.n_rho = len(self.A_rho_vect)

    def clac_estimation(self):

        """
        Call the standalone calc_estimation function and return the result.
        """
        result = calc_estimation(
            self.n_steps,
            self.n_x,
            self.n_rho,
            self.alpha,
            self.beta,
            self.kappa,
            self.A_vect,
            self.L_vect,
            self.Q_vect,
            self.A_rho_vect,
            self.L_rho_vect,
            self.Q_rho_vect,
            self.corr_matrix,
            self.returns
        )
        return result


@njit
def calc_estimation(
        n_steps,
        n_x,
        n_rho,
        alpha,
        beta,
        kappa,
        A_vect,
        L_vect,
        Q_vect,
        A_rho_vect,
        L_rho_vect,
        Q_rho_vect,
        corr_matrix,
        returns
):
    """
    Perform estimation using the Unscented Kalman Filter (UKF).

    Parameters:
    ----------
    n_steps : int
        Number of steps to iterate.
    n_x : int
        Number of state variables.
    n_rho : int
        Number of correlation variables.
    alpha, beta, kappa : float
        UKF parameters.
    A_vect, L_vect, Q_vect : ndarray
        Parameters for volatility components.
    A_rho_vect, L_rho_vect, Q_rho_vect : ndarray
        Parameters for correlation components.
    corr_matrix : ndarray
        Initial correlation matrix.
    returns : ndarray
        Observed returns.

    Returns:
    -------
    result : ndarray
        Estimated mean values for each time step, shape (n_steps, n_x + n_rho).
    """
    # Step 1: Initialize augmented state and covariance
    n_var = 2 * (n_x + n_rho)
    aug_cov = np.zeros((n_var, n_var))
    aug_cov[0:n_x, 0:n_x] = np.diag(Q_vect)
    aug_cov[n_x:n_x + n_rho, n_x:n_x + n_rho] = np.diag(Q_rho_vect)
    aug_cov[n_x + n_rho:2 * n_x + n_rho, n_x + n_rho:2 * n_x + n_rho] = corr_matrix
    aug_cov[2 * n_x + n_rho:, 2 * n_x + n_rho:] = np.eye(n_rho)

    aug_x = np.zeros(n_var)
    aug_x[0:n_x] = L_vect
    aug_x[n_x:n_x + n_rho] = L_rho_vect

    # Step 2: Calculate UKF weights
    wm, wc, lambda1 = calc_weights(n_var, alpha, beta, kappa)
    n_points = int(n_var / 2)
    wm2, _, lambda2 = calc_weights(n_points, alpha, beta, kappa)

    # Step 3: Initialize result array
    result = np.zeros((n_steps, n_points))

    # Step 4: Iterative filtering
    for t in range(n_steps):
        # Prediction Step
        X1 = generate_sp(aug_x, aug_cov, lambda1)
        A1 = np.concatenate((A_vect, A_rho_vect))
        L1 = np.concatenate((L_vect, L_rho_vect))
        Q1 = np.concatenate((Q_vect, Q_rho_vect))

        # Apply vectorized dynamics function
        X = f_vectorized(X1[:, 0:n_points], X1[:, n_points:], A1, L1, Q1)

        # Predict state mean and covariance
        X_mean = np.dot(X.T, wm)
        X_diff = X - X_mean
        P = np.dot(X_diff.T * wc, X_diff)

        # Update Step
        X2 = generate_sp(X_mean, P, lambda2)
        mean, var, Z = vectorized_update_step(X2, returns[t], wm2, n_x)

        # Update augmented state and covariance
        aug_x[0:n_points] = mean
        aug_cov[0:n_points, 0:n_points] = var

        # Store result
        result[t] = mean

    return result

@njit
def vectorized_update_step(X2, returns, wm2, n_x):
    """
    Perform a vectorized update step for a Kalman filter-like process.

    Parameters:
    ----------
    X2 : ndarray
        Sigma points, shape (n_samples, n_features).
    returns : ndarray
        Returns or observed data, shape (n_samples, n_x).
    wm2 : ndarray
        Weights for the sigma points, shape (n_samples,).
    n_x : int
        Number of state variables (volatility components).

    Returns:
    -------
    mean : ndarray
        Updated mean vector, shape (n_features,).
    P : ndarray
        Updated covariance matrix, shape (n_features, n_features).
    Z : float
        Normalizing constant / Likelihood
    """
    # Step 1: Calculate eta_vect (adjusted returns)
    # Each component of the return is divided by the corresponding exponential of X2
    eta_vect = returns / np.exp(X2[:, 0:n_x])

    # Step 2: Initialize arrays for intermediate results
    prob_eta_vect = np.zeros(X2.shape[0])  # To store probabilities of eta for each sample
    h_x = np.zeros(X2.shape[0])  # To store the final h_x for each sample

    # Step 3: Loop through all sigma points (rows of X2)
    for idx in range(X2.shape[0]):
        # Initialize an identity matrix for the correlation matrix
        corr_matrix = np.eye(n_x)

        # Populate the upper triangular part of the correlation matrix
        k = 0
        for i in range(n_x):
            for j in range(i + 1, n_x):
                corr_matrix[i, j] = np.tanh(X2[idx, n_x + k])  # Transform correlation values to (-1, 1)
                k += 1

        # Mirror the upper triangular part to make the matrix symmetric
        for i in range(n_x):
            for j in range(i + 1, n_x):
                corr_matrix[j, i] = corr_matrix[i, j]

        # Step 4: Ensure the correlation matrix is positive definite
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        if np.any(eigenvalues < 0):  # Check for negative eigenvalues
            corr_matrix = nearest_positive_semidefinite_numba(corr_matrix)

        # Step 5: Compute the probability of eta under the multivariate normal distribution
        eta = eta_vect[idx, :]  # Current eta vector
        prob_eta = multivariate_normal_pdf_numba(eta, np.zeros(n_x), corr_matrix)
        prob_eta_vect[idx] = prob_eta  # Store the probability

        # Step 6: Compute the product of absolute eta values manually (Numba-compatible)
        product_abs_eta = 1.0
        for j in range(n_x):
            product_abs_eta *= abs(eta_vect[idx, j])

        # Store the final h_x for the current sample
        h_x[idx] = product_abs_eta * prob_eta_vect[idx]

    # Step 7: Compute the normalizing constant Z
    Z = np.sum(wm2 * h_x)

    # Step 8: Compute the updated mean vector
    mean = np.sum(wm2 * h_x / Z * X2.T, axis=1)

    # Step 9: Compute the variance (std_mat) for each state variable
    X_diff = X2 - mean  # Difference from the mean for each sigma point
    std_mat = np.sqrt(np.sum(wm2 * (X_diff.T ** 2) * h_x / Z, axis=1))

    # Step 10: Compute the updated covariance matrix
    P = np.outer(std_mat, std_mat)  # Outer product of standard deviations

    return mean, P, Z


@njit
def nearest_positive_semidefinite_numba(matrix):
    """
    Ensure a matrix is positive semidefinite by adjusting eigenvalues.
    """
    # Symmetrize the matrix
    matrix = (matrix + matrix.T) / 2

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Clip eigenvalues to be non-negative
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    # Reconstruct the matrix
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


@njit
def multivariate_normal_pdf_numba(x, mean, cov):
    """
    Compute the multivariate normal PDF manually.
    """
    n = len(mean)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)

    # Normalization constant
    norm_const = 1.0 / (np.sqrt((2 * np.pi) ** n * det_cov))

    # Exponent term
    diff = x - mean
    exponent = -0.5 * np.dot(np.dot(diff, inv_cov), diff)

    return norm_const * np.exp(exponent)


def calculate_dynamic_beta(vol_asset, vol_market, correl):
    """
    Calculate beta using volatility and correlation.

    Parameters:
    ----------
    vol_asset : ndarray
        Volatility (standard deviation) of the asset's returns.
    vol_market : ndarray
        Volatility (standard deviation) of the market's returns.
    correl : ndarray
        Correlation between the asset and the market.

    Returns:
    -------
    beta : ndarray
        The beta of the asset relative to the market.
    """
    dynamic_beta = correl * (vol_asset / vol_market)
    return dynamic_beta


def plot_function(x, y, title="Plot", xlabel="X-axis", ylabel="Y-axis", labels=None,
                  linestyle='-', markers=None, colors=None, legend=True, grid=True):
    """
    A reusable function to plot data with customization options.

    Parameters:
    ----------
    x : list or ndarray
        X-axis data. Can be a single list or a list of lists for multiple series.
    y : list or ndarray
        Y-axis data. Can be a single list or a list of lists for multiple series.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the X-axis.
    ylabel : str, optional
        Label for the Y-axis.
    labels : list of str, optional
        Labels for each data series.
    linestyle : str or list of str, optional
        Line style for the plot(s). Can be a single style or a list of styles.
    markers : list of str, optional
        Marker styles for the plot(s). Can be a single marker or a list of markers.
    colors : list of str, optional
        Colors for the plot(s). Can be a single color or a list of colors.
    legend : bool, optional
        Whether to display a legend (default is True).
    grid : bool, optional
        Whether to display gridlines (default is True).
    """
    # Check if multiple series are provided
    if isinstance(y[0], (list, tuple, np.ndarray)):
        num_series = len(y)
    else:
        y = [y]
        x = [x]
        num_series = 1

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot each series
    for i in range(num_series):
        plt.plot(
            y[i], x[i],
            label=labels[i] if labels else None,
            linestyle=linestyle[i] if isinstance(linestyle, list) else linestyle,
            marker=markers[i] if markers and isinstance(markers, list) else markers,
            color=colors[i] if colors and isinstance(colors, list) else colors
        )

    # Add labels, title, legend, and grid
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend and labels:
        plt.legend()
    if grid:
        plt.grid()

    # Show the plot
    plt.show()


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
    estimate = KalmanFilterCorrelAndVolEstimation(generate.returns,
                                                  A_vect,
                                                  L_vect,
                                                  Q_vect,
                                                  corr_matrix,
                                                  A_rho_vect,
                                                  L_rho_vect,
                                                  Q_rho_vect)

    result = estimate.clac_estimation()

    estimated_vol = np.exp(result[:, 0:2])
    estimated_correl = np.tanh(result[:, 2:])



    # Plot volatility components
    fig_vol, axes_vol = plt.subplots(generate.sto_vol.shape[1], 1, figsize=(12, 4 * generate.sto_vol.shape[1]))

    for i, ax in enumerate(axes_vol):
        ax.plot(generate.sto_vol[:, i], label=f'Generated Volatility Component {i + 1}', linestyle='dashed')
        ax.plot(estimated_vol[:, i], label=f'Estimated Volatility Component {i + 1}')
        ax.set_title(f"Volatility Component {i + 1}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Volatility")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()

    # Plot correlation components
    fig_corr, ax = plt.subplots(1, 1, figsize=(12, 4))

    # Plot the generated and estimated correlation
    ax.plot(generate.sto_correl, label='Generated Correlation Component', linestyle='dashed')
    ax.plot(estimated_correl, label='Estimated Correlation Component')

    # Add labels, title, legend, and grid
    ax.set_title("Correlation Component")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Correlation")
    ax.legend()
    ax.grid()

    # Show the plot
    plt.tight_layout()
    plt.show()

vol_asset = estimated_vol[:, 1]
vol_market = estimated_vol[:, 0]
correl = estimated_correl.T

returns_asset = generate.returns[:, 1]
returns_market = generate.returns[:, 0]

dynamic_beta = calculate_dynamic_beta(vol_asset, vol_market, correl)

x = np.array([returns_asset, returns_market * dynamic_beta.flatten()]).T

plot_function(x, returns_market)

index_array = np.arange(len(returns_market))

plot_function(x, index_array)

eps = returns_asset - returns_market * dynamic_beta.flatten()

plot_function(eps, index_array)

mean_eps = np.mean(eps)
std_eps = np.std(eps)
print(f"mean = {mean_eps}")
print(f"std = {std_eps}")
