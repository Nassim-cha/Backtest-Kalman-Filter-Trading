import numpy as np
import matplotlib.pyplot as plt


class DataGeneration:
    """
    A class for generating stochastic processes and correlated returns.

    Attributes:
        A_vect (array): Mean-reverting rates for the X process.
        L_vect (array): Long-term mean for the X process.
        Q_vect (array): Volatility scaling factors for the X process.
        corr_matrix (2D array): Correlation matrix for the X process.
        A_rho_vect (array): Mean-reverting rates for the rho process.
        L_rho_vect (array): Long-term mean for the rho process.
        Q_rho_vect (array): Volatility scaling factors for the rho process.
        n_steps (int): Number of time steps to simulate.
    """
    def __init__(
            self,
            A_vect,
            L_vect,
            Q_vect,
            corr_matrix,
            A_rho_vect,
            L_rho_vect,
            Q_rho_vect,
            n_steps
    ):
        """
        Initialize the DataGeneration class with the given parameters.

        Parameters:
            A_vect (array): Mean-reverting rates for the X process.
            L_vect (array): Long-term mean for the X process.
            Q_vect (array): Volatility scaling factors for the X process.
            corr_matrix (2D array): Correlation matrix for the X process.
            A_rho_vect (array): Mean-reverting rates for the rho process.
            L_rho_vect (array): Long-term mean for the rho process.
            Q_rho_vect (array): Volatility scaling factors for the rho process.
            n_steps (int): Number of time steps to simulate.
        """
        self.A_vect = A_vect
        self.L_vect = L_vect
        self.Q_vect = Q_vect
        self.corr_matrix = corr_matrix
        self.A_rho_vect = A_rho_vect
        self.L_rho_vect = L_rho_vect
        self.Q_rho_vect = Q_rho_vect
        self.n_steps = n_steps

        self.n_x = len(self.A_vect)
        self.n_rho = len(self.A_rho_vect)
        self.X_vect = self.generate_ou_process(self.A_vect, self.L_vect, self.Q_vect, self.corr_matrix, self.n_x)
        self.X_rho_vect = self.generate_ou_process(self.A_rho_vect, self.L_rho_vect, self.Q_rho_vect,
                                                   np.eye(self.n_rho), self.n_rho)

        self.sto_vol, self.sto_correl = self.generate_sto_process()
        self.correlated_eta = self.generate_correlated_eta(self.n_x)
        self.returns = self.generate_returns()

    def generate_ou_process(self, A, L, Q, corr_mat, n_x):
        """
        Generate an Ornstein-Uhlenbeck process.

        Parameters:
            A (array): Mean-reverting rates.
            L (array): Long-term means.
            Q (array): Volatility scaling factors.
            corr_mat (2D array): Correlation matrix.
            n_x (int): Dimensionality of the process.

        Returns:
            process (2D array): Simulated Ornstein-Uhlenbeck process.
        """
        process = np.zeros((self.n_steps, n_x))
        zero_mean = np.zeros(n_x)
        noise = np.random.multivariate_normal(zero_mean, corr_mat, size=self.n_steps)
        process[0] = L + Q * noise[0]
        for t in range(1, self.n_steps):
            process[t] = A * (process[t-1] - L) + L + Q * noise[t]
        return process

    def generate_sto_process(self):
        """
        Generate stochastic volatility and correlation processes.

        Returns:
            sto_vol (2D array): Exponentiated volatility process.
            sto_correl (2D array): Tanh-transformed correlation process.
        """
        sto_vol = np.exp(self.X_vect)
        sto_correl = np.tanh(self.X_rho_vect)
        return sto_vol, sto_correl

    def generate_correlated_eta(self, n_x):
        """
        Generate correlated noise based on stochastic correlation.

        Parameters:
            n_x (int): Dimensionality of the noise process.

        Returns:
            eta (2D array): Correlated noise process.
        """
        eta = np.zeros((self.n_steps, n_x))

        for t in range(self.n_steps):
            corr_matrix_eta = np.eye(n_x)

            # Récupérer les indices du triangle supérieur (au-dessus de la diagonale)
            indices = np.triu_indices(n_x, k=1)

            # Remplir le triangle supérieur avec les valeurs du vecteur
            corr_matrix_eta[indices] = self.sto_correl[t]

            # Copier le triangle supérieur dans le triangle inférieur pour symétrie
            corr_matrix_eta = corr_matrix_eta + corr_matrix_eta.T - np.diag(np.diag(corr_matrix_eta))

            zero_mean = np.zeros(n_x)

            eta[t] = np.random.multivariate_normal(zero_mean, corr_matrix_eta, size=1)

        return eta

    def generate_returns(self):
        """
        Generate returns by combining stochastic volatility and correlated noise.

        Returns:
            returns (2D array): Simulated returns process.
        """
        return self.correlated_eta * self.sto_vol

    def plot_returns_and_volatility(self):
        """
        Plot the simulated returns and stochastic volatility.

        Dynamically adjusts to the number of processes in n_x.
        """
        fig, axs = plt.subplots(self.n_x, 2, figsize=(12, 4 * self.n_x))
        for i in range(self.n_x):
            if self.n_x > 1:
                ax1, ax2 = axs[i]
            else:
                ax1, ax2 = axs

            # Plot returns
            ax1.plot(self.returns[:, i], label=f"Returns Process {i+1}")
            ax1.set_title(f"Returns Process {i+1}")
            ax1.set_xlabel("Time Steps")
            ax1.set_ylabel("Returns")
            ax1.legend()

            # Plot volatility
            ax2.plot(self.sto_vol[:, i], label=f"Volatility Process {i+1}", color='orange')
            ax2.set_title(f"Stochastic Volatility Process {i+1}")
            ax2.set_xlabel("Time Steps")
            ax2.set_ylabel("Volatility")
            ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_X_rho_and_sto_correl(self):
        """
        Plot the X_rho and stochastic correlation processes.

        Dynamically adjusts to the number of rho processes in n_rho.
        """
        fig, axs = plt.subplots(self.n_rho, 2, figsize=(12, 4 * self.n_rho))
        for i in range(self.n_rho):
            if self.n_rho > 1:
                ax1, ax2 = axs[i]
            else:
                ax1, ax2 = axs

            # Plot X_rho
            ax1.plot(self.X_rho_vect[:, i], label=f"X_rho Process {i+1}")
            ax1.set_title(f"X_rho Process {i+1}")
            ax1.set_xlabel("Time Steps")
            ax1.set_ylabel("X_rho")
            ax1.legend()

            # Plot stochastic correlation
            ax2.plot(self.sto_correl[:, i], label=f"Stochastic Correlation {i+1}", color='green')
            ax2.set_title(f"Stochastic Correlation {i+1}")
            ax2.set_xlabel("Time Steps")
            ax2.set_ylabel("Correlation")
            ax2.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    A_vect = np.array([0.975, 0.925, 0.95])
    L_vect = np.array([0.5, 0.25, 0])
    Q_vect = np.array([0.15, 0.125, 0.175])
    corr_matrix = np.array([[1, 0.5, 0.7], [0.5, 1, 0.8], [0.7, 0.8, 1]])
    A_rho_vect = np.array([0.9, 0.95, 0.925])
    L_rho_vect = np.array([0.5, 0.7, 0.3])
    Q_rho_vect = np.array([0.125, 0.1, 0.3])
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

    # Plot returns and volatility
    generate.plot_returns_and_volatility()

    # Plot X_rho and stochastic correlation
    generate.plot_X_rho_and_sto_correl()
