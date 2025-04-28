from model.sto_vol_sto_correl_opti import StoVolStoCorrelOpti
from model.estimate import KalmanFilterCorrelAndVolEstimation
from data_loader_excel import IndexReturnsRetriever
import numpy as np
import pandas as pd
from backtest import run_backtest

np.random.seed(1)
# List of ticker symbols (e.g., S&P 500, NASDAQ, Dow Jones, etc.)
df = pd.read_excel("sx5e_comp.xlsx", index_col=0)
# MSCI EMU, MSCI Europe ex EMU, asia ex japan, Japan, US, Latam
N = 250

# Initialize the IndexReturnsRetriever (automatically downloads returns data)
retriever = IndexReturnsRetriever(prices=df, N=N)

# Retrieve in-sample and out-of-sample data
(
    in_sample_data,
    test_data_dict,
    mean_returns_dict,
    end_date,
    out_sample_data,
    remaining_out_sample,
    tickers,
    exchange,
    start_date,
    end_date,
    n_tickers
) = retriever.get_insample_data()

# Run the optimizer
opti = StoVolStoCorrelOpti(in_sample_data, 5)
LL, best_params = opti.optimize()

# build prediction of correl and prediction of vol matrices
pred_vol = np.zeros((remaining_out_sample, n_tickers+1))
forcasted_correl = np.ones((remaining_out_sample, n_tickers+1, n_tickers+1))

# Calculate the actual and forecast stochastic covariance matrix based on the optimized parameters
# Calculate forecasts for out_sample periods
for key, test_data_returns in test_data_dict.items():

    idx1 = int(key[0])
    idx2 = int(key[1])

    opt_A_vect, opt_L_vect, opt_Q_vect, opt_corr_matrix, opt_A_rho_vect, opt_L_rho_vect, opt_Q_rho_vect = best_params[key]

    estimate = KalmanFilterCorrelAndVolEstimation(test_data_returns,
                                                  opt_A_vect,
                                                  opt_L_vect,
                                                  opt_Q_vect,
                                                  opt_corr_matrix,
                                                  opt_A_rho_vect,
                                                  opt_L_rho_vect,
                                                  opt_Q_rho_vect,
                                                  mean_returns_dict[key])

    result, LL, forecast, mu_vect, prediction = estimate.clac_estimation()

    pred_vol[:, idx1] = pred_vol[:, idx1] + np.exp(forecast[N:, 0])
    pred_vol[:, idx2] = pred_vol[:, idx2] + np.exp(forecast[N:, 1])
    forcasted_correl[:, idx1, idx2] = np.tanh(forecast[N:, 2])
    forcasted_correl[:, idx2, idx1] = np.tanh(forecast[N:, 2])

pred_vol /= n_tickers

diag_vol_stocks = np.zeros((remaining_out_sample, n_tickers, n_tickers))
diag_vol_stocks[np.arange(remaining_out_sample)[:, None], np.arange(n_tickers), np.arange(n_tickers)] = pred_vol[:, 1:]
corr_stocks = forcasted_correl[:, 1:, 1:]
# Compute covariance matrix
covariance = diag_vol_stocks @ corr_stocks @ diag_vol_stocks
# compute dynamic betas
beta = forcasted_correl[:, 1:, 0] * pred_vol[:, 1:] / pred_vol[:, [0]]


# Save
np.savez(
    "cached_forecast_data3.npz",
    beta=beta,
    covariance=covariance,
    out_sample_data=out_sample_data,
    exchange=exchange,
    start_date=start_date,
    N=N,
    remaining_out_sample=remaining_out_sample,
    n_tickers=n_tickers
)

run_backtest(beta, out_sample_data, remaining_out_sample, n_tickers+1)