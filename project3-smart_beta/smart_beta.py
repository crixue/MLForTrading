import pandas as pd
import numpy as np
import helper
import project_helper
import project_tests
import cvxpy as cvx
import matplotlib.pyplot as plt
import matplotlib.dates as mdate


def load_data():
    df = pd.read_csv('eod-quotemedia.csv')

    percent_top_dollar = 0.2
    high_volume_symbols = project_helper.large_dollar_volume_stocks(df, 'adj_close', 'adj_volume', percent_top_dollar)
    df = df[df['ticker'].isin(high_volume_symbols)]

    close = df.reset_index().pivot(index='date', columns='ticker', values='adj_close')
    volume = df.reset_index().pivot(index='date', columns='ticker', values='adj_volume')
    dividends = df.reset_index().pivot(index='date', columns='ticker', values='dividends')
    return close, volume, dividends


def generate_dollar_volume_weights(close, volume):
    """
    Generate dollar volume weights.

    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    volume : str
        Volume for each ticker and date

    Returns
    -------
    dollar_volume_weights : DataFrame
        The dollar volume weights for each ticker and date
    """
    dollar_volume = close * volume
    dollar_volume_weights = dollar_volume.divide(dollar_volume.sum(axis=1), axis=0)
    return dollar_volume_weights


def calculate_dividend_weights(dividends):
    """
    Calculate dividend weights.

    Parameters
    ----------
    ex_dividend : DataFrame
        Ex-dividend for each stock and date

    Returns
    -------
    dividend_weights : DataFrame
        Weights for each stock and date
    """
    cum_dividend_yields = dividends.cumsum()
    dividend_weights = cum_dividend_yields.divide(cum_dividend_yields.sum(axis=1), axis=0)
    return dividend_weights


def generate_returns(prices):
    """
    Generate returns for ticker and date.

    Parameters
    ----------
    prices : DataFrame
        Price for each ticker and date

    Returns
    -------
    returns : Dataframe
        The returns for each ticker and date
    """
    return (prices - prices.shift(1)) / prices.shift(1)


def generate_weighted_returns(returns, weights):
    """
    Generate weighted returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    weights : DataFrame
        Weights for each ticker and date

    Returns
    -------
    weighted_returns : DataFrame
        Weighted returns for each ticker and date
    """
    return returns * weights


def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date

    Returns
    -------
    cumulative_returns : Pandas Series
        Cumulative returns for each date
    """
    return (returns.sum(axis=1) + 1).cumprod()


def tracking_error(benchmark_returns_by_date, etf_returns_by_date):
    """
    Calculate the tracking error.

    Parameters
    ----------
    benchmark_returns_by_date : Pandas Series
        The benchmark returns for each date
    etf_returns_by_date : Pandas Series
        The ETF returns for each date

    Returns
    -------
    tracking_error : float
        The tracking error
    """

    return np.sqrt(252) * np.std(etf_returns_by_date - benchmark_returns_by_date, ddof=1)


def get_covariance_returns(returns):
    """
    Calculate covariance matrices.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date

    Returns
    -------
    returns_covariance  : 2 dimensional Ndarray
        The covariance of the returns
    """
    #TODO: Implement function
    nna_returns = returns.fillna(0)
    return np.cov(nna_returns.T)


def get_optimal_weights(covariance_returns, index_weights, scale=2.0):
    """
    Find the optimal weights.

    Parameters
    ----------
    covariance_returns : 2 dimensional Ndarray
        The covariance of the returns
    index_weights : Pandas Series
        Index weights for all tickers at a period in time
    scale : int
        The penalty factor for weights the deviate from the index
    Returns
    -------
    x : 1 dimensional Ndarray
        The solution for x
    """
    assert len(covariance_returns.shape) == 2
    assert len(index_weights.shape) == 1
    assert covariance_returns.shape[0] == covariance_returns.shape[1]  == index_weights.shape[0]

    x = cvx.Variable(index_weights.shape[0])
    vars_square = cvx.quad_form(x, covariance_returns)
    l2_norms = cvx.norm(x-index_weights, p=2)
    objective = cvx.Minimize(vars_square + scale * l2_norms)
    constraints = [x >= 0, sum(x) == 1]
    problem = cvx.Problem(objective, constraints)

    problem.solve()
    x_values = x.value
    return x_values


def plot_benchmark_returns(benchmark_data, etf_data):
    fig, ax = plt.subplots()
    ax.plot(
        benchmark_data.index,
        benchmark_data,
        label='index'
    )

    ax.plot(
        etf_data.index,
        etf_data,
        label='etf'
    )

    # ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))  # 设置时间标签显示格式
    # ax.set_xticks((np.array(pd.date_range(benchmark_data.index[0], benchmark_data.index[-1], freq='M'))))
    ax.set_xlabel('date')
    ax.set_ylabel('cumulative returns')

    # plt.xticks(np.array(pd.date_range(benchmark_data.index[0], benchmark_data.index[-1], freq='M')), rotation=45)
    plt.legend()
    plt.show()


def test_optimized_portfolio(covariance_returns, index_weights, index_cumulative_returns) :
    raw_optimal_single_rebalance_etf_weights = get_optimal_weights(covariance_returns, index_weights.iloc[-1])
    optimal_single_rebalance_etf_weights = pd.DataFrame(
        np.tile(raw_optimal_single_rebalance_etf_weights, (len(returns.index), 1)),
        returns.index,
        returns.columns)
    optim_etf_returns = generate_weighted_returns(returns, optimal_single_rebalance_etf_weights)
    optim_etf_cumulative_returns = calculate_cumulative_returns(optim_etf_returns)

    optim_etf_tracking_error = tracking_error(np.sum(index_weighted_returns, 1), np.sum(optim_etf_returns, 1))
    print('Optimized ETF Tracking Error: {}'.format(optim_etf_tracking_error))

    plot_benchmark_returns(index_weighted_cumulative_returns, optim_etf_cumulative_returns)


def rebalance_portfolio(returns, index_weights, shift_size, chunk_size):
    """
    Get weights for each rebalancing of the portfolio.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    index_weights : DataFrame
        Index weight for each ticker and date
    shift_size : int
        The number of days between each rebalance
    chunk_size : int
        The number of days to look in the past for rebalancing

    Returns
    -------
    all_rebalance_weights  : list of Ndarrays
        The ETF weights for each point they are rebalanced
    """
    assert returns.index.equals(index_weights.index)
    assert returns.columns.equals(index_weights.columns)
    assert shift_size > 0
    assert chunk_size >= 0

    all_rebalance_weights = []

    for days in range(chunk_size, len(returns), shift_size):
        dynamic_returns = returns[days - chunk_size: days]
        covariance_returns = get_covariance_returns(dynamic_returns)
        optimal_weights = get_optimal_weights(covariance_returns, index_weights.iloc[days-1])
        all_rebalance_weights.append(optimal_weights)

    return all_rebalance_weights


def get_portfolio_turnover(all_rebalance_weights, shift_size, rebalance_count, n_trading_days_in_year=252):
    """
    Calculage portfolio turnover.

    Parameters
    ----------
    all_rebalance_weights : list of Ndarrays
        The ETF weights for each point they are rebalanced
    shift_size : int
        The number of days between each rebalance
    rebalance_count : int
        Number of times the portfolio was rebalanced
    n_trading_days_in_year: int
        Number of trading days in a year

    Returns
    -------
    portfolio_turnover  : float
        The portfolio turnover
    """
    assert shift_size > 0
    assert rebalance_count > 0

    # TODO: Implement function
    sum_total_turnover = np.abs(np.diff(np.flip(all_rebalance_weights, axis=0), axis=0)).sum()
    number_rebalance_events_per_year = n_trading_days_in_year // shift_size
    annualized_turnover = (sum_total_turnover / rebalance_count) * number_rebalance_events_per_year

    return annualized_turnover


if __name__ == '__main__':
    close, volume, dividends = load_data()
    dollar_volume_weights = generate_dollar_volume_weights(close, volume)
    # print(dollar_volume_weights)

    etf_weights = calculate_dividend_weights(dividends)
    # project_tests.test_calculate_dividend_weights(calculate_dividend_weights)
    index_weights = generate_dollar_volume_weights(close, volume)
    # project_tests.test_generate_dollar_volume_weights(generate_dollar_volume_weights)
    returns = generate_returns(close)
    # project_tests.test_generate_returns(generate_returns)

    index_weighted_returns = generate_weighted_returns(returns, index_weights)
    etf_weighted_returns = generate_weighted_returns(returns, etf_weights)

    index_weighted_cumulative_returns = calculate_cumulative_returns(index_weighted_returns)
    etf_weighted_cumulative_returns = calculate_cumulative_returns(etf_weighted_returns)

    # print(index_weighted_cumulative_returns)
    # print(etf_weighted_cumulative_returns)

    covariance_returns = get_covariance_returns(returns)
    smart_beta_tracking_error = tracking_error(np.sum(index_weighted_returns, 1), np.sum(etf_weighted_returns, 1))
    # project_tests.test_tracking_error(tracking_error)
    # print(smart_beta_tracking_error)

    # project_tests.test_get_covariance_returns(get_covariance_returns)

    # project_tests.test_get_optimal_weights(get_optimal_weights)
    # test_optimized_portfolio(covariance_returns, index_weights, index_weighted_cumulative_returns)

    # project_tests.test_rebalance_portfolio(rebalance_portfolio)

    chunk_size = 250
    shift_size = 5
    all_rebalance_weights = rebalance_portfolio(returns, index_weights, shift_size, chunk_size)
    print(all_rebalance_weights)