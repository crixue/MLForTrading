import pandas as pd
import numpy as np
import helper
import matplotlib.pyplot as plt
# import project_helper
import project_tests
from scipy import stats


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def get_data():
    df = pd.read_csv('eod-quotemedia.csv', parse_dates=['date'], index_col=False)
    close = df.reset_index().pivot(index='date', columns='ticker', values='adj_close')
    print('Loaded Data')
    return close


def execute():
    project_helper.print_dataframe(close)

    apple_ticker = 'AAPL'
    project_helper.plot_stock(close[apple_ticker], '{} Stock'.format(apple_ticker))


def resample_prices(close_prices, freq='M'):
    """
    Resample close prices for each ticker at specified frequency.

    Parameters
    ----------
    close_prices : DataFrame
        Close prices for each ticker and date
    freq : str
        What frequency to sample at
        For valid freq choices, see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    Returns
    -------
    prices_resampled : DataFrame
        Resampled prices for each ticker and date
    """
    # TODO: Implement Function

    return close_prices.groupby(pd.Grouper(freq=freq)).last()


def compute_log_returns(prices):
    """
    Compute log returns for each ticker.

    Parameters
    ----------
    prices : DataFrame
        Prices for each ticker and date

    Returns
    -------
    log_returns : DataFrame
        Log returns for each ticker and date
    """
    # TODO: Implement Function
    return np.log(prices) - np.log(prices.shift(1))


def shift_returns(returns, shift_n):
    """
    Generate shifted returns

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    shift_n : int
        Number of periods to move, can be positive or negative

    Returns
    -------
    shifted_returns : DataFrame
        Shifted returns for each ticker and date
    """
    # TODO: Implement Function
    return returns.shift(shift_n)


def get_top_n(prev_returns, top_n):
    """
    Select the top performing stocks

    Parameters
    ----------
    prev_returns : DataFrame
        Previous shifted returns for each ticker and date
    top_n : int
        The number of top performing stocks to get

    Returns
    -------
    top_stocks : DataFrame
        Top stocks for each ticker and date marked with a 1
    """
    top_n_df = prev_returns.apply(lambda x: x.nlargest(top_n), axis=1)
    top_n_df = top_n_df.applymap(lambda x: 0 if pd.isna(x) else 1)
    top_n_df = top_n_df.astype('int64')

    return top_n_df


def print_top(df, name, top_n=10):
    print('{} Most {}:'.format(top_n, name))
    sorted_df = df.sum().sort_values(ascending=False)
    index_n_sorted_np = sorted_df.index[:top_n].values
    print(', '.join(index_n_sorted_np.tolist()))


def portfolio_returns(df_long, df_short, lookahead_returns, n_stocks):
    """
    Compute expected returns for the portfolio, assuming equal investment in each long/short stock.

    Parameters
    ----------
    df_long : DataFrame
        Top stocks for each ticker and date marked with a 1
    df_short : DataFrame
        Bottom stocks for each ticker and date marked with a 1
    lookahead_returns : DataFrame
        Lookahead returns for each ticker and date
        前瞻性(look ahead) bias是指 根据现有信息学习或者模拟未来不确定的信息时造成的偏差
    n_stocks: int
        The number number of stocks chosen for each month

    Returns
    -------
    portfolio_returns : DataFrame
        Expected portfolio returns for each ticker and date
    """
    return (df_long - df_short) * lookahead_returns / n_stocks


def get_returns_mean(returns) :
    returns = returns.sum().dropna()
    return returns.mean()


def get_returns_standard_error(returns):
    returns = returns.sum().dropna()
    return returns.sem()


def get_annual_returns(returns):
    mean_returns = get_returns_mean(returns)
    return (np.exp(mean_returns*12) - 1) * 100


def analyze_alpha(expected_portfolio_returns_by_date):
    """
    Perform a t-test with the null hypothesis being that the expected mean return is zero.

    Parameters
    ----------
    expected_portfolio_returns_by_date : Pandas Series
        Expected portfolio returns for each date

    Returns
    -------
    t_value
        T-statistic from t-test
    p_value
        Corresponding p-value
    """
    null_hypothesis = 0.0

    t, p = stats.ttest_1samp(expected_portfolio_returns_by_date.values, null_hypothesis)

    return t, p/2


if __name__ == '__main__':
    # df = resample_prices(get_data())
    # plot_data(df)

    # monthly_close = resample_prices(close)
    # project_helper.plot_resampled_prices(
    #     monthly_close.loc[:, apple_ticker],
    #     close.loc[:, apple_ticker],
    #     '{} Stock - Close Vs Monthly Close'.format(apple_ticker))

    # print(compute_log_returns(resample_prices(get_data())))
    # print(resample_prices(get_data()), 1)
    # print(shift_returns(resample_prices(get_data()), -1))

    resample_df = resample_prices(get_data())
    log_return_df = compute_log_returns(resample_df)
    log_return_shift_df = shift_returns(log_return_df, -1)
    top_n = 10
    top_n_df = get_top_n(log_return_shift_df, top_n)
    last_n_df = get_top_n(-log_return_shift_df, top_n)
    # print(top_n_df)
    # print_top(top_n_df, 10)
    # print_top(last_n_df, 10)

    # plot_returns(log_return_df, "log return")
    # project_tests.test_portfolio_returns(portfolio_returns)
    lookahead_returns = shift_returns(compute_log_returns(resample_prices(get_data())), -1)
    expected_portfolio_returns = portfolio_returns(top_n_df, last_n_df, lookahead_returns, 2 * top_n)

    portfolio_ret_mean = get_returns_mean(expected_portfolio_returns)
    portfolio_ret_ste = get_returns_standard_error(expected_portfolio_returns)
    portfolio_ret_annual_rate = get_annual_returns(expected_portfolio_returns)

    print("""
    Mean:                       {:.6f}
    Standard Error:             {:.6f}
    Annualized Rate of Return:  {:.2f}%
    """.format(portfolio_ret_mean, portfolio_ret_ste, portfolio_ret_annual_rate))

    t_value, p_value = analyze_alpha(log_return_shift_df.mean(axis=1).dropna())
    print("""
    Alpha analysis:
     t-value:        {:.3f}
     p-value:        {:.6f}
    """.format(t_value, p_value))
    '''
    I observed a p-value of 0.082517. 
    Since this is greater than our $\alpha$ of 0.05, 
    we must conclude that there is too great a chance of observing an annualized rate of return of 3.76% under our null hypothesis $H_0$ (that the actual mean return from the signal is zero), 
    and are thus unable to reject this null hypothesis. 

    In other words, because the p-value is larger than our $\alpha$ value,
    we can't confidently state that the 3.76% annualized return we observed *was not* due to random chance.
    '''
