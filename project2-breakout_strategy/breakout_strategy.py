import pandas as pd
import numpy as np
import helper
import matplotlib.pyplot as plt
import project_helper
import project_tests
from scipy.stats import kstest


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def plot_prices(prices, title="Stock prices", xlabel="Date", ylabel="Price"):
    fig, ax = plt.subplots()
    ax.plot(
        prices.index,
        prices
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.legend()
    plt.show()


def load_data():
    df_original = pd.read_csv('eod-quotemedia.csv', parse_dates=['date'], index_col=False)

    # Add TB sector to the market
    df = df_original
    df = pd.concat([df] + project_helper.generate_tb_sector(df[df['ticker'] == 'AAPL']['date']), ignore_index=True)

    close = df.reset_index().pivot(index='date', columns='ticker', values='adj_close')
    high = df.reset_index().pivot(index='date', columns='ticker', values='adj_high')
    low = df.reset_index().pivot(index='date', columns='ticker', values='adj_low')

    print('Loaded Data')
    return close, high, low


def get_high_lows_lookback(high, low, lookback_days):
    """
    Get the highs and lows in a lookback window.

    Parameters
    ----------
    high : DataFrame
        High price for each ticker and date
    low : DataFrame
        Low price for each ticker and date
    lookback_days : int
        The number of days to look back

    Returns
    -------
    lookback_high : DataFrame
        Lookback high price for each ticker and date
    lookback_low : DataFrame
        Lookback low price for each ticker and date
    """
    high = high.shift(1).rolling(lookback_days).max()
    low = low.shift(1).rolling(lookback_days).min()
    return high, low


def plot_high_low_prices(prices, lookback_high, lookback_low, title):
    fig, ax = plt.subplots()
    ax.plot(
        lookback_high.index,
        lookback_high,
        label='lookback_high'
    )

    ax.plot(
        lookback_low.index,
        lookback_low,
        label='lookback_low')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    ax.plot(
        prices.index,
        prices,
        label='close price'
    )

    plt.legend()
    plt.show()


def plot_long_short_signal(prices, long_short_signals, title="title"):
    fig, ax = plt.subplots()
    ax.plot(
        prices.index,
        prices,
        label="close price"
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Prices')
    ax.set_title(title)

    plot_long_short_annotation(ax, long_short_signals, prices)

    plt.legend()
    plt.show()


def plot_long_short_annotation(ax, long_short_signals, prices):
    long_signal = pd.DataFrame(data=[{'x': index, 'y': price}
                                     for index, price in prices[long_short_signals == 1].iteritems()])
    handle_long = ax.scatter(
        long_signal['x'],
        long_signal['y'],
        c="g"
    )
    short_signal = pd.DataFrame(data=[{'x': index, 'y': price}
                                      for index, price in prices[long_short_signals == -1].iteritems()])
    handle_short = ax.scatter(
        short_signal['x'],
        short_signal['y'],
        c="r"
    )
    plt.legend(handles=[handle_long, handle_short], labels=['long signal', 'short signal'])


def plot_signal_returns(prices, signal_return_tuple, title, xlabel="Date", ylabel="Price", ylabel2="Returns"):
    """
    plot one image with returns and signals
    :param prices:
    :param signal_return_list:
    :param titles:
    :return:
    """

    fig, ax = plt.subplots()
    ax.plot(
        prices.index,
        prices
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    signal_return = signal_return_tuple[0]
    long_short_signals = signal_return_tuple[1]
    lookahead_days = signal_return_tuple[2]

    plot_long_short_annotation(ax, long_short_signals, prices)

    ax2 = ax.twinx()
    non_zero_signals = signal_return[signal_return != 0]
    ax2.plot(
        non_zero_signals.index,
        non_zero_signals,
        'c',
        label="{} Day Lookahead Returns".format(lookahead_days)
    )
    ax2.set_ylabel(ylabel2)
    ax2.set_title(title)

    plt.legend()
    plt.show()


def plot_returns_signal_histograms(return_signal_list, title):
    """
    绘制回报率信号的直方图
    需要首先合并所有的回报率数据，然后取出所有回报率非0的数据
    :param return_signal_list:
    :param title:
    :return:
    """
    fig, ax = plt.subplots()

    filtered_series = pd.Series(data=return_signal_list.stack())
    series = filtered_series[filtered_series != 0].dropna()
    ax.hist(series, 'auto', normed=True)
    ax.set_xticks([-0.5 , 0, 0.5])
    ax.set_xlabel('log returns')
    ax.set_ylabel('days * 100')
    ax.set_title(title)

    fig.tight_layout()
    plt.show()


def plot_returns_signal_to_normal_histograms(return_signal_list, title):
    fig, ax = plt.subplots()

    filtered_series = pd.Series(data=return_signal_list.stack())
    series = filtered_series[filtered_series != 0].dropna()
    normal_series = np.random.normal(np.mean(series), np.std(series), len(series))

    ax.hist(series, 'auto', normed=True, color='r', alpha=0.7)
    # ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xlabel('log returns')
    ax.set_ylabel('days * 100')
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.hist(normal_series, 'auto', normed=True, color='#FFB5C5', alpha=0.5)

    fig.tight_layout()
    plt.show()


def get_long_signal(close, lookback_high):
    return (close > lookback_high).astype(np.int64)


def get_short_signal(close, lookback_low):
    return (close < lookback_low).astype(np.int64) * -1


def get_long_short(close, lookback_high, lookback_low):
    """
    Generate the signals long, short, and do nothing.

    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookback_high : DataFrame
        Lookback high price for each ticker and date
    lookback_low : DataFrame
        Lookback low price for each ticker and date

    Returns
    -------
    long_short : DataFrame
        The long, short, and do nothing signals for each ticker and date
    """

    long_signals = (close > lookback_high).astype(np.int64)
    short_signals = (close < lookback_low).astype(np.int64) * -1
    long_short = long_signals + short_signals

    return long_short


def clear_signals(signals, window_size):
    """
    Clear out signals in a Series of just long or short signals.

    Remove the number of signals down to 1 within the window size time period.

    Parameters
    ----------
    signals : Pandas Series
        The long, short, or do nothing signals
    window_size : int
        The number of days to have a single signal

    Returns
    -------
    signals : Pandas Series
        Signals with the signals removed from the window size
    """
    # Start with buffer of window size
    # This handles the edge case of calculating past_signal in the beginning
    clean_signals = [0] * window_size

    for signal_i, current_signal in enumerate(signals):
        # Check if there was a signal in the past window_size of days
        has_past_signal = bool(sum(clean_signals[signal_i:signal_i + window_size]))
        # Use the current signal if there's no past signal, else 0/False
        clean_signals.append(not has_past_signal and current_signal)

    # Remove buffer
    clean_signals = clean_signals[window_size:]

    # Return the signals as a Series of Ints
    return pd.Series(np.array(clean_signals).astype(np.int64), signals.index)


def filter_signals(signal, lookahead_days):
    """
    Filter out signals in a DataFrame.

    Parameters
    ----------
    signal : DataFrame
        The long, short, and do nothing signals for each ticker and date
    lookahead_days : int
        The number of days to look ahead

    Returns
    -------
    filtered_signal : DataFrame
        The filtered long, short, and do nothing signals for each ticker and date
    """
    long_signal = np.ceil((signal + 1) / 2).astype(np.int64) - 1
    short_signal = np.floor((signal - 1) / 2).astype(np.int64) + 1

    signal_copy = signal.copy()
    for title, single_signal in signal.iteritems():
        single_long_signal = long_signal.loc[:, title]
        single_short_signal = short_signal.loc[:, title]
        clear_long_signal = clear_signals(single_long_signal, lookahead_days)
        clear_short_signal = clear_signals(single_short_signal, lookahead_days)
        signal_copy.loc[:, title] = clear_long_signal + clear_short_signal
    return signal_copy


def get_lookahead_prices(close, lookahead_days):
    """
    Get the lookahead prices for `lookahead_days` number of days.

    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_days : int
        The number of days to look ahead

    Returns
    -------
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date
    """

    return close.shift(-1 * lookahead_days)


def get_log_return_lookahead(close, lookahead_prices):
    """
    Calculate the log returns from the lookahead days to the signal day.

    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date

    Returns
    -------
    lookahead_returns : DataFrame
        The lookahead log returns for each ticker and date
    """
    # TODO: Implement function
    return np.log(lookahead_prices) - np.log(close)


def get_signal_return(signal, lookahead_returns):
    """
    Compute the signal returns.
    相当于买卖信号乘以提前n天的回报率
    Parameters
    ----------
    signal : DataFrame
        The long, short, and do nothing signals for each ticker and date
    lookahead_returns : DataFrame
        The lookahead log returns for each ticker and date

    Returns
    -------
    signal_return : DataFrame
        Signal returns for each ticker and date
    """
    # TODO: Implement function

    return signal * lookahead_returns


def calculate_kstest(long_short_signal_returns):
    """
    Calculate the KS-Test against the signal returns with a long or short signal.

    Parameters
    ----------
    long_short_signal_returns : DataFrame
        The signal returns which have a signal.
        This DataFrame contains two columns, "ticker" and "signal_return"

    Returns
    -------
    ks_values : Pandas Series
        KS static for all the tickers
    p_values : Pandas Series
        P value for all the tickers
    """
    normal_args = (np.mean(long_short_signal_returns.loc[:, 'signal_return']),
                   np.std(long_short_signal_returns.loc[:, 'signal_return'], ddof=0))
    ktest_series = long_short_signal_returns.groupby('ticker').agg(lambda x: kstest(rvs=x, cdf='norm', args=normal_args))

    d_series = ktest_series.applymap(lambda x: x[0])
    d_series.columns = ['D']
    d_series = d_series.squeeze()

    p_values_series = ktest_series.applymap(lambda x: x[1])
    p_values_series.columns = ['p-values']
    p_values_series = p_values_series.squeeze()

    return d_series, p_values_series


def find_outliers(ks_values, p_values, ks_threshold, pvalue_threshold=0.05):
    """
    Find outlying symbols using KS values and P-values

    Parameters
    ----------
    ks_values : Pandas Series
        KS statistic for all the tickers
    p_values : Pandas Series
        P value for all the tickers
    ks_threshold : float
        The threshold for the KS statistic
    pvalue_threshold : float
        The threshold for the p-value

    Returns
    -------
    outliers : set of str
        Symbols that are outliers
    """
    ks_outliers = ks_values > ks_threshold
    p_outliers = p_values < pvalue_threshold
    concat_outliers = (ks_outliers & p_outliers).where(lambda x: x==True).dropna()
    # ks_outliers_list = ks_outliers.dropna().index.values
    # p_outliers_list = p_outliers.dropna().index.values
    # return set(np.concatenate((ks_outliers_list, p_outliers_list)))
    return set(concat_outliers.index.values)


def get_tickers_without_outliers(close, outlier_tickers):
    """
    获取排查有问题的其他股票的名称
    :param close:
    :param outlier_tickers:
    :return:
    """
    return list(set(close.columns) - outlier_tickers)


if __name__ == '__main__':
    close, high, low = load_data()

    apple_ticker = 'AAPL'
    # df_apple = close[apple_ticker]
    # plot_data(df_apple)

    # project_tests.test_get_high_lows_lookback(get_high_lows_lookback)

    lookback_days = 50
    lookback_high, lookback_low = get_high_lows_lookback(high, low, lookback_days)
    # plot_high_low_prices(close[apple_ticker], lookback_high[apple_ticker], lookback_low[apple_ticker], "title")

    # plot_long_short_signal(close[apple_ticker], "sell signals")
    signal = get_long_short(close, lookback_high, lookback_low)
    filter_signal_5 = filter_signals(signal, 5)
    filter_signal_10 = filter_signals(signal, 10)
    filter_signal_20 = filter_signals(signal, 20)

    # for filter_signal in [filter_signals(signal, 5), filter_signals(signal, 10), filter_signals(signal, 20)]:
    #     plot_long_short_signal(close[apple_ticker], filter_signal[apple_ticker])


    # lookahead_prices_5 = get_lookahead_prices(close, 5)
    # lookahead_prices_10 = get_lookahead_prices(close, 10)
    # lookahead_prices_20 = get_lookahead_prices(close, 20)
    #
    # for lookahead_price in (lookahead_5, lookahead_10, lookahead_20):
    #     plot_prices(lookahead_price[apple_ticker], title='lookahead x days prices')

    # project_tests.test_get_return_lookahead(get_log_return_lookahead)

    log_return_5 = get_log_return_lookahead(close, get_lookahead_prices(close, 5))
    log_return_10 = get_log_return_lookahead(close, get_lookahead_prices(close, 10))
    log_return_20 = get_log_return_lookahead(close, get_lookahead_prices(close, 20))

    signal_return_5 = get_signal_return(filter_signal_5, log_return_5)
    signal_return_10 = get_signal_return(filter_signal_10, log_return_10)
    signal_return_20 = get_signal_return(filter_signal_20, log_return_20)

    signal_return_list = [(signal_return_5[apple_ticker], filter_signal_5[apple_ticker], 5),
     (signal_return_10[apple_ticker], filter_signal_10[apple_ticker], 10),
     (signal_return_20[apple_ticker], filter_signal_20[apple_ticker], 20)
     ]

    # for one_signal_return in signal_return_list:
        # plot_signal_returns(close[apple_ticker], one_signal_return,
        #                     '{} day LookaheadSignal Returns for {} Stock'.format(one_signal_return[2], apple_ticker))

    for signal_return_tuple in zip([signal_return_5, signal_return_10, signal_return_20],[5, 10, 20]):
        # plot_returns_signal_histograms(signal_return_tuple[0], '{} days signal return'.format(signal_return_tuple[1]))
        plot_returns_signal_to_normal_histograms(signal_return_tuple[0], '{} days signal return'.format(signal_return_tuple[1]))

    # Filter out returns that don't have a long or short signal.
    long_short_signal_returns_5 = signal_return_5[signal_return_5 != 0].stack()
    long_short_signal_returns_10 = signal_return_10[signal_return_10 != 0].stack()
    long_short_signal_returns_20 = signal_return_20[signal_return_20 != 0].stack()

    # Get just ticker and signal return
    long_short_signal_returns_5 = long_short_signal_returns_5.reset_index().iloc[:, [1, 2]]
    long_short_signal_returns_5.columns = ['ticker', 'signal_return']
    long_short_signal_returns_10 = long_short_signal_returns_10.reset_index().iloc[:, [1, 2]]
    long_short_signal_returns_10.columns = ['ticker', 'signal_return']
    long_short_signal_returns_20 = long_short_signal_returns_20.reset_index().iloc[:, [1, 2]]
    long_short_signal_returns_20.columns = ['ticker', 'signal_return']

    # View some of the data
    long_short_signal_returns_5.head(20)

    ks_values_5, p_values_5 = calculate_kstest(long_short_signal_returns_5)
    ks_values_10, p_values_10 = calculate_kstest(long_short_signal_returns_10)
    ks_values_20, p_values_20 = calculate_kstest(long_short_signal_returns_20)
    print('ks_values_5')
    print(ks_values_5.head(10))
    print('p_values_5')
    print(p_values_5.head(10))

    ks_threshold = 0.8
    outliers_5 = find_outliers(ks_values_5, p_values_5, ks_threshold)
    outliers_10 = find_outliers(ks_values_10, p_values_10, ks_threshold)
    outliers_20 = find_outliers(ks_values_20, p_values_20, ks_threshold)

    outlier_tickers = outliers_5.union(outliers_10).union(outliers_20)
    print('{} Outliers Found:\n{}'.format(len(outlier_tickers), ', '.join(list(outlier_tickers))))

    good_tickers = get_tickers_without_outliers(close, outlier_tickers)
    lookahead_Ndays_signal_return_list = [(signal_return_5[good_tickers], 5),
                          (signal_return_10[good_tickers], 10),
                          (signal_return_20[good_tickers], 20)
                          ]
    for single_signal_return_tuple in lookahead_Ndays_signal_return_list:
        plot_returns_signal_to_normal_histograms(single_signal_return_tuple[0],
                                                 "After remove outliers, {} days ahead signal returns".format(single_signal_return_tuple[1]))