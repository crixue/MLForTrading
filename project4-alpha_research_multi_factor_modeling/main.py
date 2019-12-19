import sys
import cvxpy as cvx
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import AverageDollarVolume, Returns
from zipline.utils.calendars import get_calendar
from zipline.data import bundles
from zipline.data.bundles import ingest
from zipline.data.bundles.csvdir import csvdir_equities
from zipline.data.data_portal import DataPortal
import graphviz
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.factors import SimpleMovingAverage
from sklearn.decomposition import PCA
from zipline.pipeline.factors import Returns
import alphalens as al

from AbstractOptimalHoldings import AbstractOptimalHoldings
import extension
import project_helper
import project_tests

# %matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)
# Get nicer looking graphs for retina displays
# %config InlineBackend.figure_format = 'retina'

bundle_name = 'spy-quotemedia-bundle'

# environ = {
#             'ZIPLINE_ROOT': os.path.join(os.getcwd(), "data"),
#             'QUANDL_API_KEY': 'S7SBmxBJhN7nzpx97ue1',
#         }
# Create an ingest function
# Register the data bundle and its ingest function
bundles.register(bundle_name, extension.squant_bundle)


# N=10
# high_dollar_volume = universe.percentile_between(N, 100)
# recent_returns = Returns(window_length=N, mask=high_dollar_volume)
# low_returns = recent_returns.percentile_between(0, 10)
# high_returns = recent_returns.percentile_between(N, 100)
# trading_calendar = get_calendar('NYSE')
# Load the data bundle
bundle_data = bundles.load(bundle_name)

# 为每一只股票都生成一个120天平均美元成交量的收盘价。
universe = AverageDollarVolume(window_length=120).top(100)

# Create an empty Pipeline with the given screen
# pipeline = Pipeline(screen = universe)

# Set the dataloader
pricing_loader = USEquityPricingLoader(bundle_data.equity_daily_bar_reader, bundle_data.adjustment_reader)

# Define the function for the get_loader parameter
def choose_loader(column):
    if column not in USEquityPricing.columns:
        raise Exception('Column not in USEquityPricing')
    return pricing_loader

# Set the trading calendar
trading_calendar = get_calendar('NYSE')

# build pipeline and pipeline engine
engine = project_helper.build_pipeline_engine(bundle_data, trading_calendar)

# Set the start and end dates
# start_date = pd.Timestamp('2011-01-05', tz = 'utc')
# end_date = pd.Timestamp('2016-01-05', tz = 'utc')
#
# # Run our pipeline for the given start and end dates
# pipeline_output = engine.run_pipeline(pipeline, start_date, end_date)
#
# # If the pipeline output is a MultiIndex Dataframe we print the two levels of the index
# # if isinstance(pipeline_output.index, pd.core.index.MultiIndex):
# #     # We print the index level 0
# #     print('Index Level 0:\n\n', pipeline_output.index.get_level_values(0), '\n')
# #     # We print the index level 1
# #     print('Index Level 1:\n\n', pipeline_output.index.get_level_values(1), '\n')
#
universe_end_date = pd.Timestamp('2019-01-03', tz='UTC')
universe_tickers = engine.run_pipeline(
        Pipeline(screen=universe),
        universe_end_date,
        universe_end_date)\
    .index.get_level_values(1)\
    .values.tolist()

data_portal = DataPortal(
    bundle_data.asset_finder,
    trading_calendar=trading_calendar,
    first_trading_day=bundle_data.equity_daily_bar_reader.first_trading_day,
    equity_minute_reader=None,
    equity_daily_reader=bundle_data.equity_daily_bar_reader,
    adjustment_reader=bundle_data.adjustment_reader)

# # Create a factor that computes the 15-day mean closing price of securities
# mean_close_15 = SimpleMovingAverage(inputs = [USEquityPricing.close], window_length = 15)
# # Add the factor to our pipeline
# pipeline.add(mean_close_15, '15 Day MCP')
# # Render the pipeline as a DAG
# # pipeline.show_graph()
# # Set starting and end dates
# start_date = pd.Timestamp('2014-01-08', tz='utc')
# end_date = pd.Timestamp('2016-01-05', tz='utc')
# # Run our pipeline for the given start and end dates
# output = engine.run_pipeline(pipeline, start_date, end_date)
# # Display the pipeline output
# output.head()


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


class OptimalHoldingsRegualization(OptimalHoldings):
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

        return cvx.Minimize(-alpha_vector.T.values[0] * weights + self.lambda_reg * cvx.pnorm(weights, 2))


    def __init__(self, lambda_reg=0.5, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55,
                 weights_min=-0.55):
        self.lambda_reg = lambda_reg
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min


class OptimalHoldingsStrictFactor(OptimalHoldings):
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

        # TODO: Implement function
        alpha_vec_vals = alpha_vector.values[:, 0]
        x_star = (alpha_vec_vals - np.mean(alpha_vec_vals)) / sum(abs(alpha_vec_vals))
        return cvx.Minimize(cvx.pnorm(weights - x_star, 2))


def get_pricing(data_portal, trading_calendar, assets, start_date, end_date, field='close'):
    '''
    get history window price
    :param data_portal:
    :param trading_calendar:
    :param assets:
    :param start_date:
    :param end_date:
    :param field:
    :return:
    '''
    end_dt = pd.Timestamp(end_date.strftime('%Y-%m-%d'), tz='UTC', offset='C')
    start_dt = pd.Timestamp(start_date.strftime('%Y-%m-%d'), tz='UTC', offset='C')

    end_loc = trading_calendar.closes.index.get_loc(end_dt)
    start_loc = trading_calendar.closes.index.get_loc(start_dt)

    return data_portal.get_history_window(
        assets=assets,
        end_dt=end_dt,
        bar_count=end_loc - start_loc,
        frequency='1d',
        field=field,
        data_frequency='daily')


def fit_pca(returns, num_factor_exposeures, svd_solver):
    '''
    使用PCA（主成成分分析）来构建静态的risk model
    求出的是returns的特征值矩阵
    :param returns:
    :param num_factor_exposeures:
    :param svd_solver:
    :return: Model fit to returns
    '''

    return PCA(n_components=num_factor_exposeures, svd_solver=svd_solver).fit(returns)


def factor_betas(pca, factor_beta_indices, factor_beta_columns):
    '''
    从主成分分析模型中得到因子β
    Parameters
    ----------
    pca : PCA
        Model fit to returns
    factor_beta_indices : 1 dimensional Ndarray
        Factor beta indices
    factor_beta_columns : 1 dimensional Ndarray
        Factor beta columns

    Returns
    -------
    factor_betas : DataFrame
        Factor betas
    '''
    assert len(factor_beta_indices.shape) == 1
    assert len(factor_beta_columns.shape) == 1

    return pd.DataFrame(data=pca.components_.T, index=factor_beta_indices, columns=factor_beta_columns)


def factor_returns(pca, returns, factor_return_indices, factor_return_columns):
    """
    Get the factor returns from the PCA model.
    PCA transform 将数据降维到num_factor_exposeures后的数据

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    returns : DataFrame
        Returns for each ticker and date
    factor_return_indices : 1 dimensional Ndarray
        Factor return indices
    factor_return_columns : 1 dimensional Ndarray
        Factor return columns

    Returns
    -------
    factor_returns : DataFrame
        Factor returns
    """
    assert len(factor_return_indices.shape) == 1
    assert len(factor_return_columns.shape) == 1

    return pd.DataFrame(data=pca.transform(returns), index=factor_return_indices, columns=factor_return_columns)


def factor_cov_matrix(factor_returns, ann_factor):
    """
    Get the factor covariance matrix
    PCA model中不同因素之间的协方差为0，即没有关系
    Parameters
    ----------
    factor_returns : DataFrame
        Factor returns
    ann_factor : int
        Annualization factor

    Returns
    -------
    factor_cov_matrix : DataFrame
        Factor covariance matrix
    """

    return pd.DataFrame(np.diag(factor_returns.var(axis=0, ddof=1) * ann_factor))


def idiosyncratic_var_matrix(returns, factor_returns, factor_betas, ann_factor):
    """
    Get the idiosyncratic variance matrix

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    factor_returns : DataFrame
        Factor returns
    factor_betas : DataFrame
        Factor betas
    ann_factor : int
        Annualization factor

    Returns
    -------
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    """
    residuals_ = returns - pd.DataFrame(np.dot(factor_returns, factor_betas.T), index=returns.index, columns=returns.columns)
    return pd.DataFrame(np.diag(residuals_.var(axis=0, ddof=1) * ann_factor),returns.columns,returns.columns)


def idiosyncratic_var_vector(returns, idiosyncratic_var_matrix):
    """
    Get the idiosyncratic variance vector

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix

    Returns
    -------
    idiosyncratic_var_vector : DataFrame
        Idiosyncratic variance Vector
    """

    return pd.DataFrame(np.diag(idiosyncratic_var_matrix), returns.columns)


def predict_portfolio_risk(factor_betas, factor_cov_matrix, idiosyncratic_var_matrix, weights):
    """
    Get the predicted portfolio risk

    Formula for predicted portfolio risk is sqrt(X.T(BFB.T + S)X) where:
      X is the portfolio weights
      B is the factor betas
      F is the factor covariance matrix
      S is the idiosyncratic variance matrix

    Parameters
    ----------
    factor_betas : DataFrame
        Factor betas
    factor_cov_matrix : 2 dimensional Ndarray
        Factor covariance matrix
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    weights : DataFrame
        Portfolio weights

    Returns
    -------
    predicted_portfolio_risk : float
        Predicted portfolio risk
    """
    assert len(factor_cov_matrix.shape) == 2

    return np.sqrt(weights.T.dot(factor_betas.dot(factor_cov_matrix.dot(factor_betas.T)) + idiosyncratic_var_matrix).dot(weights)).values[0][0]


def momentum_1yr(window_length, universe, sector):
    return Returns(window_length=window_length, mask=universe)\
        .demean()\
        .rank()\
        .zscore()


def mean_reversion_5day_sector_neutral(window_length, universe, sector):
    """
    Generate the mean reversion 5 day sector neutral factor

    Parameters
    ----------
    window_length : int
        Returns window length
    universe : Zipline Filter
        Universe of stocks filter
    sector : Zipline Classifier
        Sector classifier

    Returns
    -------
    factor : Zipline Factor
        Mean reversion 5 day sector neutral factor
    """

    # TODO: Implement function
    return momentum_1yr(window_length, universe, sector) * -1


def mean_reversion_5day_sector_neutral_smoothed(window_length, universe, sector):
    """
    Generate the mean reversion 5 day sector neutral smoothed factor

    Parameters
    ----------
    window_length : int
        Returns window length
    universe : Zipline Filter
        Universe of stocks filter
    sector : Zipline Classifier
        Sector classifier

    Returns
    -------
    factor : Zipline Factor
        Mean reversion 5 day sector neutral smoothed factor
    """

    # TODO: Implement function
    return SimpleMovingAverage(inputs=[mean_reversion_5day_sector_neutral(window_length, universe, sector)], window_length=window_length)\
        .demean().rank().zscore()


class CTO(Returns):
    inputs = [USEquityPricing.open, USEquityPricing.close]

    def compute(self, today, assets, out, opens, closes):
        """
        The opens and closes matrix is 2 rows x N assets, with the most recent at the bottom.
        As such, opens[-1] is the most recent open, and closes[0] is the earlier close

        这里window_length取2，即取遍历的当日和前一天的数据来计算
        """
        out[:] = (opens[-1] - closes[0]) / closes[0]


class TrailingOvernightReturns(Returns):
    """
    Sum of trailing 1m O/N returns

    window_length 是取多少天的数据来计算
    """
    window_safe = True

    def compute(self, today, asset_ids, out, cto):
        out[:] = np.nansum(cto, axis=0)


def overnight_sentiment(cto_window_length, trail_overnight_returns_window_length, universe):
    cto_out = CTO(mask=universe, window_length=cto_window_length)
    return TrailingOvernightReturns(inputs=[cto_out], window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()


def overnight_sentiment_smoothed(cto_window_length, trail_overnight_returns_window_length, universe):
    unsmoothed_factor = overnight_sentiment(cto_window_length, trail_overnight_returns_window_length, universe)
    return SimpleMovingAverage(inputs=[unsmoothed_factor], window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()


def sharpe_ratio(factor_returns, annualization_factor):
    """
    Get the sharpe ratio for each factor for the entire period

    Parameters
    ----------
    factor_returns : DataFrame
        Factor returns for each factor and date
    annualization_factor: float
        Annualization Factor

    Returns
    -------
    sharpe_ratio : Pandas Series of floats
        Sharpe ratio
    """

    return annualization_factor * factor_returns.mean() / factor_returns.std()


if __name__ == '__main__':

    five_year_returns_df = get_pricing(
        data_portal,
        trading_calendar,
        universe_tickers,
        universe_end_date - pd.DateOffset(years=5),
        universe_end_date
    )
    five_year_returns = five_year_returns_df.pct_change()[1:].fillna(0)
    print(five_year_returns)

    num_factor_exposures = 20
    pca = fit_pca(five_year_returns, num_factor_exposures, 'full')
    print(pca.components_)
    plt.bar(np.arange(num_factor_exposures), pca.explained_variance_ratio_)
    # plt.show()

    risk_model = {}
    risk_model['factor_betas'] = factor_betas(pca, five_year_returns.columns.values, np.arange(num_factor_exposures))
    print(risk_model['factor_betas'])

    risk_model['factor_returns'] = factor_returns(
        pca,
        five_year_returns,
        five_year_returns.index,
        np.arange(num_factor_exposures))
    risk_model['factor_returns'].cumsum().plot(legend=None);

    ann_factor = 252
    risk_model['factor_cov_matrix'] = factor_cov_matrix(risk_model['factor_returns'], ann_factor)
    print(risk_model['factor_cov_matrix'])

    risk_model['idiosyncratic_var_matrix'] = idiosyncratic_var_matrix(five_year_returns, risk_model['factor_returns'],
                                                                      risk_model['factor_betas'], ann_factor)
    print(risk_model['idiosyncratic_var_matrix'])

    risk_model['idiosyncratic_var_vector'] = idiosyncratic_var_vector(five_year_returns,
                                                                      risk_model['idiosyncratic_var_matrix'])
    print(risk_model['idiosyncratic_var_vector'])

    all_weights = pd.DataFrame(np.repeat(1 / len(universe_tickers), len(universe_tickers)), universe_tickers)
    portfolio_risk = predict_portfolio_risk(
        risk_model['factor_betas'],
        risk_model['factor_cov_matrix'],
        risk_model['idiosyncratic_var_matrix'],
        all_weights)
    print("predict_portfolio_risk:{}".format(portfolio_risk))

    """
    factor_start_date = universe_end_date - pd.DateOffset(years=2)
    window_length = 5

    pipeline = Pipeline(screen=universe)
    pipeline.add(
        mean_reversion_5day_sector_neutral(window_length, universe, ""),
        'Mean_Reversion_5Day_Sector_Neutral')
    mean_reversion_5day_sector_neutral_pipeline_output = engine.run_pipeline(pipeline, factor_start_date, universe_end_date)
    print(mean_reversion_5day_sector_neutral_pipeline_output)

    pipeline.add(
        mean_reversion_5day_sector_neutral_smoothed(5, universe, ""),
        'Mean_Reversion_5Day_Sector_Neutral_Smoothed')
    mean_reversion_5day_sector_neutral_smoothed_pipeline_output = engine.run_pipeline(pipeline, factor_start_date,
                                                                             universe_end_date)
    print(mean_reversion_5day_sector_neutral_smoothed_pipeline_output)

    pipeline.add(
        overnight_sentiment(2, 5, universe),
        'Overnight_Sentiment')
    overnight_Sentiment_pipeline_output = engine.run_pipeline(pipeline, factor_start_date,
                                                                                      universe_end_date)
    print(overnight_Sentiment_pipeline_output)
    """

    # combine factors to a single pipeline
    # universe_end_date = pd.Timestamp('2019-01-03', tz='UTC')
    factor_start_date = universe_end_date - pd.DateOffset(years=2)
    universe = AverageDollarVolume(window_length=120).top(100)
    sector = ""

    pipeline = Pipeline(screen=universe)
    pipeline.add(
        momentum_1yr(252, universe, sector),
        'Momentum_1YR')
    pipeline.add(
        mean_reversion_5day_sector_neutral(5, universe, sector),
        'Mean_Reversion_5Day_Sector_Neutral')
    pipeline.add(
        mean_reversion_5day_sector_neutral_smoothed(5, universe, sector),
        'Mean_Reversion_5Day_Sector_Neutral_Smoothed')
    pipeline.add(
        overnight_sentiment(2, 5, universe),
        'Overnight_Sentiment')
    pipeline.add(
        overnight_sentiment_smoothed(2, 5, universe),
        'Overnight_Sentiment_Smoothed')
    all_factors = engine.run_pipeline(pipeline, factor_start_date, universe_end_date)

    print(all_factors)

    assets = all_factors.index.levels[1].values.tolist()
    pricing = get_pricing(
        data_portal,
        trading_calendar,
        assets,
        factor_start_date,
        universe_end_date)

    clean_factor_data = {
        factor: al.utils.get_clean_factor_and_forward_returns(factor=factor_data, prices=pricing, periods=[1])
        for factor, factor_data in all_factors.iteritems()}

    unixt_factor_data = {
        factor: factor_data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in factor_data.index.values],
            names=['date', 'asset']))
        for factor, factor_data in clean_factor_data.items()}

    ls_factor_returns = pd.DataFrame()
    for factor, factor_data in clean_factor_data.items():
        ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]
    (1 + ls_factor_returns).cumprod().plot();
    plt.show()

    momentum_1yr_factor_data = all_factors['Momentum_1YR']
    factor_data_test = al.utils.get_clean_factor_and_forward_returns(factor=momentum_1yr_factor_data, prices=pricing, periods=[1])
    # al.tears.create_full_tear_sheet(factor_data=factor_data_test)

    qr_factor_returns = pd.DataFrame()
    for factor, factor_data in unixt_factor_data.items():
        qr_factor_returns[factor] = al.performance.mean_return_by_quantile(factor_data)[0].iloc[:, 0]
    (10000 * qr_factor_returns).plot.bar(
        subplots=True,
        sharey=True,
        layout=(4, 2),
        figsize=(14, 14),
        legend=False);
    plt.show()

    ls_FRA = pd.DataFrame()
    for factor, factor_data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)
    ls_FRA.plot(title="Factor Rank Autocorrelation");
    plt.show()

    ic_factor_returns = pd.DataFrame()
    for factor, factor_data in unixt_factor_data.items():
        ic_factor_returns[factor] = al.performance.mean_information_coefficient(factor_data)
    print(ic_factor_returns)

    daily_annualization_factor = np.sqrt(252)
    sharpe_ratio = sharpe_ratio(ls_factor_returns, daily_annualization_factor).round(2)
    print(sharpe_ratio)

    selected_factors = all_factors.columns[[1, 2, 4]]
    print('Selected Factors: {}'.format(', '.join(selected_factors)))
    all_factors['alpha_vector'] = all_factors[selected_factors].mean(axis=1)
    alphas = all_factors[['alpha_vector']]
    alpha_vector = alphas.loc[all_factors.index.get_level_values(0)[-1]]
    print(all_factors)

    factor_betas = risk_model['factor_betas']
    factor_cov_matrix = risk_model['factor_cov_matrix']
    idiosyncratic_var_vector = risk_model['idiosyncratic_var_vector']
    optimal_weights = OptimalHoldings().find(alpha_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector)
    optimal_weights.plot.bar(legend=None, title='Portfolio % Holdings by Stock')
    x_axis = plt.axes().get_xaxis()
    x_axis.set_visible(False)
    plt.show()

    optimal_weights_1 = OptimalHoldingsRegualization(lambda_reg=5.0).find(alpha_vector, factor_betas,
                                                                          factor_cov_matrix,
                                                                          idiosyncratic_var_vector)

    optimal_weights_1.plot.bar(legend=None, title='Portfolio % Holdings by Stock')
    x_axis = plt.axes().get_xaxis()
    x_axis.set_visible(False)
    plt.show()

    optimal_weights_2 = OptimalHoldingsStrictFactor(
        weights_max=0.02,
        weights_min=-0.02,
        risk_cap=0.0015,
        factor_max=0.015,
        factor_min=-0.015).find(alpha_vector, risk_model['factor_betas'], risk_model['factor_cov_matrix'],
                                risk_model['idiosyncratic_var_vector'])

    optimal_weights_2.plot.bar(legend=None, title='Portfolio % Holdings by Stock')
    x_axis = plt.axes().get_xaxis()
    x_axis.set_visible(False)
    plt.show()