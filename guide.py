import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import datetime
import time
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
engine = create_engine('mysql+pymysql://root:123456@localhost:3306/stock')


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def get_data_compute_daily_returns(codes, start_date, end_date):
    data = {'log_date': pd.date_range(start_date, end_date)}
    df = pd.DataFrame(data)
    for code in codes:
        sql = '''
              SELECT `log_date`, close_price, `name` FROM `st_history_51` 
              where `code`=%(code)s
              and log_date>=%(start_date)s
              and log_date<=%(end_date)s;'''
        df_temp = pd.read_sql_query(sql, engine, params={'code': code, 'start_date': start_date, 'end_date': end_date}, index_col='log_date', parse_dates=True)
        name = df_temp['name'].ix[0]
        df_temp = df_temp.rename(columns={'close_price': name}).drop(['name'], axis=1)
        df = df.merge(df_temp, on='log_date').dropna(axis=0)
        # df = pd.merge(df, df_temp,on=['log_date', 'log_date'])
    df = df.set_index('log_date')  #一定要设置date的索引，否则在绘图时会报错
    print(df)
    return df


def compute_daily_returns(df):
    """日均收益率"""
    daily_return = (df / df.shift(1)) - 1;
    daily_return = daily_return[1:] # pandas 使第一行都为NAN
    # daily_return.ix[0, :] = 0  # pandas 使第一行都为NAN
    return daily_return


def test_compute_daily_returns():
    code = (600199, 510230)
    start_date = '2018-05-18 00:00:00'
    end_date = '2018-06-18 00:00:00'
    df = get_data_compute_daily_returns(code, start_date, end_date)
    plot_data(df)

    df = compute_daily_returns(df)
    plot_data(df, xlabel="date", ylabel="daily_return")


def get_data_rolling_mean(code, start_date, end_date):
    data = {'log_date': pd.date_range(start_date, end_date)}
    df = pd.DataFrame(data)
    sql = '''
              SELECT `log_date`, close_price, `name` FROM `st_history_56` 
              where `code`=%(code)s
              and log_date>=%(start_date)s
              and log_date<=%(end_date)s;'''
    df_temp = pd.read_sql_query(sql, engine, params={'code': code, 'start_date': start_date, 'end_date': end_date},
                                index_col='log_date', parse_dates=True)
    df_temp = df_temp.rename(columns={'close_price': code}).drop(['name'], axis=1)  #close_price 换成股票的名称， 去掉name这个列
    df = df.merge(df_temp, on='log_date').dropna(axis=0)  # 去掉行中有nan的列
    df = df.set_index('log_date')  # 一定要设置date的索引，否则在绘图时会报错
    return df


def get_rolling_mean(dataFrame, window):
    """移动平均值"""
    return dataFrame.rolling(window=window).mean()


def get_rolling_std(dataFrame, window):
    """ 移动标准差"""
    return dataFrame.rolling(window=window).std()


def get_bollinger_bands(rm, rstd):
    """返回布林线的上下线"""
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band


def test_rolling_mean():
    code = 600519
    start_date = '2017-01-01 00:00:00'
    end_date = '2019-01-01 00:00:00'
    df = get_data_rolling_mean(code, start_date, end_date)
    df_copy = df.copy().rename(columns={code: 'close_price'})
    df = df.rolling(window=22).mean().rename(columns={code: 'rm_price'})
    df = df.merge(df_copy, how="right", left_index=True, right_index=True)
    plot_data(df, xlabel="date", ylabel="price")


def test_bollinger_bands():
    code = 600519
    start_date = '2017-01-01 00:00:00'
    end_date = '2019-01-01 00:00:00'
    df = get_data_rolling_mean(code, start_date, end_date)
    df_rm = get_rolling_mean(df, window=22)
    df_rstd = get_rolling_std(df, window=22)
    upper_band, lower_band = get_bollinger_bands(df_rm, df_rstd)
    ax = df.rename(columns={code: "price"}).plot(title="bollinger bands", label=code)
    ax.set_xlabel("date")
    ax.set_ylabel("price")

    df_rm.rename(columns={code: "rolling_mean"}).plot(ax=ax)
    # df_rstd.rename(columns={code: "rolling_std"}).plot(ax=ax)
    upper_band.rename(columns={code: "upper_band"}).plot(ax=ax)
    lower_band.rename(columns={code: "lower_band"}).plot(ax=ax)
    plt.show()


def compute_cumulative_returns(df):
    """累计回报率"""
    daily_return = (df / df[0].values) - 1;
    daily_return = daily_return[1:]  # pandas 使第一行都为NAN
    # daily_return.ix[0, :] = 0  #pandas 使第一行都为NAN
    return daily_return


def test_compute_cumulative_returns():
    code = 600519
    start_date = '2017-01-01 00:00:00'
    end_date = '2019-01-01 00:00:00'
    df = get_data_rolling_mean(code, start_date, end_date)
    df = compute_daily_returns(df).rename(columns={code: "cumulative returns"})
    plot_data(df, title="cumulative_returns")


def get_data_compute_daily_returns_1(codes, start_date, end_date):
    data = {'log_date': pd.date_range(start_date, end_date)}
    df = pd.DataFrame(data)
    for code in codes:
        sql = '''
              SELECT `log_date`, close_price, `name` FROM `st_history_56` 
              where `code`=%(code)s
              and log_date>=%(start_date)s
              and log_date<=%(end_date)s;'''
        df_temp = pd.read_sql_query(sql, engine, params={'code': code, 'start_date': start_date, 'end_date': end_date}, index_col='log_date', parse_dates=True)
        df_temp = df_temp.rename(columns={'close_price': code}).drop(['name'], axis=1)
        df = df.merge(df_temp, on='log_date').dropna(axis=0)
        # df = pd.merge(df, df_temp,on=['log_date', 'log_date'])
    df = df.set_index('log_date')  #一定要设置date的索引，否则在绘图时会报错
    print(df)
    return df


def test_plot_two_of_daily_return_histogram_together():
    """
    通过日均收益率绘制直方图
    :return:
    """
    code = (510390, 600519)
    start_date = '2017-01-01 00:00:00'
    end_date = '2019-01-01 00:00:00'
    df = get_data_compute_daily_returns_1(code, start_date, end_date)

    df = compute_daily_returns(df)
    df[510390].hist(bins=22, label="平安300")
    df[600519].hist(bins=22, label="贵州茅台")
    plt.legend(loc="upper right")
    plt.show()


def get_data_scatter(code, start_date, end_date, tbl_name):
    data = {'log_date': pd.date_range(start_date, end_date)}
    df = pd.DataFrame(data)
    sql = '''
                  SELECT `log_date`, close_price, `name` FROM {0} 
                  where `code`=%(code)s
                  and log_date>=%(start_date)s
                  and log_date<=%(end_date)s order by log_date asc;'''
    sql = sql.format(tbl_name)
    df_temp = pd.read_sql_query(sql, engine, params={'code': code, 'start_date': start_date, 'end_date': end_date},
                                index_col='log_date', parse_dates=True)
    df_temp = df_temp.rename(columns={'close_price': code}).drop(['name'], axis=1)  # close_price 换成股票的名称， 去掉name这个列
    df = df.merge(df_temp, on='log_date').dropna(axis=0)  # 去掉行中有nan的列
    df = df.set_index('log_date')  # 一定要设置date的索引，否则在绘图时会报错
    return df


def test_scatter_plots():
    code = '000001'
    start_date = '2017-01-01 00:00:00'
    end_date = '2019-03-01 00:00:00'
    tbl_name = 'st_history_1'
    df_basic = get_data_scatter(code, start_date, end_date,tbl_name) # 上证指数
    df_basic = compute_daily_returns(df_basic)

    code1 = '601336'
    tbl_name1 = 'st_history_16'
    df_1 = get_data_scatter(code1, start_date, end_date, tbl_name1)  # 新华保险
    df_1 = compute_daily_returns(df_1)
    df_basic_1 = df_basic.merge(df_1, on='log_date').fillna(method='ffill')
    df_basic_1.plot(kind="scatter", x=code, y=code1)
    beta_basic1, alpha_basic1 = np.polyfit(df_basic_1[code], df_basic_1[code1], deg=1) # 一次线性拟合数据
    print("新华保险 beta: %s", beta_basic1)
    print("新华保险 alpha: %s", alpha_basic1)
    print(df_basic_1.corr(method='pearson'))
    plt.plot(df_basic_1[code], beta_basic1 * df_basic_1[code] + alpha_basic1, '-', color='red')
    plt.show()

    code2 = '600031'
    tbl_name2 = 'st_history_12'
    df_2 = get_data_scatter(code2, start_date, end_date, tbl_name2)  # 三一重工
    df_2 = compute_daily_returns(df_2)
    df_basic_2 = df_basic.merge(df_2, on='log_date').fillna(method='ffill')
    df_basic_2.plot(kind="scatter", x=code, y=code2)
    beta_basic2, alpha_basic2 = np.polyfit(df_basic_2[code], df_basic_2[code2], deg=1)  # 拟合数据
    print("三一重工 beta{}", beta_basic2)
    print("三一重工 alpha{}", alpha_basic2)
    print(df_basic_2.corr(method="pearson"))
    plt.plot(df_basic_2[code], beta_basic2 * df_basic_2[code] + alpha_basic2, '-', color='red')
    plt.show()


def normalized_price(stock_lists):
    '''
    归一化
    :param stock_lists:
    :return:
    '''
    normalized_stocks = []
    for stock in stock_lists:
        code = stock[0]
        df = stock[1]
        df['normalized_price'] = df[code] / df[code].iloc[0]
        df = df.drop(columns=code)
        normalized_stocks.append((code, df))
    return normalized_stocks


def allocated_stock(allocas_stocks):
    '''
    根据投资组合按比例分配
    :param allocas_stocks:
    :return:
    '''
    list = []
    for normalized_stocks, weight in allocas_stocks:
        code = normalized_stocks[0]
        df = normalized_stocks[1]
        df = df * weight
        df = df.rename(columns={'normalized_price': 'weighted_daily_return'})
        # print(df)
        list.append((code, df))
    return list


def sum_everyday_val(init_invest_val, allocas_stocks):
    total_df = []
    codes = []
    for code, stock_df in allocas_stocks:
        codes.append(code)
        temp_df = stock_df.rename(columns={'weighted_daily_return': code})
        total_df.append(temp_df)
    df = pd.concat(total_df, axis=1).fillna(method='ffill', axis=0)
    df = df * init_invest_val
    df['total_pos'] = df.sum(axis=1)
    return df


def cal_SR(df) :
    '''
    计算没有日期粒度的夏普比率
    :param df:
    :return:
    '''
    return df.mean() / df.std()

def cal_sharpe_ratio():
    '''
    计算夏普比率
    :return:
    '''
    start_date = '2017-01-01 00:00:00'
    end_date = '2019-03-01 00:00:00'

    code1 = '601336'  # 贵州茅台
    tbl_name1 = 'st_history_16'
    code2 = '600031' # 新华保险
    tbl_name2 = 'st_history_12'
    code3 = '600519'  # 三一重工
    tbl_name3 = 'st_history_56'
    code4 = '601012'  # 隆基股份
    tbl_name4 = 'st_history_56'

    gzmt_df = get_data_scatter(code1, start_date, end_date, tbl_name1)
    xhbx_df = get_data_scatter(code2, start_date, end_date, tbl_name2)
    syzg_df = get_data_scatter(code3, start_date, end_date, tbl_name3)
    ljgf_df = get_data_scatter(code4, start_date, end_date, tbl_name4)

    stock_list_unnormalized = [(code1,gzmt_df), (code2, xhbx_df), (code3, syzg_df), (code4, ljgf_df)]
    normalized_stocks = normalized_price(stock_list_unnormalized)
    weight = (0.1, 0.4, 0.3, 0.2)
    normalized_stocks_zip = zip(normalized_stocks, weight)
    sum_everyday_df = sum_everyday_val(10000, allocated_stock(normalized_stocks_zip))
    plot_data(sum_everyday_df['total_pos'], title="Total Portfolio Value", xlabel="date", ylabel="value")  # 日均总收益

    total_daily_return = compute_daily_returns(sum_everyday_df['total_pos'])
    SR = cal_SR(total_daily_return)
    ASR = np.sqrt(252) * SR
    print('total pos ASR:', ASR)


if __name__ == '__main__':
    # test_compute_daily_returns()
    # test_rolling_mean()
    # test_bollinger_bands()
    # test_compute_cumulative_returns()
    # test_plot_two_of_daily_return_histogram_together()
    # test_scatter_plots()
    cal_sharpe_ratio()