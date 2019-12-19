import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from zipline.data.bundles import register, ingest
import sqlite3
import requests
from zipline.utils.cli import maybe_show_progress

import matplotlib.pyplot as plt

import logging;

logging.basicConfig(level=logging.INFO)

IFIL = 'spy_history.db'
sqlite_conn = sqlite3.connect(IFIL, check_same_thread=False)

os.environ['ZIPLINE_ROOT'] = os.path.join(os.getcwd())

traceDebug = True # Set True to get trace messages
def _cachpath(symbol, type_):
    return '-'.join((symbol.replace(os.path.sep, '_'), type_))

def squant_bundle(environ,
               asset_db_writer,
               minute_bar_writer,
               daily_bar_writer,
               adjustment_writer,
               calendar,
               start_session,
               end_session,
               cache,
               show_progress,
               output_dir):
    """
    # Define our custom ingest function
    :param environ:
    :param asset_db_writer:
    :param minute_bar_writer:
    :param daily_bar_writer:
    :param adjustment_writer:
    :param calendar:
    :param start_session:
    :param end_session:
    :param cache:
    :param show_progress:
    :param output_dir:
    :return:
    """
    # reader = MysqlReader()
    # find all tickers name
    tickers_df_sql_text = "SELECT `id` as sid, `name` as asset_name from `stock_spy_tickers`"
    data = pd.read_sql(tickers_df_sql_text, con=sqlite_conn)
    tickers_df = pd.DataFrame(data=data)
    if traceDebug:
        print("tickers_df:")
        print(tickers_df)

    metadata = pd.DataFrame(np.empty(tickers_df.size, dtype=[
        ('start_date', 'datetime64[ns]'),
        ('end_date', 'datetime64[ns]'),
        ('auto_close_date', 'datetime64[ns]'),
        ('symbol', 'object'),
    ]))

    def _pricing_iter():
        with maybe_show_progress(
                tickers_df.iterrows(),
                show_progress,
                label='Fetch stocks pricing data from db: ') as it, \
                requests.Session() as session:
            for index, row in tickers_df.iterrows():
                symbol = row['asset_name']
                path = _cachpath(symbol, 'ohlcv')

                try:
                    data = cache[path]
                except:
                    sql_text = "SELECT tran_date as date, open, high, low, close, volume FROM `stock_spy` WHERE name='{0}' order by tran_date desc".format(symbol)
                    data = cache[path] = pd.read_sql(sql_text, con=sqlite_conn, index_col='date', parse_dates=['date']).sort_index()
                    if traceDebug:
                        print("read {} sql and get df data:".format(symbol))
                        print(data)

                # the start date is the date of the first trade and
                # the end date is the date of the last trade
                start_date = pd.to_datetime(data.iloc[0].name)
                end_date = pd.to_datetime(data.iloc[-1].name)
                if traceDebug:
                    print("start_date: ")
                    print(start_date)
                # The auto_close date is the day after the last trade.
                ac_date = end_date + pd.Timedelta(days=1)

                sid = row['sid']
                if traceDebug:
                    print("sid-{}:symbol-{}", sid, symbol)
                    print("start_date", type(start_date), start_date)
                    print("end_date", type(end_date), end_date)
                    print("ac_date", type(ac_date), ac_date)

                metadata.iloc[sid] = start_date, end_date, ac_date, symbol
                new_index = ['open', 'high', 'low', 'close', 'volume']
                data_df = data.reindex(columns=new_index, copy=False)  # fix bug

                sessions = calendar.sessions_in_range(start_date, end_date)
                data_df = data_df.reindex(
                    sessions.tz_localize(None),
                    copy=False,
                ).fillna(0.0)

                yield sid, data_df

    # 写入数据文件
    daily_bar_writer.write(_pricing_iter(), show_progress=True)

    # Hardcode the exchange to "YAHOO" for all assets and (elsewhere)
    # register "YAHOO" to resolve to the NYSE calendar, because these are
    # all equities and thus can use the NYSE calendar.
    metadata['exchange'] = "YAHOO"
    if traceDebug:
        print("returned from daily_bar_writer")
        print("calling asset_db_writer")
        print("metadata", type(metadata))

    # drop metadata nan val which exists in any items first
    metadata = metadata.dropna(axis=0, how="any")

    # Not sure why symbol_map is needed
    symbol_map = pd.Series(metadata.symbol.index, metadata.symbol)
    if traceDebug:
        print("symbol_map", type(symbol_map))
        print(symbol_map)

    # 写入基础信息
    asset_db_writer.write(equities=metadata)

    adjustment_writer.write()

    return

bundle_name = 'spy-quotemedia-bundle'
register(bundle_name,
         squant_bundle,
         calendar_name='NYSE'  # US equities
        )

ingest(bundle_name)

__all__ = [
    'squant_bundle'
]

