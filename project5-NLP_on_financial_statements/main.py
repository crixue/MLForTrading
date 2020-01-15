import nltk
import numpy as np
import pandas as pd
import pickle
import pprint
import project_helper
import project_tests
import os
import re
import sqlite3
import time
import alphalens as al

import project_helper

from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
from ratelimit import limits, sleep_and_retry

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('stopwords')
# nltk.download('wordnet')

cik_lookup = {
    'AMZN': '0001018724',
    'AMD': '0000002488',
    'AAPL': '0000320193',
    'ORCL': '0001341439',
    'BMY': '0000014272',
    'CNP': '0001130310',
    'CVX': '0000093410',
    'FL': '0000850209',
    'FRT': '0000034903',
    'HON': '0000773840'
}

additional_cik = {
    'AEP': '0000004904',
    'AXP': '0000004962',
    'BA': '0000012927',
    'BK': '0001390777',
    'CAT': '0000018230',
    'DE': '0000315189',
    'DIS': '0001001039',
    'DTE': '0000936340',
    'ED': '0001047862',
    'EMR': '0000032604',
    'ETN': '0001551182',
    'GE': '0000040545',
    'IBM': '0000051143',
    'IP': '0000051434',
    'JNJ': '0000200406',
    'KO': '0000021344',
    'LLY': '0000059478',
    'MCD': '0000063908',
    'MO': '0000764180',
    'MRK': '0000310158',
    'MRO': '0000101778',
    'PCG': '0001004980',
    'PEP': '0000077476',
    'PFE': '0000078003',
    'PG': '0000080424',
    'PNR': '0000077360',
    'SYY': '0000096021',
    'TXN': '0000097476',
    'UTX': '0000101829',
    'WFC': '0000072971',
    'WMT': '0000104169',
    'WY': '0000106535',
    'XOM': '0000034088'}


class SecAPI(object):
    SEC_CALL_LIMIT = {'calls': 10, 'seconds': 1}

    @staticmethod
    @sleep_and_retry
    # Dividing the call limit by half to avoid coming close to the limit
    @limits(calls=SEC_CALL_LIMIT['calls'] / 2, period=SEC_CALL_LIMIT['seconds'])
    def _call_sec(url):
        return requests.get(url)

    def get(self, url):
        return self._call_sec(url).text


sec_api = SecAPI()


def get_sec_data(cik, doc_type, start=0, count=60):
    newest_pricing_data = pd.to_datetime('2020-01-01')
    rss_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
        '&CIK={}&type={}&start={}&count={}&owner=exclude&output=atom' \
        .format(cik, doc_type, start, count)
    sec_data = sec_api.get(rss_url)
    feed = BeautifulSoup(sec_data.encode('ascii'), 'xml').feed
    entries = [
        (
            entry.content.find('filing-href').getText(),
            entry.content.find('filing-type').getText(),
            entry.content.find('filing-date').getText())
        for entry in feed.find_all('entry', recursive=False)
        if pd.to_datetime(entry.content.find('filing-date').getText()) <= newest_pricing_data]

    return entries


def get_documents(text):
    """
    Extract the documents from the text

    Parameters
    ----------
    text : str
        The text with the document strings inside

    Returns
    -------
    extracted_docs : list of str
        The document strings found in `text`
    """
    extracted_docs = []
    # regex for the tags
    start_pattern = re.compile(r'<DOCUMENT>')
    end_pattern = re.compile(r'</DOCUMENT>')

    # isolate indices of document bounds
    start_idx = [x.end() for x in re.finditer(start_pattern, text)]
    end_idx = [x.start() for x in re.finditer(end_pattern, text)]

    # append document body
    for doc_start, doc_end in zip(start_idx, end_idx):
        extracted_docs.append(text[doc_start:doc_end])
    return extracted_docs


def get_document_type(doc):
    """
    Return the document type lowercased

    Parameters
    ----------
    doc : str
        The document string

    Returns
    -------
    doc_type : str
        The document type lowercased
    """
    # find next word after <TYPE> tag
    type_pattern = re.compile(r'(?<=<TYPE>)\w+[^\n]+')
    doc_type = re.search(type_pattern, doc).group(0).lower()

    return doc_type


def remove_html_tags(text):
    text = BeautifulSoup(text, 'html.parser').get_text()

    return text


def clean_text(text):
    text = text.lower()
    text = remove_html_tags(text)

    return text


def cache_clean_text(ten_k, file_doc_key='file'):
    """
    因为计算多个文件的clean_text需要花费大量时间，所以计算一次后就直接当做文件存起来，下次直接使用
    :rtype: clean_text
    """
    file_name = ten_k['cik'] + '-' + ten_k['file_date'] + '-cache_clean_text.txt'
    cache_clean_text_file_path = os.path.join(os.getcwd(), "data", "cache", file_name)
    if os.path.exists(cache_clean_text_file_path):
        with open(cache_clean_text_file_path, 'r', encoding='utf8') as f:
            text = f.read()
        return text

    text = clean_text(ten_k[file_doc_key])
    with open(cache_clean_text_file_path, 'w+', encoding='utf8') as f:
        f.write(text)
        f.flush()
    return text


def lemmatize_words(words):
    """
    Lemmatize words

    lemmatize默认处理的是nouns的词干，我们处理的是verb词干

    Parameters
    ----------
    words : list of str
        List of words

    Returns
    -------
    lemmatized_words : list of str
        List of lemmatized words
    """
    wnl = WordNetLemmatizer()
    lem_words = [wnl.lemmatize(word, pos='v') for word in words]
    return lem_words


def is_lemmatize_english_stopwords(word):
    lemmatize_en_stopwords = lemmatize_words(stopwords.words('english'))
    return word in lemmatize_en_stopwords


def get_bag_of_words(sentiment_words, docs):
    """
    Generate a bag of words from documents for a certain sentiment

    Parameters
    ----------
    sentiment_words: Pandas Series
        Words that signify a certain sentiment
    docs : list of str
        List of documents used to generate bag of words

    Returns
    -------
    bag_of_words : 2-d Numpy Ndarray of int
        Bag of words sentiment for each document
        The first dimension is the document.
        The second dimension is the word.
    """
    cv = CountVectorizer(vocabulary=sentiment_words.values)
    cv_fit = cv.transform(docs)

    return cv_fit.toarray()


def get_jaccard_similarity(bag_of_words_matrix):
    """
    Get jaccard similarities for neighboring documents

    Parameters
    ----------
    bag_of_words : 2-d Numpy Ndarray of int
        Bag of words sentiment for each document
        The first dimension is the document.
        The second dimension is the word.

    Returns
    -------
    jaccard_similarities : list of float
        Jaccard similarities for neighboring documents
    """
    jaccard_similarities = []

    bag_of_words_bool_matrix = bag_of_words_matrix.astype(bool)
    for i in range(0, bag_of_words_matrix.shape[0]-1):
        jaccard_similarities.append(jaccard_similarity_score(bag_of_words_bool_matrix[i], bag_of_words_bool_matrix[i+1]))

    return jaccard_similarities


def get_tfidf(sentiment_words, docs):
    """
    Generate TFIDF values from documents for a certain sentiment

    Parameters
    ----------
    sentiment_words: Pandas Series
        Words that signify a certain sentiment
    docs : list of str
        List of documents used to generate bag of words

    Returns
    -------
    tfidf : 2-d Numpy Ndarray of float
        TFIDF sentiment for each document
        The first dimension is the document.
        The second dimension is the word.
    """
    # filter out words not in sentiment_words
    vectorizer = TfidfVectorizer(vocabulary=sentiment_words.values)
    fit_vect = vectorizer.fit_transform(docs)

    return fit_vect.toarray()


def get_cosine_similarity(tfidf_matrix):
    """
    Get cosine similarities for each neighboring TFIDF vector/document

    Parameters
    ----------
    tfidf : 2-d Numpy Ndarray of float
        TFIDF sentiment for each document
        The first dimension is the document.
        The second dimension is the word.

    Returns
    -------
    cosine_similarities : list of float
        Cosine similarities for neighboring documents
    """
    cosine_similarities = list(np.diag(cosine_similarity(tfidf_matrix, tfidf_matrix), k=1))
    return cosine_similarities


from sqlalchemy import create_engine
import datetime
IFIL = os.path.join(os.path.join(os.path.dirname(os.getcwd()), 'project4-alpha_research_multi_factor_modeling', 'spy_history.db'))
sqlite_conn = sqlite3.connect(IFIL, check_same_thread=False)
engine = create_engine('sqlite:///'+IFIL, echo=True)


def get_pricing_data():
    all_data_df = pd.DataFrame()
    rng = pd.date_range('19941231', periods=25, freq='A')
    for pd_date in rng:
        add_one_day_date = last_day_datetime = pd_date.to_datetime()
        first_day_of_year = last_day_datetime + datetime.timedelta(days=1)

        tickers_df_sql_text = "SELECT `name` as ticker, `tran_date` as date, `adj_close` from main.stock_spy where tran_date=:date"
        one_data = pd.read_sql(sql=tickers_df_sql_text, con=engine, params={'date': last_day_datetime.strftime("%Y-%m-%d %H:%M:%S")})

        check_days = 4
        while one_data.shape[0] == 0 and check_days > 0:
            add_one_day_date = add_one_day_date + datetime.timedelta(days=1)
            one_data = pd.read_sql(sql=tickers_df_sql_text, con=engine,
                                   params={'date': add_one_day_date.strftime("%Y-%m-%d %H:%M:%S")})
            check_days -= 1

        one_data['date'] = first_day_of_year
        tickers_df = one_data.pivot(index='date', columns='ticker', values='adj_close')
        all_data_df = all_data_df.append(tickers_df)

    return all_data_df


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
    example_ticker = 'AMZN'
    sec_data = {}

    for ticker, cik in cik_lookup.items():
        sec_data[ticker] = get_sec_data(cik, '10-K')
    pprint.pprint(sec_data[example_ticker][:5])

    raw_fillings_by_ticker = {}
    for ticker, data in sec_data.items():
        raw_fillings_by_ticker[ticker] = {}
        for index_url, file_type, file_date in tqdm(data, desc='Downloading {} Fillings'.format(ticker), unit='filling'):
            if (file_type == '10-K'):
                file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')

                file_name = file_url.split("/")[-1]
                file_path = os.path.join(os.getcwd(), "data", file_name)
                file_date = file_date[2:4]
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf8') as f:
                        raw_fillings_by_ticker[ticker][file_date] = f.read()
                    continue

                raw_fillings_by_ticker[ticker][file_date] = sec_api.get(file_url)
                with open(file_path, 'w+', encoding='utf8') as f:
                    f.write(raw_fillings_by_ticker[ticker][file_date])
                    f.flush()
    print('Example Document:\n\n{}...'.format(next(iter(raw_fillings_by_ticker[example_ticker].values()))[:1000]))

    filling_documents_by_ticker = {}
    for ticker, raw_fillings in raw_fillings_by_ticker.items():
        filling_documents_by_ticker[ticker] = {}
        for file_date, filling in tqdm(raw_fillings.items(), desc='Getting Documents from {} Fillings'.format(ticker), unit='filling'):
            filling_documents_by_ticker[ticker][file_date] = get_documents(filling)


    ten_ks_by_ticker = {}
    for ticker, filling_documents in filling_documents_by_ticker.items():
        ten_ks_by_ticker[ticker] = []
        for file_date, documents in filling_documents.items():
            for document in documents:
                if get_document_type(document) == '10-k':
                    ten_ks_by_ticker[ticker].append({
                        'cik': cik_lookup[ticker],
                        'file': document,
                        'file_date': file_date})
    project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], ['cik', 'file', 'file_date'])

    del document
    del documents
    del filling
    del filling_documents
    del filling_documents_by_ticker
    del raw_fillings
    del raw_fillings_by_ticker

    for ticker, ten_ks in ten_ks_by_ticker.items():
        for ten_k in tqdm(ten_ks, desc='Cleaning {} 10-Ks'.format(ticker), unit='10-K'):
            ten_k['file_clean'] = cache_clean_text(ten_k, 'file')
    project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], ['file_clean'])

    word_pattern = re.compile('\w+')
    for ticker, ten_ks in ten_ks_by_ticker.items():
        for ten_k in tqdm(ten_ks, desc='Lemmatize {} 10-Ks'.format(ticker), unit='10-K'):
            ten_k['file_lemma'] = lemmatize_words(word_pattern.findall(ten_k['file_clean']))
    project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], ['file_lemma'])

    lemma_english_stopwords = lemmatize_words(stopwords.words('english'))
    for ticker, ten_ks in ten_ks_by_ticker.items():
        for ten_k in tqdm(ten_ks, desc='Remove Stop Words for {} 10-Ks'.format(ticker), unit='10-K'):
            ten_k['file_lemma'] = [word for word in ten_k['file_lemma'] if word not in lemma_english_stopwords]
    print('Stop Words Removed')

    sentiments = ['negative', 'positive', 'uncertainty', 'litigious', 'constraining', 'interesting']
    sentiment_df = pd.read_csv(os.path.join(os.getcwd(), "data", "LoughranMcDonald_MasterDictionary_2018.csv"))
    sentiment_df.columns = [column.lower() for column in sentiment_df.columns]

    # Remove unused information
    sentiment_df = sentiment_df[sentiments + ['word']]
    sentiment_df[sentiments] = sentiment_df[sentiments].astype(bool)
    sentiment_df = sentiment_df[(sentiment_df[sentiments]).any(1)]

    # Apply the same preprocessing to these words as the 10-k words
    sentiment_df['word'] = lemmatize_words(sentiment_df['word'].str.lower())
    sentiment_df = sentiment_df.drop_duplicates('word')

    sentiment_bow_ten_ks = {}
    for ticker, ten_ks in ten_ks_by_ticker.items():
        lemma_docs = [' '.join(ten_k['file_lemma']) for ten_k in ten_ks]

        sentiment_bow_ten_ks[ticker] = {
            sentiment: get_bag_of_words(sentiment_df[sentiment_df[sentiment]]['word'], lemma_docs)
            for sentiment in sentiments}

    project_helper.print_ten_k_data([sentiment_bow_ten_ks[example_ticker]], sentiments)

    # Get dates for the universe & get jaccard_similarities
    file_dates = {
        ticker: [ten_k['file_date'] for ten_k in ten_ks]
        for ticker, ten_ks in ten_ks_by_ticker.items()}

    jaccard_similarities = {
        ticker: {
            sentiment_name: get_jaccard_similarity(sentiment_values)
            for sentiment_name, sentiment_values in ten_k_sentiments.items()}
        for ticker, ten_k_sentiments in sentiment_bow_ten_ks.items()}
    project_helper.plot_similarities(
        [jaccard_similarities[example_ticker][sentiment] for sentiment in sentiments],
        file_dates[example_ticker][1:],
        'Jaccard Similarities for {} Sentiment'.format(example_ticker),
        sentiments)

    # get sentiment_tfidf
    sentiment_tfidf_ten_ks = {}
    for ticker, ten_ks in ten_ks_by_ticker.items():
        lemma_docs = [' '.join(ten_k['file_lemma']) for ten_k in ten_ks]

        sentiment_tfidf_ten_ks[ticker] = {
            sentiment: get_tfidf(sentiment_df[sentiment_df[sentiment]]['word'], lemma_docs) for sentiment in sentiments
        }
    project_helper.print_ten_k_data([sentiment_tfidf_ten_ks[example_ticker]], sentiments)

    #Get cosine similarities
    cosine_similarities = {}
    for ticker, sentiment_tfidf_ten_k in sentiment_tfidf_ten_ks.items():
        cosine_similarities[ticker] = {
            sentiment: get_cosine_similarity(sentiment_tfidf_ten_k[sentiment]) for sentiment in sentiments
        }
    project_helper.plot_similarities(
        [cosine_similarities[example_ticker][sentiment] for sentiment in sentiments],
        file_dates[example_ticker][1:],
        'Cosine Similarities for {} Sentiment'.format(example_ticker),
        sentiments)

    #dict to dataFrame
    cosine_similarities_df_dict = {'date': [], 'ticker': [], 'sentiment': [], 'value': []}
    for ticker, ten_k_sentiments in cosine_similarities.items():
        for sentiment_name, sentiment_values in ten_k_sentiments.items():
            for sentiment_values, sentiment_value in enumerate(sentiment_values):
                cosine_similarities_df_dict['ticker'].append(ticker)
                cosine_similarities_df_dict['sentiment'].append(sentiment_name)
                cosine_similarities_df_dict['value'].append(sentiment_value)
                cut_out_date = file_dates[ticker][1:][sentiment_values]
                if int(cut_out_date) < 50:
                    cut_out_date = datetime.datetime(2000 + int(cut_out_date), 1, 1)
                else:
                    cut_out_date = datetime.datetime(1900 + int(cut_out_date), 1, 1)
                cosine_similarities_df_dict['date'].append(cut_out_date)

    cosine_similarities_df = pd.DataFrame(cosine_similarities_df_dict)

    pricing = get_pricing_data()

    # Alphalens Format
    factor_data = {}
    skipped_sentiments = []
    for sentiment in sentiments:
        cs_df = cosine_similarities_df[cosine_similarities_df['sentiment'] == sentiment]
        cs_df = cs_df.pivot(index='date', columns='ticker', values='value')

        try:
            clean_data = al.utils.get_clean_factor_and_forward_returns(factor=cs_df.stack(), prices=pricing, periods=[1], quantiles=5)
            factor_data[sentiment] = clean_data
        except:
            skipped_sentiments.append(sentiment)

    if skipped_sentiments:
        print('\nSkipped the following sentiments:\n{}'.format('\n'.join(skipped_sentiments)))

    # Alphalens Format with Unix Time
    unixt_factor_data = {}
    for sentiment, clean_factor_data in factor_data.items():
        unixt_factor_data[sentiment] = clean_factor_data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in clean_factor_data.index.values],
            names=['date', 'asset']
        ))

    # factor returns
    ls_factor_returns = pd.DataFrame()
    for factor, clean_factor_data in factor_data.items():
        ls_factor_returns[factor] = al.performance.factor_returns(clean_factor_data).iloc[:, 0]

    (1 + ls_factor_returns).cumprod().plot()

    # Basis Points Per Day per Quantile
    qr_factor_returns = pd.DataFrame()
    for factor, clean_factor_data in unixt_factor_data.items():
        qr_factor_returns[factor] = al.performance.mean_return_by_quantile(clean_factor_data)[0].iloc[:, 0]

    (10000*qr_factor_returns).plot.bar(
        subplots=True,
        sharey=True,
        layout=(5,3),
        figsize=(14, 14),
        legend=False);

    ls_FRA = pd.DataFrame()
    for factor, clean_factor_data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(clean_factor_data)

    ls_FRA.plot(title="Factor Rank Autocorrelation")

    #sharpe ratio
    sharpe_ratio_df = sharpe_ratio(qr_factor_returns, np.sqrt(252)).round(2)
    print(sharpe_ratio_df)