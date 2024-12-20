import numpy as np
import pandas as pd
from company_info import company_info_list
import scipy as scp
import yfinance as yf
from dtaidistance import dtw

from tqdm.auto import tqdm

#Weight everything by its own relevance score
weighted_relevance_dict = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}


def get_price_data(company, start_date, end_date):
    '''
    Get the price data from yfinance for a given company
    '''

    ticker = company_info_list[company]['ticker']

    dat = yf.Ticker(ticker)
    # print(f'Long name for company {company}: {dat.info["longName"]}')

    df_yf = dat.history(period='5y')
    df_yf = df_yf.loc[start_date:end_date]
    df_yf.index = df_yf.index.normalize().tz_localize(None)

    df = pd.read_csv('../data/date_dt.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df['Price'] = df_yf['Close']
    df['Price'] = df['Price'].ffill()

    return df

def apply_dtw(timeseries1, timeseries2, dtw_window):

    #Do z-normalization
    s1 = scp.stats.zscore(timeseries1)
    s2 = scp.stats.zscore(timeseries2)

    d, paths = dtw.warping_paths_fast(s1, s2, window=dtw_window, psi=dtw_window, inner_dist='euclidean')
    best_path = dtw.best_path(paths)

    return s1, s2, d, best_path

def filter_data(data, start_date, end_date, company, aspect_filter, relevance_cutoff, weighted_relevance):
    '''
    Filters and processes ESG sentiment data for a specific company within a given date range.
    Parameters:
        data (pd.DataFrame): The input data containing ESG sentiment information.
        start_date (str): The start date for the filtering in 'YYYY-MM-DD' format.
        end_date (str): The end date for the filtering in 'YYYY-MM-DD' format.
        company (str): The company name to filter the data.
        aspect_filter (str): The aspect to filter by ('env_soc', 'env_gov', 'soc_gov', or specific aspect).
        relevance_cutoff (float): The minimum relevance score to filter the data.
        weighted_relevance (bool): Whether to weight sentiment by relevance score.
    Returns:
        pd.DataFrame: A DataFrame with daily sentiment scores from start_date to end_date.
    
    '''
    if company:
        data_comp = data[data['company'] == company]
    else:
        data_comp = data.copy()
    if relevance_cutoff:
        data_comp = data_comp[data_comp['relevance_score'] >= relevance_cutoff]
    if aspect_filter:
        if aspect_filter == 'env_soc':
            data_comp = data_comp[data_comp['aspect'] != 'governance']
        elif aspect_filter == 'env_gov':
            data_comp = data_comp[data_comp['aspect'] != 'social']
        elif aspect_filter == 'soc_gov':
            data_comp = data_comp[data_comp['aspect'] != 'environmental']
        else:
            data_comp = data_comp[data_comp['aspect'] == aspect_filter]
    
    data_comp = data_comp.copy()
    data_comp['relevance_weight'] = data_comp['relevance_score'].apply(lambda x: weighted_relevance_dict[x])
    if weighted_relevance:
        #We weight each article depending on its relevance score as defined in the weighted_relevance_dict
        data_comp['sentiment_int'] = data_comp['sentiment_int'] * data_comp['relevance_weight']

    data_comp = data_comp.sort_values(by='date')
    esg_sentiment = data_comp[['date', 'sentiment_int', 'relevance_score']]
    if len(esg_sentiment) == 0:
        #Return empty dataframe from start to end date
        daily_data = pd.DataFrame(index=pd.date_range(start_date, end_date, freq='D'))
        daily_data['sentiment_int'] = [0] * len(daily_data)
        return daily_data

    daily_sum = data_comp[['date', 'volume', 'relevance_weight']].resample('d', on='date').max()
    daily_data = esg_sentiment.resample('d', on='date').mean()
    if weighted_relevance:
        #This is needed to scale back the sentiment to the original scale
        daily_data['sentiment_int'] = daily_data['sentiment_int'] / daily_sum['relevance_weight']
    
    if np.max(daily_data['sentiment_int']) > 1 or np.min(daily_data['sentiment_int']) < -1:
        raise Exception('Sentiment is not in the expected range!')

    #fill edges with 0
    daily_data = daily_data.reindex(pd.date_range(start_date, end_date, freq='D'))
    
    #Can fill with 0s or with previous value
    daily_data.fillna(0, inplace=True)

    return daily_data

def get_daily_estimator(dtw_window, data_window, kwargs):
    '''
    Calculate daily return estimators for companies based on Dynamic Time Warping (DTW) between price and ESG sentiment data.
    Parameters:
        dtw_window (int): The window size for DTW calculation.
        data_window (int): The window size for data slicing. If None, use all available data.
        kwargs (dict): A dictionary containing the following keys:
            - company_date_dt_dict (dict): Dictionary with company dates.
            - company_price_dict (dict): Dictionary with company prices.
            - company_aspect_esg_sent_dict (dict): Dictionary with company ESG sentiment data.
            - company_aspect_daily_data_dict (dict): Dictionary with company daily data.
            - aspect_filters (list): List of aspect filters to consider.
            - use_lag_to_invest (bool): Flag to determine if lag should be used for investment decisions.
    Returns:
        dict: A dictionary with companies as keys and their respective daily return estimators as values.
    '''

    #Unpack kwargs
    company_date_dt_dict = kwargs['company_date_dt_dict']
    company_price_dict = kwargs['company_price_dict']
    company_aspect_esg_sent_dict = kwargs['company_aspect_esg_sent_dict']
    company_aspect_daily_data_dict = kwargs['company_aspect_daily_data_dict']
    aspect_filters = kwargs['aspect_filters']
    use_lag_to_invest = kwargs['use_lag_to_invest']


    company_estimator_dict = {}         #here, we always just append the currently decided estimator
    for company in tqdm(company_info_list.keys(), desc='Calculating daily return estimators...'):
        company_estimator_dict[company] = [0, 0]
        #Iterate over days, determine best aspect and set the return estimator
        for i in range(2, len(company_date_dt_dict[company])):
            aspect_distance_dict = {}
            no_trade_dict = {}
            dtw_dict = {}
            for aspect_filter in aspect_filters:
                dtw_dict[aspect_filter] = {}    #temp
                no_trade_dict[aspect_filter] = False
                #retrieve data from dicts - including that day
                if data_window:
                    price = company_price_dict[company][np.max([0, i-data_window+1]):i+1]
                    esg_sentiment = company_aspect_esg_sent_dict[company][aspect_filter][np.max([0, i-data_window+1]):i+1]
                else:
                    price = company_price_dict[company][:i+1]
                    esg_sentiment = company_aspect_esg_sent_dict[company][aspect_filter][:i+1]
                #Normalize s1, s2, get sentiment gradient and calculate best path
                _, _, d, best_path = apply_dtw(price, esg_sentiment, int(np.min([len(esg_sentiment), dtw_window])))

                #Calculate lag - set distance to 1000 if it is negative
                lag = [l[0] - l[1] for l in best_path]
                if lag:
                    if lag[-1] <= 0 and use_lag_to_invest:
                        #price is leading esg-sentiment -> we cannot use this to predict future returns
                        no_trade_dict[aspect_filter] = True     #this is not enough because there could still be an aspect which has pos lag but slightly
                                                                #worse distance which would not be picked without manipulating the distance
                        d = 1000
                aspect_distance_dict[aspect_filter] = d
                
            
            #Get aspect with smallest distance
            best_aspect = min(aspect_distance_dict, key=aspect_distance_dict.get)

            #If the smallest distance is still too big, set the estimator to be neutral
            if aspect_distance_dict[best_aspect] >= 1000 or no_trade_dict[best_aspect]:# (450/260*data_len):
                company_estimator_dict[company].append(0)
            else:
                company_estimator_dict[company].append(company_aspect_daily_data_dict[company][best_aspect].loc[company_date_dt_dict[company][i]].values[0])
    
    return company_estimator_dict