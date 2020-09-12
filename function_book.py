import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import baostock as bs
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
import collections

TMT_stocks_name = ["深科技","中兴通讯","中国长城","TCL科技","国新健康","汇源通信","振华科技","华闻集团","富通鑫茂","电广传媒","紫光股份","软控股份","国脉科技","三维通信","北纬科技","北斗星通","广电运通","粤传媒","二三四五","网宿科技","华谊兄弟","蓝色光标","天喻信息","腾信股份","海信视像","中视传媒","方正科技","综艺股份","四川长虹","博瑞传播","出版传媒"]
#shift_size int The number of days between each rebalance
#setp1
def generate_stock_price(stock_basic_df,start_date_str,end_date_str,shift_size,columns_str = "date,code,close,pbMRQ,isST0",saved = False):

    lg = bs.login()

    result = pd.DataFrame()

    for code in stock_basic_df['code']:
        #print(code,end = ',')
        rs = bs.query_history_k_data_plus(code,columns_str,start_date_str,end_date_str,frequency="d",adjustflag="3")

        temp = rs.get_data()

        for i in range(shift_size,len(temp),shift_size):

            temp.loc[[i-1],'ave_close'] = temp['close'][i-shift_size:i].astype('float').mean()
            temp.loc[[i-1],'ave_pbMRQ'] = temp['pbMRQ'][i-shift_size:i].astype('float').mean()

        temp.dropna(inplace = True)
        result = pd.concat([result,temp],ignore_index=True)
    bs.logout()

    result = pd.merge(result,stock_basic_df[['code','code_name']],on='code')
    if saved is True:
        result.to_csv("TMT_stock_price.csv",index = False,encoding='utf_8_sig')
    else:
        result.reset_index(drop =True)
        return result



#setp1.5
def get_quarter_date_list(start_date_str,end_date_str):
    start_date = datetime.datetime.strptime(start_date_str ,"%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date_str ,"%Y-%m-%d")

    temp = []
    for y in np.arange(start_date.year,end_date.year+1):
        temp = temp + [str(y)+"-01-01",str(y)+"-04-01",str(y)+"-07-01",str(y)+"-10-01"]

    for i in range(1,len(temp)):
        if temp[i-1]< start_date_str and temp[i] > start_date_str:
            break
    temp = temp[i-1:]

    for i in range(len(temp),1,-1):
        if temp[i-1] < end_date_str:
            break
    return temp[:i]

#quarter_date_list = get_quarter_date_list(start_date_str,end_date_str)

#step2
def generate_stock_details(stock_price_df,quarter_date_list,saved=False):
    code_list = stock_price_df['code'].unique()
    result = pd.DataFrame()
    lg = bs.login()

    for i in range(len(quarter_date_list)-1):
        print(quarter_date_list[i],end = ',')
        left_date = datetime.datetime.strptime(quarter_date_list[i],"%Y-%m-%d") - datetime.timedelta(1)
        left_date_str = datetime.datetime(left_date.year, left_date.month, left_date.day, 0, 0).strftime("%Y-%m-%d")
        right_date_str = quarter_date_list[i+1]

        profit_list = []
        for code in code_list:
            rs_profit = bs.query_profit_data(code=code, year=left_date.year, quarter=left_date.month//4+1)
            while (rs_profit.error_code == '0') & rs_profit.next():
                temp = rs_profit.get_row_data()
                profit_list.append(temp)

        profit_df = pd.DataFrame(profit_list, columns=rs_profit.fields)
        result = pd.concat([result,pd.merge(stock_price_df[(stock_price_df['date']>left_date_str)&(stock_price_df['date']< right_date_str)],profit_df[['code','totalShare']],on='code')])

    result[['totalShare']] = result[['totalShare']].astype('float')
    result['marketCapitalization']  = result['totalShare']*result['ave_close']
    bs.logout()
    if saved is True:
        result.to_csv("TMT_stock_details.csv",index = False,encoding='utf_8_sig')
    else:
        result.reset_index(drop =True)
        return result

'''
stock_price_df = pd.read_csv("TMT_stock_price.csv")
quarter_date_list = ["2019-04-01","2019-07-01","2019-10-01","2020-01-01","2020-04-01","2020-07-01"]
'''
#step3
def get_famafrench_stocks(unsorted_df,prof_size = None,freq = False, short = False ):

    result = pd.DataFrame()
    short_result = pd.DataFrame()

    selected_list= []
    sselected_list= []
    pre_holding = []
    spre_holding = []

    sorted_HML_factor ='pbMRQ'

    sorted_SMB_factor = 'close'


    if 'ave_pbMRQ' in unsorted_df.columns:
        sorted_HML_factor = 'ave_pbMRQ'

    if 'ave_close' in unsorted_df.columns:
        sorted_SMB_factor = 'ave_close'


    for date in unsorted_df['date'].unique():
        sorted_df = unsorted_df[unsorted_df['date']==date].copy()
        sorted_df.sort_values(sorted_HML_factor, inplace=True)
        sorted_df['HML'] = np.arange(1,1+len(sorted_df),1)
        sorted_df['marketCapitalization']  =sorted_df['totalShare']*sorted_df[sorted_SMB_factor]
        sorted_df.sort_values('marketCapitalization', inplace=True)
        sorted_df['SMB'] = np.arange(1,1+len(sorted_df),1)
        sorted_df['ratio']=sorted_df['marketCapitalization']/sorted_df['marketCapitalization'].sum()+sorted_df['ave_pbMRQ']/sorted_df['ave_pbMRQ'].sum()
        sorted_df.sort_values('ratio', inplace=True)
        sorted_long_df = sorted_df.loc[(sorted_df['SMB']<(len(sorted_df)+1)/2)&(sorted_df['HML']<(len(sorted_df)+1)/3)]

        if prof_size is not None:
            cur_holding = sorted_long_df['code'][:prof_size].tolist()
        else:
            cur_holding = sorted_long_df['code'].tolist()
        if freq is False:
            if (collections.Counter(pre_holding)!=collections.Counter(cur_holding)):
                pre_holding = cur_holding.copy()
                cur_holding.append(date)
                selected_list.append(cur_holding)
        else:
            cur_holding.append(date)
            selected_list.append(cur_holding)
        result = pd.concat([result,sorted_long_df])

        if short:
            sorted_short_df = sorted_df.loc[(sorted_df['SMB']>(len(sorted_df)+1)/2)&(sorted_df['HML']>(len(sorted_df)+1)/3*2)]
            if prof_size is not None:
                cur_holding = sorted_short_df['code'][-prof_size:].tolist()
            else:
                cur_holding = sorted_short_df['code'].tolist()
            if freq is False:
                if (collections.Counter(spre_holding)!=collections.Counter(cur_holding)):
                    spre_holding = cur_holding.copy()
                    cur_holding.append(date)
                    sselected_list.append(cur_holding)
            else:
                cur_holding.append(date)
                sselected_list.append(cur_holding)
            short_result = pd.concat([short_result,sorted_short_df])
    result.to_csv("TMT_stock_results.csv",index = False,encoding='utf_8_sig')
    if short:
        short_result.to_csv("TMT_stock_short_results.csv",index = False,encoding='utf_8_sig')
    return (selected_list,sselected_list)
#unsorted_df = pd.read_csv("TMT_stock_details.csv",encoding='utf_8')
#get_famafrench_stocks(unsorted_df)

#step4
#chunk_size : int The number of year to look in the past for rebalancing
def processing_selected_list(result,chunk_size = 2):
    today = datetime.datetime.strptime(result[0][-1],"%Y-%m-%d")
    start_opt_date = datetime.datetime(today.year-chunk_size, today.month, today.day, 0, 0).strftime("%Y-%m-%d")
    end_opt_date= result[-1][-1]

    stocks_id = []
    for l in result:
        stocks_id +=  l[:-1]
    return (stocks_id,start_opt_date,end_opt_date)
#(stocks_id,start_opt_date,end_opt_date) = processing_selected_list(result)

#step5
def update_history_df(stocks_id,start_opt_date,end_opt_date):

    def loading_prices(stocks_id_str,start_date,end_date):
        k_data_df = pd.DataFrame()
        rs = bs.query_history_k_data_plus(stocks_id_str,"date,close",start_date=start_date, end_date=end_date, frequency="d", adjustflag="3")
        k_data_df = k_data_df.append(rs.get_data())
        k_data_df.set_index(keys = 'date',inplace=True)
        k_data_df['close'] = k_data_df['close'].astype('float')
        k_data_df.rename(columns={"close": stocks_id_str},inplace=True)
        return k_data_df


    try:
        print("try")
        history_df = pd.read_csv("history_TMT.csv")
        history_df.set_index(keys = 'date',inplace=True)
        start_date = min(start_opt_date,history_df.index[0])
        end_date = max(history_df.index[-1],end_opt_date)

        if start_opt_date < history_df.index[0] or history_df.index[-1] < end_opt_date:
            history_df = pd.DataFrame()
            print("try if")

    except:
        print("except")
        history_df = pd.DataFrame()
        start_date = start_opt_date
        end_date = end_opt_date

    print(start_date,end_date)

    lg = bs.login()

    for stocks_id_str in stocks_id:
        if stocks_id_str not in history_df.columns:
            print(stocks_id_str,end = ',')
            k_data_df = loading_prices(stocks_id_str,start_date,end_date)
            history_df = pd.concat([history_df,k_data_df],axis=1)

    bs.logout()
    history_df.to_csv("history_TMT.csv",index_label = 'date')
#update_history_df(stocks_id,start_opt_date,end_opt_date)
#update_history_df(default_stocks_id,start_opt_date,end_opt_date)

#step6
def all_rebalance_weights(result,default_id = ['sh.600845','sz.000651','sh.601318'],chunk_size = 2,method='SLSQP'):

    def statistics(weights):
        weights = np.array(weights)
        risk_free_rate = 0.04
        pret = np.sum(rets.mean() * weights) * 252 #- risk_free_rate
        pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
        return np.array([pret, pvol, pret / pvol])


    def min_func_sharpe(weights):
        return -statistics(weights)[2]

    def min_variance(weights):
        return statistics(weights)[1]

    profolio = []
    for idx, l in enumerate(result):

        today = datetime.datetime.strptime(l[-1],"%Y-%m-%d")
        start_date_str = datetime.datetime(today.year-chunk_size, today.month, today.day, 0, 0).strftime("%Y-%m-%d")
        end_date_str = datetime.datetime(today.year, today.month, today.day, 0, 0).strftime("%Y-%m-%d")

        data = pd.read_csv('history_TMT.csv', usecols=['date']+default_id+l[:-1])
        data.set_index(keys = 'date',inplace=True)

        if idx+1 == len(result):
            next_trading_date_str = "2020-04-30"
        else:
            next_trading_date_str = result[idx+1][-1]

        next_price = data.loc[next_trading_date_str].values
        data = data[start_date_str:end_date_str]
        #print(end_date_str,next_trading_date_str)


        log_returns = np.log(data / data.shift(1))
        rets = log_returns
        year_ret = rets.mean() * 252
        year_volatility = rets.cov() * 252

        number_of_assets = len(default_id)+len(l)-1
        bnds = tuple((0, 1) for x in range(number_of_assets))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        optv = sco.minimize(min_variance, number_of_assets * [1. / number_of_assets,],method = 'SLSQP', bounds = bnds, constraints = cons)
        #print("得到的预期收益率、波动率和夏普指数",statistics(optv['x']).round(3))
        if len(profolio) == 0:
            capital_base = 1000000
        else:
            capital_base = sum(profolio[-1]['holdings'] * profolio[-1]["next_prices"])

        temp = {"weights":optv['x'],
                 "id":default_id+l[:-1],
                 "date":end_date_str,
                 "next_date":next_trading_date_str,
                 "next_prices":next_price,
                 "prices":data.iloc[-1].tolist(),
                 "info":statistics(optv['x']).round(3),
                 "capital":capital_base}

        temp['holdings'] = capital_base*temp['weights']/temp['prices']

        profolio.append(temp)
    return profolio
#profolio = all_rebalance_weights(result,default_id = ['sh.600845','sz.000651','sh.601318'],chunk_size = 2,method='SLSQP')

def profolio_ana(profolio,prof_size):
    lg = bs.login()
    result = pd.DataFrame()
    col = ['date']
    for i in range(0,prof_size ):
        col.append('capital'+str(i))
    #print(col)
    for d in profolio:
        _df = pd.DataFrame()
        for i in range(0,prof_size):
            #print(i,end = ',')
            rs = bs.query_history_k_data_plus(d['id'][i],"date,code,close",start_date=d['date'], end_date=d['next_date'], frequency="d", adjustflag="3")
            temp = rs.get_data()
            temp['capital'+str(i)]  = temp['close'].astype(float)*  d['holdings'][i]
            temp.drop(temp.tail(1).index,inplace=True)
            if i == 0:
                _df = pd.concat([_df,temp[['date','capital'+str(i)]]],axis=1)
            else:
                _df = pd.concat([_df,temp[['capital'+str(i)]]],axis=1)
        result = pd.concat([result,_df[col]],axis=0)
    bs.logout()
    result.set_index(keys = 'date',inplace=True)
    return result

def max_withdrawal_rate(a):
    high = a[0]
    low = a[0]
    delta = 0
    for i in a:
        if not i < high: #i >= high
            delta = max(delta,(high - low)/high)
            high = i
            low = high
        elif i < low:
            low = i
    delta = max(delta,(high - low)/high)
    return delta*100

'''
def get_weights(stocks_id,opts_weight):

    data = pd.read_csv('history_TMT.csv', usecols=stocks_id.tolist()+['date','sh.600845','sz.000651','sh.601318'])
    data.set_index(keys ='date',inplace=True)

    log_returns = np.log(data / data.shift(1))
    rets = log_returns
    #year_ret = rets.mean() * 252
    #year_volatility = rets.cov() * 252

    number_of_assets = len(stocks_id)+3

    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_weights = []

    for p in range (5000):
        weights = np.random.random(number_of_assets)
        weights /= np.sum(weights)
        portfolio_weights.append(weights)
        portfolio_returns.append(np.sum(rets.mean() * weights) * 252)
        portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))

    portfolio_weights.append(opts_weight)
    portfolio_returns.append(np.sum(rets.mean() * weights) * 252)
    portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))

    #print(len(portfolio_weights),len(portfolio_returns),len(portfolio_volatilities))
    portfolio_returns=np.array(portfolio_returns)
    portfolio_volatilities =np.array(portfolio_volatilities)


    plt.figure(figsize=(9, 5)) #作图大小
    plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns / portfolio_volatilities, marker='o')
    plt.scatter(portfolio_volatilities[-1], portfolio_returns[-1], c = 'r',marker='x',)
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')


def print_my_df(_df,details=False):
    for row in range(len(_df)):
        if details:
            for col in ['code_name','close','weight', 'stocks','pbMRQ']:
                print(col,":",_df[col][row],end = ', ')
        else:
            for col in ['code_name','close','weight', 'stocks']:
                print(col,":",_df[col][row],end = ', ')
        print()

def date_range(start, end, step=1, format="%Y-%m-%d"):
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days + 1
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days, step)]


def validate_date(test_date):
    lg = bs.login()
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    rs = bs.query_trade_dates(start_date=test_date.strftime("%Y-%m-%d"), end_date=test_date.strftime("%Y-%m-%d"))
    print('query_trade_dates respond error_code:'+rs.error_code)
    print('query_trade_dates respond  error_msg:'+rs.error_msg)

    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    print(bool(result['is_trading_day'][0]))

#tool1
def get_stock_code_by_name(stock_name_list,isreturn = False):

    lg = bs.login()
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    data_list = []

    for stock_name_str in stock_name_list:

        rs = bs.query_stock_basic(code_name=stock_name_str)  # 支持模糊查询

        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    bs.logout()

    if isreturn == True:
        return result

    print("! save tmt_stock csv file.")
    result.loc[(result['status']=='1')][['code', 'code_name']].to_csv("tmt_stock.csv", encoding='utf_8_sig')

    #return result
'''
