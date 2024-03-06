#librerie generali per la gestione dei dati
import pandas as pd
import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt
#librerie per la gestione delle date
import datetime as dt
from pandas.tseries.offsets import BDay
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import  Dense, Dropout
from tensorflow.keras.models import  Sequential

import seaborn as sns
import math
import scipy.optimize as sco
import warnings
warnings.filterwarnings('ignore')

columns_bank = ['Goldman Sachs', 'Bank of America', 'Wells Fargo']
columns_auto = ['General Motors', 'Ford Motors', 'Tesla']
columns_tech = ['Qualcom', 'BroadCom', 'Intel']
columns = {'Settore bancario' : columns_bank, 'Settore automobilistico' : columns_auto, 'Settore wireless e semiconduttori ': columns_tech}

columns_bank__ = ['Goldman_Sachs', 'Bank_of_America', 'Wells_Fargo']
columns_auto__ = ['General_Motors', 'Ford_Motors', 'Tesla']
columns_tech__ = ['Qualcom', 'BroadCom', 'Intel']
columns__ = {'Settore bancario' : columns_bank__, 'Settore automobilistico' : columns_auto__, 'Settore wireless e semiconduttori ': columns_tech__}



color_list = ['#00A86B', '#5B9EE3', '#0F52BA', '#513BA5', '#D370FA', '#F58195', '#E3775B', '#800020', '#EE9B00']
color_light_list= ['#5CFFC3', '#98C3E6', '#007BB8', '#7C69C9', '#E1AFFB', '#F5BECB', '#E6A89A', '#D9544D', '#FFBF47']

tickers = ('GS', 'BAC', 'WFC', 'GM', 'F', 'TSLA', 'QCOM', 'AVGO', 'INTC')


start_date = '2011-11-30'
end_date = '2021-11-30'

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = True

#'''''''''''''''''''''' Mie funzioni ''''''''''''''''''''''''
def plt_color_selection(color_index):
	if color_index == 0:
		return 'tab:blue'
	elif color_index == 1:
		return 'tab:green'
	else:
		return 'tab:red'

def show_plot_property(dataframe, property_title):
	for sector in columns:

		plt.subplot(2,1,1)
		plt.plot(dataframe[columns[sector]])
		plt.title(property_title + " " + sector)
		plt.legend(columns[sector])
		plt.xlim(dt.datetime.strptime(start_date, '%Y-%m-%d'), dt.datetime.strptime(end_date, '%Y-%m-%d'))

		for elem in columns[sector]:

			i = int(columns[sector].index(elem))
			plt.subplot(2,3,(4+i))

			plt.plot(dataframe[elem], color=plt_color_selection(i))
			plt.title(elem)

		plt.tight_layout()
		plt.show()


def diagnostics_Diagram(dataframe, stock_name, color='tab:blue', dark_color = 'blue'):

	plt.suptitle(stock_name, fontsize=18, fontweight='bold')

	plt.subplot(2, 2, 1)
	plt.hist(dataframe[stock_name], density = True, color=color)
	plt.title('Istogramma ritorni ')

	plt.subplot(2, 2, 2)
	dataframe[stock_name].plot.density(color = color)
	plt.title('Densità ritorni ')

	axsis = plt.subplot(2, 2, 3)
	fig = sm.qqplot(dataframe[stock_name], line='s', ax=axsis)

	plt.title('QQPlot ritorni ')


	plt.subplot(2, 2, 4)
	boxplot = plt.boxplot(dataframe[stock_name], patch_artist=True)
	setboxplot(boxplot, dark_color, color)
	plt.title('Boxplot ritorni ')

	plt.tight_layout()
	plt.show()

def setboxplot(boxplot, color, light_color, index = 0, index2 = 0):
	boxplot['medians'][index].set_color(color)
	boxplot['boxes'][index].set(facecolor = light_color, edgecolor=color)
	boxplot['fliers'][index].set(markeredgecolor = color, marker = 'D', alpha=0.7)

	boxplot['whiskers'][index2].set_color(color)
	boxplot['caps'][index2].set_color(color)

	boxplot['caps'][index2+1].set_color(color)
	boxplot['whiskers'][index2+1].set_color(color)


def custom_boxplot(dataframe, title=""):

	boxplot = plt.boxplot(dataframe, vert=False, patch_artist=True, labels=(columns_bank + columns_auto + columns_tech))

	j = 0
	for i in range(len(boxplot['boxes'])):
		setboxplot(boxplot, color_list[i], color_light_list[i], i, j)
		j = j+2

	plt.title(title)
	plt.show()

def add_Data_df(dataframe, ts, nday, tf = 'day', price = True):

	start_d = dataframe.index[0]
	before_start_d = start_d - BDay(nday)

	tmp = web.get_data_yahoo(ts, before_start_d, start_d ) ['Adj Close']
	tmp.columns = dataframe.columns

	if tf == 'month':
		tmp = tmp.resample('M').last().dropna()

	if not(price):
		tmp = tmp.pct_change().dropna()

	dataframe = pd.concat([tmp, dataframe])
	dataframe = dataframe[~dataframe.index.duplicated(keep='first')]
	return dataframe

#''''''''''''''''''''''' fine mie funzioni''''''''''''''''''''''''''''''''''''



#  versione con daily base
# df_stock_full = web.get_data_yahoo(tickers, start_date, end_date) ['Adj Close']
# df_stock_full.columns = ['Goldman Sachs', 'Bank of America', 'Wells Fargo', 'General Motors', 'Ford Motors', 'Tesla', 'Qualcom', 'BroadCom', 'Intel']
# df_stock_full.head()
# df_stock_full_montly = df_stock_full.resample('M').last().dropna()
# df_stock_full.head()

#nota: ricordarsi di sistemare i dati degli altri dataset (SP500, Fama-French)


#versione con montly base
df_stock_full_daily = web.get_data_yahoo(tickers, start_date, end_date) ['Adj Close']
df_stock_full_daily.columns = ['Goldman Sachs', 'Bank of America', 'Wells Fargo', 'General Motors', 'Ford Motors', 'Tesla', 'Qualcom', 'BroadCom', 'Intel']
df_stock_full_daily.head()



df_stock_full = df_stock_full_daily.resample('M').last().dropna()
df_stock_full.head()

show_plot_property(df_stock_full, 'Chiusura') #print delle chiusure per settore

#Parte 2 ----------------------------------------------------------------------------------------------------------

#####   Punto A


#calcolo rendimenti semplici e composti
simple_return_full = df_stock_full.pct_change().dropna()
simple_return_full_daily = df_stock_full_daily.pct_change().dropna()

log_return_full = np.log(df_stock_full/df_stock_full.shift(1)).dropna()
log_return_full_daily = np.log(df_stock_full_daily/df_stock_full_daily.shift(1)).dropna()

show_plot_property(simple_return_full, 'Simple return') #print dei simple return

compound_return_full = (simple_return_full+1).cumprod()-1

show_plot_property(compound_return_full, 'Compound return')

show_plot_property(log_return_full, 'Log return')


#punto C
custom_boxplot(simple_return_full, title='Distribuzione dei ritorni a confronto')

###### punto D
for sector in columns:
	for elem in columns[sector]:
		color = plt_color_selection(int(columns[sector].index(elem)))
		diagnostics_Diagram(simple_return_full, elem, color = color, dark_color =color.split(':')[1] )



#### Punto E

df_stats = pd.DataFrame(index = simple_return_full.columns)
df_stats['Media'] = simple_return_full.mean()
df_stats['Varianza'] = simple_return_full.var()
df_stats['Deviazione standard'] = simple_return_full.std()
df_stats['Asimmetria'] = simple_return_full.skew()
df_stats['Curtosi'] = simple_return_full.kurtosis()

df_stats

### Punto F
matrice_var = simple_return_full.var()
matrice_cov = simple_return_full.cov()

print('matrice varianza:\n', matrice_var)
print('matrice covarianza:\n', matrice_cov)

### Punto G

matrice_corr = simple_return_full.corr().applymap(lambda x: float('NaN') if x == 1 else x)

print('Matrice della correlazione tra tutte le stock')
simple_return_full.corr().style.background_gradient(cmap='coolwarm')

plt.subplot(2, 1, 1)
plt.plot(matrice_corr.mean(), 'D', markersize=10, color='red')
plt.title("Correlazione medie delle stock")

i = 0
for x, y in columns.items():
	plt.subplot(2, 3, 4+i)
	plt.title("Nel " + x)
	plt.plot(simple_return_full.corr().applymap(lambda x: float('NaN') if x == 1 else x) [y].mean(), 'bD', markersize=10, color=plt_color_selection(i))
	i = i+1

plt.tight_layout()
plt.show()


###Punto H

pairp = sns.pairplot(simple_return_full)
pairp.fig.suptitle("Scatter delle correlazioni fra le stock", y=1)


#correlazione media nel tempo di ogni stock
offset_date_correlation_num = 14 #month
df_progressive_correlation = pd.DataFrame()

for i, cont in simple_return_full.iterrows():
	new_row = simple_return_full.loc[:i] [:].corr().applymap(lambda x: float('NaN') if x == 1 else x).mean()
	new_row.name = i
	df_progressive_correlation = df_progressive_correlation.append(new_row)

df_progressive_correlation = df_progressive_correlation[offset_date_correlation_num:]#taglio le prime perchè la correlazione non è precisa per mancanza di dati precedenti

df_progressive_correlation = df_progressive_correlation[columns_bank + columns_auto + columns_tech]
df_progressive_correlation

show_plot_property(df_progressive_correlation, 'Correlazione media nel tempo')



#--------------------------PUNTO 3
#---funzioni usate per il forecast

#prepara i dati X e y ed effettua la divisione tra i periodi di validazione e di train-test
def preprocessing_preparation_validation_split(dataframe, nsplit_day, prediction_days, offset_day = 1):
	X_train_test = pd.Series(index = dataframe.columns, dtype=object)
	X_validation = pd.Series(index = dataframe.columns, dtype=object)

	offset_day = offset_day -1 #perchè già c'è un offsets minimo di 1

	for column in dataframe.columns:

		tmp_list = []
		for i in range(prediction_days, len( dataframe.index)):
			new_row = dataframe[column][i-prediction_days : i].values.tolist()
			new_row.reverse()
			tmp_list.append(new_row)

		new_col = pd.DataFrame(data = tmp_list, index= dataframe.index[prediction_days:], columns = [column + '_prec' + str(j) for j in range(1, prediction_days+1)])
		new_col = new_col.shift(offset_day)

		X_train_test[column] = new_col[offset_day:-nsplit_day]#non parto da [prediction_days:] perchè è gia cantato in new col
		X_validation[column] = new_col[-nsplit_day:]

	y_train_test = dataframe.iloc[prediction_days + offset_day:-nsplit_day]
	y_validation = dataframe.iloc[-nsplit_day:]

	return X_validation, y_validation, X_train_test, y_train_test


#funzione che effettua il train test split a seconda del formato dei dati, restituisce come x sempre una serie che contiene df per ogni colonna
def preprocessing_train_test_split(X_data, y_data, test_size):

	l_X_train = pd.Series(index=X_data.index, dtype=object, name='series X_train')
	l_X_test = pd.Series(index=X_data.index, dtype=object, name='series X_test')

	l_y_train = pd.Series(index=y_data.columns, dtype=object, name='series y_train')
	l_y_test = pd.Series(index=y_data.columns, dtype=object, name='series y_test')

	for column in X_data.index:
		X_train, X_test, y_train, y_test = train_test_split(X_data[column], y_data[column], test_size=test_size)

		l_X_train[column] = X_train
		l_X_test[column] = X_test
		l_y_train[column] = y_train
		l_y_test[column] = y_test
	return l_X_train, l_X_test, l_y_train, l_y_test

def preprocessing_X_reshape(X_data):
	return X_data.values.reshape((X_data.values.shape[0], X_data.values.shape[1]))

def train_model(X_train, y_train):

	X_train_array = preprocessing_X_reshape(X_train)
	model = Sequential()

	model.add(Dense(units = 100, input_shape=(X_train_array.shape[1],)))
	model.add(Dense(units = 100))
	model.add(Dropout(0.2))
	model.add(Dense(units = 50))
	model.add(Dropout(0.2))
	model.add(Dense(units = 50))
	model.add(Dropout(0.2))
	model.add(Dense(units = 20))
	model.add(Dropout(0.2))
	model.add(Dense(units = 1))

	model.compile(optimizer='adam', loss='mean_squared_error')
	model.fit(X_train, y_train, epochs=35, batch_size=32)

	return model




#--fine funzioni


train_month = 80
test_month = 30
val_month = 10
prediction_days = 42 #2 mesi lavorativi
test_size = test_month / (train_month + test_month)

offset_day = 3#1 mese lavorativo

forecast_df_input = add_Data_df(df_stock_full_daily, tickers, prediction_days+offset_day+1, price=True)

scalers = pd.Series(dtype=object, index=forecast_df_input.columns)
scaled_data = pd.DataFrame(columns = forecast_df_input.columns, index = forecast_df_input.index)

for column in scaled_data.columns:
	scalers[column] = MinMaxScaler(feature_range=(0,1))
	scaled_data[column] = scalers[column].fit_transform(forecast_df_input[column].values.reshape(-1, 1))

X_validation, y_validation, X_train_test, y_train_test = preprocessing_preparation_validation_split(scaled_data, int(val_month*21), prediction_days, offset_day = offset_day)


l_X_train, l_X_test, l_y_train, l_y_test = preprocessing_train_test_split(X_train_test, y_train_test, test_size)

forecast_models = pd.Series(dtype=object, index=l_X_train.index)

for column in forecast_models.index:

	forecast_model = train_model(l_X_train[column], l_y_train[column])
	print('model: ' , column, ' finished')
	forecast_models[column] = forecast_model

from keras.utils.vis_utils import plot_model
plot_model(forecast_models['Intel'], to_file='model_plot.png', show_shapes=True, show_layer_names=True)

for i in forecast_models.index: #testing

	prediction = scalers[i].inverse_transform(forecast_models[i].predict(preprocessing_X_reshape(l_X_test[i])))
	true_val = scalers[i].inverse_transform(l_y_test[i].values.reshape(-1, 1))

	print(i, '\nmean_squared_error: ')
	print(mean_squared_error(true_val, prediction))
	print('\nmean_absolute_error: ')
	print(mean_absolute_error(true_val, prediction))

	confronto = pd.DataFrame(index= l_y_test[i].index, columns=['prediction', 'true'])
	confronto['prediction'] = prediction
	confronto['true'] = true_val

	confronto.plot(title=i)

df_forecast_validation_set = pd.DataFrame(columns = y_validation.columns, index = y_validation.index)

for i in forecast_models.index: #validation

	prediction = scalers[i].inverse_transform(forecast_models[i].predict(preprocessing_X_reshape(X_validation[i])))
	true_val = scalers[i].inverse_transform(y_validation[i].values.reshape(-1, 1))

	print(i, '\nmean_squared_error: ')
	print(mean_squared_error(true_val, prediction))
	print('\nmean_absolute_error: ')
	print(mean_absolute_error(true_val, prediction))
	print('\n\n')

	confronto = pd.DataFrame(index= y_validation.index, columns=['prediction', 'true'])
	confronto['prediction'] = prediction
	confronto['true'] = true_val

	confronto = confronto.dropna()

	confronto.plot(title=i)

	df_forecast_validation_set[i] = prediction



#--------------------------Punto4
#compra tramite macd
ema1_period, ema2_period, signal_line_period,  = 12, 26, 9 #periodi per le medie mobili usate nel calcolo
lock_time_max = 14 #uso un lock_time per effettuare operazioni con frequenza massima di n giorni (per avere una strategia più realistica)

trading_data = pd.Series(index = df_stock_full_daily.columns, dtype=object)
df_trading_expanded = add_Data_df(df_stock_full_daily, tickers, ema2_period)#estraggo alcuni giorni precedenti per non perdere dati

for column in df_trading_expanded.columns: #calcolo macd per ogni stock
	tmp = pd.DataFrame(index = df_trading_expanded.index, dtype=object)

	tmp[column] =  df_trading_expanded[column]
	tmp[column + '_rtn'] = df_trading_expanded[column].pct_change()
	tmp['MACD'] = tmp[column].ewm(span=ema1_period, adjust = False).mean() - tmp[column].ewm(span=ema2_period, adjust = False).mean() #(vedere teoria per capire com'è il calcolo nello specifico)
	tmp['Signal line'] = tmp['MACD'].ewm(span=ema1_period, adjust = False).mean()

	trading_data[column] =  tmp[[column, column + '_rtn','MACD', 'Signal line']].dropna() [start_date :]

for column in trading_data.index:#applico la strategia
	last_op, new_op, lock_time = 0, 0, 0  #lock_time funziona come contatore da max a 0, viene resettato ogni volta che si fa un'operazione di acquisto o vendita
	for i in trading_data[column].index:
		if lock_time != 0:
			trading_data[column].at[i, 'invested'] = last_op
			lock_time = lock_time -1
		else:
			new_op = 1 if trading_data[column].at[i, 'MACD'] >= trading_data[column].at[i, 'Signal line'] else 0
			trading_data[column].at[i, 'invested'] = new_op

			if new_op != last_op:
				last_op = new_op
				lock_time = lock_time_max


strategy_rtn_comparison = pd.Series(index = trading_data.index, name='Compare between strategy and buy and hold', dtype=object)

for column in strategy_rtn_comparison.index:#calcolo i ritorni
	tmp = pd.DataFrame(index = trading_data[column].index, columns=['Buy and hold rtn', 'Strategy rtn'])

	tmp['Buy and hold rtn'] = np.cumprod(1+trading_data[column][column + '_rtn'])-1
	tmp['Strategy rtn'] = np.cumprod(1+trading_data[column].apply(lambda x: x[column+ '_rtn'] if x['invested'] == 1 else 0, axis=1))-1

	strategy_rtn_comparison[column] = tmp

for column in strategy_rtn_comparison.index:#faccio il plot delle performance
	strategy_rtn_comparison[column].plot(title='Performance della strategia rispetto al buy and hold di ' + column)




#--------------------------PUNTO 5
#punto A


 #versione con daily base
 #df_spx = web.get_data_yahoo('^GSPC', start_date, end_date) ['Adj Close']
 #df_spx.name = 'SP500'
 #df_spx_montly = df_spx.resample('M').last().dropna()


#versione con mensile base
df_spx_daily = web.get_data_yahoo('^GSPC', start_date, end_date) ['Adj Close']
df_spx_daily.name = 'SP500'
df_spx = df_spx_daily.resample('M').last().dropna()


simple_return_spx = df_spx.pct_change().dropna()
compound_return_spx = (simple_return_spx+1).cumprod()-1


beta = pd.Series(dtype='float64')

for i in simple_return_full:
		beta[i] = simple_return_spx.cov(simple_return_full[i]) / simple_return_spx.var()


plt.plot(beta, 'bD', markersize=10)
plt.title("I beta a confronto")
plt.show()


#beta nel tempo

offset_date_beta_num = 14 #month
df_progressive_beta = pd.DataFrame(columns = simple_return_full.columns, index = simple_return_full.index[14:])

for column in simple_return_full.columns:
	for i in df_progressive_beta[column].index:
		df_progressive_beta[column].loc[i] = simple_return_spx.loc[:i].cov(simple_return_full[column].loc[:i]) / simple_return_spx.loc[:i].var()

df_progressive_beta

show_plot_property(df_progressive_beta, 'Beta nel tempo')
#punto  #B

df_fama_french = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start = start_date)[0]

df_fama_french.columns = [ 'mkt', 'smb', 'hml', 'rf']
df_fama_french = df_fama_french.div(100)
df_fama_french = df_fama_french[start_date:end_date]

# in caso di daily base
#df_fama_french_montly = df_fama_french.resample('M').last().dropna()
#

#in caso di montly base:
df_fama_french_daily = df_fama_french
df_fama_french = df_fama_french_daily.resample('M').last().dropna() [1:]

simple_return_full_excess_rtn = simple_return_full.apply(lambda x:  x - df_fama_french['rf'])
simple_return_full_excess_rtn.columns = columns_bank__ + columns_auto__ + columns_tech__

ff_models = pd.Series(index = simple_return_full_excess_rtn.columns, dtype=object)

for column in simple_return_full_excess_rtn.columns:
	ff_models[column] = smf.ols(formula = column + ' ~ mkt + smb + hml', data = df_fama_french.join(simple_return_full_excess_rtn [column])).fit()

ff_models_out = []

for column in ff_models.index:
	ff_models_out.append(ff_models[column].params.values)

ff_models_out = pd.DataFrame(index = ff_models.index, columns=['Intercept', 'mkt', 'smb', 'hml'], data=ff_models_out)

#punto C

df_risk_free_T10 = web.DataReader('DGS10', 'fred', start = start_date).resample('M').last().dropna() [1:-2]
df_risk_free_T10

rf = df_risk_free_T10.loc[end_date].values [0] /100
rf

expected_return = rf + beta * (simple_return_spx.loc[end_date] - rf)

df_expected_returns = simple_return_spx.apply(lambda x: rf + beta* (x - rf))
df_expected_returns

scarto_expected_returns = (simple_return_full - df_expected_returns)

scarto_expected_returns.boxplot()
plt.title('Lo scarto tra i ritorni effettivi e quelli attesi')
plt.figure()
plt.show()



#------Punto 6
#alcune funzioni dichiarate per questo punto

def random_weighted_portfolios(df_portfolio_input, n_port):
	portfolio_avg_return = df_portfolio_input.mean() * 252 #si moltiplica per 252 per annualizzarlo
	cov_mat = df_portfolio_input.cov() * 252

	#generiamo portafogli con pesi casuali
	weights = np.random.random(size = (n_port, len(df_portfolio_input.columns)))
	weights /= np.sum(weights, axis = 1) [:, np.newaxis]
	portfs_rtn = np.dot(weights, portfolio_avg_return)

	portfs_vol = []
	for w in weights:
	    portfs_vol.append(np.sqrt(np.dot(w.T, np.dot(cov_mat, w))))

	df_portfs_param = pd.DataFrame({'returns': portfs_rtn, 'volatility': portfs_vol, 'sharpe_ratio': portfs_rtn/portfs_vol })
	return df_portfs_param, weights

def portfs_efficient_frontier(df_portfs_param, n_points):
	portf_ef_vol = []
	indices_to_skip = []
	portf_ef_rtn = np.round(np.linspace(df_portfs_param['returns'].min(), df_portfs_param['returns'].max(), n_points), 3)
	df_portfs_param = np.round(df_portfs_param, 3)

	#creo la linea di efficenza
	for i in range(n_points):
		if portf_ef_rtn[i] not in df_portfs_param['returns'].values:
			indices_to_skip.append(i)
		else:
			matched_i = np.where(df_portfs_param['returns'] == portf_ef_rtn[i])
			portf_ef_vol.append(np.min(df_portfs_param.volatility.values [matched_i]))

	portf_ef_rtn = np.delete(portf_ef_rtn, indices_to_skip)
	return portf_ef_rtn, portf_ef_vol

def get_portf_vol(w, rtn_avg, cov):
	return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

def neg_sharpe_ratio(w, avg_rtns, cov_mat, rf_rate):
    portf_returns = np.sum(avg_rtns * w)
    portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    portf_sharpe_ratio = (portf_returns - rf_rate) / portf_volatility
    return -portf_sharpe_ratio

def portfs_efficient_frontier_scipy(df_portfolio_input, range_frontier):
	cov_mat = df_portfolio_input.cov()*252
	portfolio_avg_return = df_portfolio_input.mean() * 252
	portf_efs = list()

	args = (portfolio_avg_return, cov_mat)
	bounds = tuple((0,1) for asset in range(len(df_portfolio_input.columns)))
	initial_guess = [np.min(rtns_range)] * len(df_portfolio_input.columns)

	for i in range_frontier:
		constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x * portfolio_avg_return) - i}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
		tmp = sco.minimize(get_portf_vol, initial_guess, args=args, method='SLSQP', constraints=constraints, bounds=bounds)

		portf_efs.append(tmp)
	return portf_efs

def portfs_max_sharp_ratio_scipy(df_portfolio_input):
	cov_mat = df_portfolio_input.cov()*252
	portfolio_avg_return = df_portfolio_input.mean() * 252

	RF_RATE = 0
	args = (portfolio_avg_return, cov_mat, RF_RATE)
	bounds = tuple((0,1) for asset in range(len(df_portfolio_input.columns)))
	initial_guess = len(df_portfolio_input.columns) * [1. /len(df_portfolio_input.columns), ]
	constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

	portf_max_sharp = sco.minimize(neg_sharpe_ratio, initial_guess, args=args, method='SLSQP', constraints=constraints, bounds=bounds)
	return portf_max_sharp

def plot_portfolio_optimization(random_porfolio_params, portf_ef_rtn, portf_ef_vol, portf_sharpe):
	fig, ax = plt.subplots()
	random_porfolio_params.plot(kind='scatter', x='volatility',  y='returns', c='sharpe_ratio', cmap='RdYlGn', edgecolors='black', ax=ax, figsize=[10,5])
	ax.set(xlabel='Volatility',  ylabel='Expected Returns', title='Efficient Frontier')
	ax.plot(portf_ef_vol, portf_ef_rtn, 'b--')#plotto al linea di efficenza
	ax.scatter(x=portf_sharpe['volatility'], y=portf_sharpe['returns'], c='black', marker='*', s=200, label='Max Sharpe Ratio')

	ax.legend()
	plt.tight_layout()
	plt.show()


#------------------------------------------


#punto A
#metodo analitico
df_portfolio_input = simple_return_full_daily.loc[:dt.datetime(2020,11,30)]
cov_mat = df_portfolio_input.cov() * 252
portfolio_avg_return = df_portfolio_input.mean() * 252

#calcoliamo i portafogli casuali e la efficent line
df_portfs_param, weights = random_weighted_portfolios(df_portfolio_input, n_port = 100000)
portf_ef_rtn, portf_ef_vol = portfs_efficient_frontier(df_portfs_param, n_points = 100)

#stampiamo il portafoglio ottimale
portf_max_sharp = df_portfs_param.loc[np.argmax(df_portfs_param['sharpe_ratio'])]
portf_max_sharp_w = pd.Series(data=weights[np.argmax(df_portfs_param['sharpe_ratio'])], index=df_portfolio_input.columns)

print('Portafoglio ottimale:')
for i, val in portf_max_sharp.items():
	print(i, ': ',np.round(val, 4), flush = True, end=' ')
print('\nweights: ')
for column, w in zip(df_portfolio_input.columns, portf_max_sharp_w.values):
	print(column, ': ', np.round(w, 4), flush = True, end='| ')


#plotto il tutto
plot_portfolio_optimization(df_portfs_param, portf_ef_rtn, portf_ef_vol, portf_max_sharp)




#-------------metodo con scipy
rtns_range = np.round (np.linspace(df_portfs_param['returns'].min(), df_portfs_param['returns'].max(), 200), 4)
portf_efs = portfs_efficient_frontier_scipy(df_portfolio_input, rtns_range)

vols_range = [x['fun'] for x in portf_efs]


#ricerca dello sharpe portfolio
portf_max_sharp2 = portfs_max_sharp_ratio_scipy(df_portfolio_input)
portf_max_sharp2_w = portf_max_sharp2['x']
portf_max_sharp2 = {'returns': np.sum(portf_max_sharp2_w * portfolio_avg_return), 'volatility': get_portf_vol(portf_max_sharp2_w, portfolio_avg_return, cov_mat), 'sharpe_ratio': -portf_max_sharp2['fun']}

#ora stampo i risultati


print('Portafoglio ottimale:')
for i, val in portf_max_sharp2.items():
	print(i, ': ',np.round(val, 3), flush = True, end=' ')
print('\nweights: ')
for column, w in zip(df_portfolio_input.columns, portf_max_sharp2_w):
	print(column, ': ', np.round(w, 3), flush = True, end='| ')


#plotto i risultati
plot_portfolio_optimization(df_portfs_param, rtns_range, vols_range, portf_max_sharp2)

#punto A con rendimenti attesi

df_portfolio_input_forecast = df_forecast_validation_set.pct_change().dropna()
cov_mat = df_portfolio_input_forecast.cov() * 252
portfolio_avg_return = df_portfolio_input_forecast.mean() * 252

#calcoliamo i portafogli casuali e la efficent line
df_portfs_param, weights = random_weighted_portfolios(df_portfolio_input_forecast, n_port = 100000)
portf_ef_rtn, portf_ef_vol = portfs_efficient_frontier(df_portfs_param, n_points = 100)

#stampiamo il portafoglio ottimale
portf_frecast_max_sharp = df_portfs_param.loc[np.argmax(df_portfs_param['sharpe_ratio'])]
print('Portafoglio ottimale:')
for i, val in portf_frecast_max_sharp.items():
	print(i, ': ',np.round(val, 4), flush = True, end=' ')
print('\nweights: ')
for column, w in zip(df_portfolio_input_forecast.columns, weights[np.argmax(df_portfs_param['sharpe_ratio'])]):
	print(column, ': ', np.round(w, 4), flush = True, end='| ')


#plotto il tutto
plot_portfolio_optimization(df_portfs_param, portf_ef_rtn, portf_ef_vol, portf_frecast_max_sharp)

#-------------metodo con scipy
rtns_range = np.round (np.linspace(df_portfs_param['returns'].min(), df_portfs_param['returns'].max(), 200), 5)
portf_efs = portfs_efficient_frontier_scipy(df_portfolio_input_forecast, rtns_range)

vols_range = [x['fun'] for x in portf_efs]


#ricerca dello sharpe portfolio
portf_forecast_max_sharp2 = portfs_max_sharp_ratio_scipy(df_portfolio_input_forecast)
portf_forecast_max_sharp2_w = portf_forecast_max_sharp2['x']
portf_forecast_max_sharp2 = {'returns': np.sum(portf_forecast_max_sharp2_w * portfolio_avg_return), 'volatility': get_portf_vol(portf_forecast_max_sharp2_w, portfolio_avg_return, cov_mat), 'sharpe_ratio': -portf_forecast_max_sharp2['fun']}

#ora stampo i risultati


print('Portafoglio ottimale:')
for i, val in portf_forecast_max_sharp2.items():
	print(i, ': ',np.round(val, 3), flush = True, end=' ')
print('\nweights: ')
for column, w in zip(df_portfolio_input.columns, portf_forecast_max_sharp2_w):
	print(column, ': ', np.round(w, 3), flush = True, end='| ')


#plotto i risultati
plot_portfolio_optimization(df_portfs_param, rtns_range, vols_range, portf_forecast_max_sharp2)



#punto B
if portf_max_sharp['sharpe_ratio'] > portf_max_sharp2['sharpe_ratio']:
	optimal_portfolio = portf_max_sharp
	optimal_portfolio_w = portf_max_sharp_w
else:
	optimal_portfolio = portf_max_sharp2
	optimal_portfolio_w = pd.Series(data=np.round (portf_max_sharp2_w, 6), index=df_portfolio_input.columns)

#calcolo il beta del Portafoglio
#uso il beta di ogni titolo dal punto 5
optimal_portfolio_beta = np.sum(df_progressive_beta.loc[dt.datetime(2020,11,30)] * optimal_portfolio_w)
print('beta del portafoglio ottimo :', optimal_portfolio_beta)

#punto C

portf_effective_w = np.full(len(df_stock_full.columns), 1/len(df_stock_full.columns)).tolist()

portfolio_compare_simple_return = pd.DataFrame(index= simple_return_full.index, columns = ['Optimal', 'Effective'])
portfolio_compare_simple_return['Optimal'] = (simple_return_full * optimal_portfolio_w).apply(lambda x: np.sum(x), axis=1 )
portfolio_compare_simple_return['Effective'] = (simple_return_full *portf_effective_w).apply(lambda x: np.sum(x), axis=1 )


portfolio_compare_compound_return = pd.DataFrame(index= simple_return_full.index, columns = ['Optimal', 'Effective'])
portfolio_compare_compound_return['Optimal'] = (portfolio_compare_simple_return['Optimal']+1).cumprod()-1
portfolio_compare_compound_return['Effective'] = (portfolio_compare_simple_return['Effective']+1).cumprod()-1


portfolio_compare_simple_return.plot()
portfolio_compare_compound_return.plot()
