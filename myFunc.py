import pandas as pd
import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.neural_network import MLPRegressor

def preprocessing_train_test_split(X_data, y_data, test_size):

	l_y_train = pd.Series(index=y_data.columns, dtype=object, name='series y_train')
	l_y_test = pd.Series(index=y_data.columns, dtype=object, name='series y_test')

	l_X_train = pd.Series(index=X_data.index, dtype=object, name='series X_train')
	l_X_test = pd.Series(index=X_data.index, dtype=object, name='series X_test')

	for column in y_data.columns:
		X_train, X_test, y_train, y_test = train_test_split(X_data[column], y_data[column], test_size=test_size)

		l_X_train[column] = X_train
		l_X_test[column] = X_test
		l_y_train[column] = y_train
		l_y_test[column] = y_test

	return l_X_train, l_X_test, l_y_train, l_y_test

def find_best_MLPRegressor(X_train, y_train, default_model, active = True):

	if not(active):
		return default_model

	hyperparameters = pd.Series({'hidden_layer': [(15,),(30,15,),(60,30,15,)], 'solver':['lbfgs', 'sgd', 'adam'], 'activation':['logistic', 'tanh', 'relu'], 'alpha':[0.00001,0.0001,0.001], 'max_iter':[200,350,500] })
	best_model = default_model
	best_score = -1
	counter = 0

	for hidden_layer in hyperparameters['hidden_layer']:
		for solver in hyperparameters['solver']:
			for activation in hyperparameters['activation']:
				for alpha in hyperparameters['alpha']:
					for max_iter in hyperparameters['max_iter']:

						tmp_neural_network = MLPRegressor(hidden_layer_sizes=hidden_layer, solver=solver, activation=activation, alpha=alpha, max_iter=max_iter)
						score = cross_val_score(tmp_neural_network, X_train, y_train).mean()
						counter = counter +1

						if(score > best_score):
							best_score = score
							best_hyper = tmp_neural_network
							print('\n\n\nNew best: ' , score, ' ', counter, '\n',  hidden_layer, solver, activation, alpha, max_iter)

	f = open('F:\\appunti\\3° uni\\1° semestre\\Business intelligence\\python\\progetto\\model.txt', 'a')

	f.write('\n', best_model)

	f.close()
	return best_model

def si():
	print('porcodio')

def preprocessing_shift_split_df_return(df_return, nshift_day):#per reti neurali

	X_data_prev = df_return.iloc[-nshift_day:]
	y_data = df_return.iloc[:-nshift_day]

	X_data = pd.Series(dtype=object, name="Serie_stock_shiftate", index=df_return.columns)

	for col in X_data.index:
		tmp_df = pd.DataFrame(index=df_return.index)

		for i in range(1, 1 + nshift_day):
			tmp_df[col + '_lag' + str(i)] = df_return[col].shift(-i)

		tmp_df = tmp_df.dropna()
		X_data[col] = tmp_df

	return X_data_prev, X_data, y_data

neural_network = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30,), random_state=1)
forecast_model = neural_network.fit([[1], [0], [7], [9], [4], [5]], [11, 10, 17, 19, 14, 15] )


#l_X_test['Goldman Sachs'].dropna(), l_y_test['Goldman Sachs'].dropna()
forecast_model.score([[3], [6], [11]], [13, 16, 21])
forecast_model.predict([[3], [6], [11]])


def preprocessing_shift_split_df_return(df_return, nsplit_day):

	X_data_prev = df_return.iloc[-nsplit_day:]
	y_data = df_return.iloc[: -nsplit_day]

	X_data = pd.DataFrame(index=df_return.index)

	for col in df_return.columns:
		X_data[col] = df_return[col].shift(-nsplit_day)

	X_data = X_data.dropna()
	return X_data_prev, X_data, y_data






def predict_not_supervised(forecast_model, input_serie, data_index):


	df_X = pd.DataFrame(columns = input_serie.index, index = data_index)
	df_y = pd.Series(name='Risultato', index=data_index, dtype='float64')

	df_X.iloc[0] = input_serie
	df_y[0] = forecast_model.predict(df_X.iloc[:1])

	for next_row in range(1, len (df_X.index)):

		df_X.iloc[next_row] =  np.append(df_y[next_row-1], df_X.iloc[next_row-1].shift(1).dropna())

		pred = forecast_model.predict(df_X.iloc[next_row-1:next_row])
		df_y[next_row] = pred

	return df_y



#compra tramite macd
#per l'RSI si usa il daily
overbought = 70
oversold = 4

nday = 14
trading_data = pd.Series(index = df_stock_full_daily.columns, dtype=object)
df_trading_expanded = add_Data_df(df_stock_full_daily, tickers, nday+4)


for column in df_trading_expanded.columns: #calcolo RSI per ogni stock
	tmp = pd.DataFrame(index = df_trading_expanded.index, dtype=object)

	tmp[column] =  df_trading_expanded[column]
	tmp[column + '_rtn'] = df_trading_expanded[column].pct_change()

	tmp['abs change'] = df_trading_expanded[column].diff(1)
	tmp = tmp.dropna()

	tmp['positive move'] = tmp['abs change'].apply(lambda x: x if x>0 else 0)
	tmp['negative move'] = tmp['abs change'].apply(lambda x: -x if x<0 else 0)
	tmp['RSI'] = 100 - (100 /(1 + (tmp['positive move'].rolling(nday).mean() /tmp['negative move'].rolling(nday).mean() )))

	trading_data[column] =  tmp[[column, column + '_rtn','RSI']].dropna()

for column in trading_data.index:
	last_op = 0

	for i in trading_data[column].index:

		if trading_data[column].at[i, 'RSI'] <= oversold:
			trading_data[column].at[i, 'invested'] = 1
			last_op = 1
		elif trading_data[column].at[i, 'RSI'] >= overbought:
			trading_data[column].at[i, 'invested'] = 0
			last_op = 0
		else:
			trading_data[column].at[i, 'invested'] = last_op

trading_data[column].tail(50)
strategy_rtn = pd.DataFrame(index = df_trading_expanded.columns, columns = ['Buy and hold rtn', 'Strategy rtn'], dtype='float64')

(trading_data[column][column + '_rtn']>0).sum()
(trading_data[column][trading_data[column]['invested'] == 1] [column+ '_rtn'] >0).sum()

for column in strategy_rtn.index:
	strategy_rtn.at[column, 'Buy and hold rtn'] = ((trading_data[column][column + '_rtn']+1).cumprod()-1)[-1:]
	strategy_rtn.at[column, 'Strategy rtn'] = np.cumprod(1+trading_data[column][trading_data[column]['invested'] == 1] [column+ '_rtn']) [-1:]-1

strategy_rtn

tmp.plot()



















def preprocessing_train_test_split(X_data, y_data, test_days):
	l_X_train = pd.Series(index=X_data.index, dtype=object, name='series X_train')
	l_X_test = pd.Series(index=X_data.index, dtype=object, name='series X_test')

	for column in X_data.index:
		l_X_train[column] = X_data[column][: -test_days]
		l_X_test[column] = X_data[column][-test_days :]

	l_y_train = y_data[: -test_days]
	l_y_test = y_data[-test_days :]

	return l_X_train, l_X_test, l_y_train, l_y_test
