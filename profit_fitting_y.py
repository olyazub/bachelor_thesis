import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import  numpy_financial as np_fin
from profit_fitting_utils import *
np.set_printoptions(precision=3,suppress=True)
py = read_data(prd='Y')

py_till_2020 = py[py.DateTime < "2020-01-01"]

# %%

forecast_dates_2022 = pd.date_range(pd.to_datetime("2022-01-01"), periods=12, freq="MS")

forecast_dates_2021 = pd.date_range(pd.to_datetime("2021-01-01"), periods=12, freq="MS")

py_revenue_till_2021 = pd.Series(py.loc[py.DateTime < pd.to_datetime("2021-01-01"), "Revenue"].values,
                                 index=py.loc[py.DateTime < pd.to_datetime("2021-01-01"), "DateTime"])

#%%
dates_2019_2020 = ((pd.to_datetime("2018-12-01") < py.DateTime) & (py.DateTime<= pd.to_datetime("2020-12-01")))

dates_2020_2021 = ((pd.to_datetime("2019-12-01") < py.DateTime) & (py.DateTime<= pd.to_datetime("2021-12-01")))

# cubic spline for year using data for 2020-2021 till 2022
py_revenue_2020_2021 = pd.Series(py.loc[dates_2020_2021, "Revenue"].values, index=py.loc[dates_2020_2021,"DateTime"])

forecast_dates_2022_series = pd.Series(np.tile(None,12), index = forecast_dates_2022)

rev_2020_2021_forecast_2022 = pd.concat([py_revenue_2020_2021,forecast_dates_2022_series], axis = 0)

#%%
def choose_forecast_window(past_start_date, past_end_date, future_start_date, months =12):
    past_start_date = pd.to_datetime(past_start_date)
    past_end_date = pd.to_datetime(past_end_date)
    future_start_date = pd.to_datetime(future_start_date)

    past_dates = ((past_start_date <= py.DateTime) & (py.DateTime <= past_end_date))
    past_series = pd.Series(py.loc[past_dates,"Revenue"].values,index=py.loc[past_dates,"DateTime"].values)
    future_dates = pd.date_range(future_start_date, periods=months, freq="MS")

    return [past_series, future_dates]

def cubic_spline(past_series,forecast_dates):
    '''
    :param past_series: series with DateTime index and values
    :param forecast_dates: dates for forecasting
    :return: series of past + future
    '''
    interpolator = PchipInterpolator(np.arange(len(past_series)), past_series.values,
                                     extrapolate=True)
    forecasts = []
    for t in range(1, len(forecast_dates)+1):
        forecasts.append(interpolator(len(past_series) -1 + t))

    forecasts = pd.Series(dict(zip(forecast_dates, forecasts)))

    return pd.concat([past_series.tail(1),forecasts])


def extrapolation(past_start_date, past_end_date='2021-12-01', future_start_date="2022-01-01",months = 12):
    return(cubic_spline(*choose_forecast_window(past_start_date, past_end_date, future_start_date,months)))


plot_forecast_comparison(py,{
    'after 2021': extrapolation('2019-01-01'),

}, metrics="Revenue")

# NPV till August 22: -219341.73786735797
#profit = revenues - costs
#np_fin.npv(0.0125, profit.values[:62])
# NPV August 22: 335592.7472596968



#%%
forecast_dates_y = pd.date_range(min(py.DateTime),periods = 120, freq="MS")
revenue_series = pd.Series(py["Revenue"].values,index=py["DateTime"].values)
year_1_prediction = extrapolation('2019-01-01')[1:]
revenues = pd.concat([revenue_series, year_1_prediction])
costs = pd.Series(np.tile(py["Fixed costs"].values[1]*2, 65),index=revenues.index)


year_1_prediction_opt = extrapolation('2019-01-01')[1:]*1.1
year_1_prediction_pes = extrapolation('2019-01-01')[1:]*0.9

revenues_opt = pd.concat([revenue_series, year_1_prediction_opt])
revenues_pes = pd.concat([revenue_series, year_1_prediction_pes])

plot_forecast_comparison(py,{
    'base': year_1_prediction,
    '+ 10 %' : year_1_prediction_opt,
    '- 10 % ' : year_1_prediction_pes

}, metrics="Revenue")

#%%

params_y_4pl, _ = scipy.optimize.curve_fit(log_fun_4pl, np.arange(len(revenues)),revenues.values,
                                     bounds=([2_000_000, 0.045, 50, -2e5], [7_000_000, 0.8, 100, 2e5]),
                                     p0=[4_500_000, 0.05, 65, 0],
                                     method = "trf")
print(params_y_4pl)
long_term_frcst_y = log_fun_4pl(np.arange(len(forecast_dates_y)),*params_y_4pl)
long_term_frcst_y_series = pd.Series(long_term_frcst_y, index = forecast_dates_y)

#%%
future_forecast = np.setdiff1d(forecast_dates_y.values, revenues.index.values)
revenues_opt_pred = revenues_opt.tail(24)
params_y_5pl_opt, _ = scipy.optimize.curve_fit(log_fun_5pl, np.arange(len(revenues_opt_pred)),revenues_opt_pred.values,
                                     bounds=([7_000_000, 0.1, 10, -2e5, 0], [10_000_000, 0.8, 40, 2e5, 100]),
                                     p0=[7312296.958   ,    0.1    ,    34.059 , 171144.444     ,  0.909],
                                     method = "trf")
print(params_y_5pl_opt)
num_for_pred = np.array(list(range(-41,79))).astype(int)
long_term_frcst_y_opt = log_fun_5pl(num_for_pred,*params_y_5pl_opt)
long_term_frcst_y_opt_series = pd.Series(long_term_frcst_y_opt, index = forecast_dates_y)

#%%

params_y_5pl_pes, _ = scipy.optimize.curve_fit(log_fun_5pl, np.arange(len(revenues)),revenues_pes.values,
                                     bounds=([5_000_000, 0.04, 50, -2e5,0.9], [6_300_000, 0.6, 100, 2e5,100]),
                                     p0=[6_000_000, 0.06, 65, 0,0.9],
                                     method = "trf")
print(params_y_5pl_pes)
long_term_frcst_y_pes = log_fun_5pl(np.arange(len(forecast_dates_y)),*params_y_5pl_pes)
long_term_frcst_y_pes_series = pd.Series(long_term_frcst_y_pes, index = forecast_dates_y)

#%%
plot_forecast_comparison(py,{
    'base_forecast' : long_term_frcst_y_series[pd.to_datetime("2021-12-01"):],
    'pes_forecast': long_term_frcst_y_pes_series[pd.to_datetime("2021-12-01"):],
    'opt_forecast' : long_term_frcst_y_opt_series[pd.to_datetime("2021-12-01"):]


}, metrics="Revenue")

#%%
np.linspace(start=-41, stop = 55, num = 97).astype(int)

y_profit = pd.concat([revenue_series,costs,long_term_frcst_y_series,long_term_frcst_y_opt_series,long_term_frcst_y_pes_series], axis = 1)

y_profit.columns = ['Revenue',"Costs","rev_pred",'rev_opt_pred','rev_pes_pred']

with pd.ExcelWriter('output.xlsx') as writer:
    y_profit.to_excel(writer, sheet_name='product_Y')
