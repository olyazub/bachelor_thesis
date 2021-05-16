import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
from scipy import optimize
import numpy_financial as np_fin

sys.path.append('D:/uni/thesis/code')

from profit_fitting_utils import _decompose
np.set_printoptions(precision=3,suppress=True)

from profit_fitting_utils import *

from statsmodels.tsa import filters
px = read_data(prd = "X")

py = read_data(prd='Y')

#%%
# try fitting revenue to trend
trend_rev, _,_= _decompose(px["Revenue"],12)
#%%
values, ratios = plot_ratios(trend_rev)
#%%
values,ratios = plot_ratios(trend_rev,min_value=3_100_000)

#%%

a,b = linear_regression(values,ratios)
k = b
L = -b/ a

#%%
dates_x = px.DateTime
forecast_dates_x = pd.date_range(min(px.DateTime),periods = 120, freq="MS")

trend_rev_dated = pd.Series(trend_rev.values, index = dates_x)

#%%
params, result = fit_other_params(k,L,trend_rev_dated,min(px.Revenue),fit_fun="5PL")
#%%
params_rev, _ = optimize.curve_fit(log_fun_4pl, np.arange(len(trend_rev)), trend_rev.values,method = "trf",
                                          bounds=([max(px["Revenue"]),0,0,0],[20_000_000,1,100,np.inf]),
                                               p0=[10_000_000,       0.5,      60,       0.7e6 ]
                                         )
params_rev_refined, _ = optimize.curve_fit(log_fun_5pl, np.arange(len(trend_rev)), trend_rev.values,method = "trf",
                                          bounds=([max(px["Revenue"]),0,0,0,0.9],[20_000_000,0.5,100,np.inf,np.inf]),
                                               p0=[*params_rev,1]
                                         )
print(params_rev)
print(params_rev_refined)
params_refined = [11893302.373, 0.049, 8.512, 1399305.923, 5.524]


#%%
long_term_frcst_simple = log_fun_5pl(np.arange(len(forecast_dates_x)),*params_rev_refined)
long_term_frcst_simple = pd.Series(long_term_frcst_simple, index = forecast_dates_x )

long_term_frcst_rev = log_fun_5pl(np.arange(len(forecast_dates_x)),*params)
long_term_frcst_rev = pd.Series(long_term_frcst_rev, index = forecast_dates_x )

long_term_frcst_rev_refined = log_fun_5pl(np.arange(len(forecast_dates_x)),*params_refined)
long_term_frcst_rev_refined = pd.Series(long_term_frcst_rev_refined, index = forecast_dates_x )


long_term_frcst_t = log_fun_5pl(np.arange(len(forecast_dates_x)),*params)
long_term_frcst_rev = pd.Series(long_term_frcst_rev, index = forecast_dates_x )

long_term_frcst_rev_refined = log_fun_5pl(np.arange(len(forecast_dates_x)),*params_refined)
long_term_frcst_rev_refined = pd.Series(long_term_frcst_rev_refined, index = forecast_dates_x )


plot_forecast_comparison(px,{ "5pl fit to trend": long_term_frcst_simple,
                             "4pl fitted":long_term_frcst_rev,
                             "5pl refined" : long_term_frcst_rev_refined,
                             }, metrics="Revenue")
print(params)


long_term_frcst_rev_refined.to_csv("revenue_X_forecast.csv")
#%%
#variable cost fitting

trend_var, _,_= _decompose(px["Variable costs"])

params_log_var, _ = optimize.curve_fit(log_fun_4pl, np.arange(len(trend_var)), trend_var.values,method = "trf",
                                          bounds=([max(px["Variable costs"]),0,0,0],[20_000_000,1,100,np.inf]),
                                               p0=[7_000_000,       0.5,      60,       0.7e6 ]
                                         )

params_log_var_5pl, _ = optimize.curve_fit(log_fun_5pl, np.arange(len(trend_var)), trend_var.values,method = "trf",
                                          bounds=([8_500_000,0,0,0,0.8],[20_000_000,0.2,100,np.inf,1]),
                                               p0=[*params_log_var,1]
                                         )



#%%

long_term_frcst_var = log_fun_4pl(np.arange(len(forecast_dates_x)),*params_log_var)
long_term_frcst_var = pd.Series(long_term_frcst_var, index = forecast_dates_x )

long_term_frcst_var_refined = log_fun_5pl(np.arange(len(forecast_dates_x)),*params_log_var_5pl)
long_term_frcst_var_refined = pd.Series(long_term_frcst_var_refined, index = forecast_dates_x )

plot_forecast_comparison(px,{"5pl fitted": long_term_frcst_var_refined,
                             "4pl fitted":long_term_frcst_var,},
                         metrics = 'Variable costs')


#%%

# fit fixed costs


trend_fixed_costs, _,_ = _decompose(px["Fixed costs"], model = "additive", period = 12)

trend_fixed_costs = pd.Series(trend_fixed_costs.values, index = px.DateTime)

params_log_fixed, _ = optimize.curve_fit(log_fun_4pl, np.arange(len(trend_fixed_costs)), trend_fixed_costs.values,method = "trf",
                                          bounds=([max(trend_fixed_costs.values),0,0,0],[5_000_000,1,100,np.inf]),
                                               p0=[3_000_000,       0.3,      60,       2.5e5       ]
                                         )

params_log_fixed_5pl, _ = optimize.curve_fit(log_fun_5pl, np.arange(len(trend_fixed_costs)), trend_fixed_costs.values,method = "trf",
                                          bounds=([max(trend_fixed_costs.values),0,-120,0,0],[10_000_000,1,100,np.inf, np.inf]),
                                               p0=[*params_log_fixed, 1]
                                         )

help(np.atleast_2d)

#%%

params_log_fixed_5pl, _ = optimize.curve_fit(log_fun_5pl, np.arange(len(trend_fixed_costs)), trend_fixed_costs.values,method = "trf",
                                          bounds=([max(trend_fixed_costs.values),0,-120,0,0],[10_000_000,1,100,np.inf, np.inf]),
                                               p0=[*params_log_fixed, 1]
                                         )

#%%
long_term_frcst_fixed = log_fun_4pl(np.arange(len(forecast_dates_x)),*params_log_fixed)
long_term_frcst_fixed = pd.Series(long_term_frcst_fixed, index = forecast_dates_x )

long_term_frcst_fixed_refined = log_fun_5pl(np.arange(len(forecast_dates_x)),*params_log_fixed_5pl)
long_term_frcst_fixed_refined = pd.Series(long_term_frcst_fixed_refined, index = forecast_dates_x )

plot_forecast_comparison(px,{"5pl fitted": long_term_frcst_fixed_refined,
                             "4pl fitted":long_term_frcst_fixed,},
                         metrics = 'Fixed costs')



#gross profit
#%%
plot_forecast_comparison(px, {
    "Gross profit": long_term_frcst_rev_refined - long_term_frcst_fixed- long_term_frcst_var_refined,
    "Revenue": long_term_frcst_rev,
    "Fixed costs": long_term_frcst_fixed,
    "Variable costs": long_term_frcst_var_refined
},   metrics="Gross Profit")
#%%
rev = pd.Series(px.Revenue.values, index = px.DateTime)
fixed_costs = pd.Series(px["Fixed costs"].values, index = pd.DatetimeIndex(px.DateTime,freq = 'MS'))
variable_costs = pd.Series(px["Variable costs"].values, index = pd.DatetimeIndex(px.DateTime,freq = 'MS'))

x_profit = pd.concat([rev,fixed_costs,variable_costs,long_term_frcst_rev,
                     long_term_frcst_var,long_term_frcst_var_refined,long_term_frcst_fixed], axis = 1)
x_profit.columns = ["Revenue","Fixed costs","Variable costs","rev_forecast","var_frcst_3",'var_frcst_2', 'fixed_forecast']
x_real = pd.concat([rev,fixed_costs,variable_costs,], axis = 1)
x_real.columns = ["Revenue","Fixed costs","Variable costs"]
with pd.ExcelWriter('output_X.xlsx') as writer:
    x_profit.to_excel(writer, sheet_name='product_X')
    x_real.to_excel(writer, sheet_name='product_X_real')
#product Y
#%%

def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power)




uncertainties_past = np.ones(46)
#uncertainties_future = powspace(0.5, 0.01, 1.5, 15)
uncertainties_future = 1e-4*np.ones(7)
uncertainties= np.concatenate([uncertainties_past, uncertainties_future], axis = 0)


forecast_dates_y = pd.date_range(min(px.DateTime),periods = 120, freq="MS")
params_y_4pl, _ = optimize.curve_fit(log_fun_4pl, np.arange(len(py.Revenue)),py.Revenue.values,
                                     bounds=([1_500_000, 0.045, 50, -2e5], [4_500_000, 0.5, 80, 1e5]),
                                     p0=[3_000_000, 0.05, 54, -2e5],
                                     method = "trf")

#params_y_4pl = [4000000.000 ,     0.050, 79.849 , 59195.954]

long_term_frcst_y = log_fun_4pl(np.arange(len(forecast_dates_y)),*params_y_4pl)
long_term_frcst_y = pd.Series(long_term_frcst_y, index = forecast_dates_y )

plot_forecast_comparison(py,{"5pl fitted": long_term_frcst_y-12} )

#%%
fixed_costs_y = pd.Series(np.tile(py["Fixed costs"][1],120),index = forecast_dates_y)
variable_costs_y = pd.Series(np.tile(py["Variable costs"][1],120),index=forecast_dates_y)
plot_forecast_comparison(py, {
    "Gross profit": long_term_frcst_y - fixed_costs_y - variable_costs_y,
    "Fixed costs": fixed_costs_y,
    "Variable costs": variable_costs_y},
                         metrics="Gross Profit")



#%%
forecast_x = pd.concat([long_term_frcst_rev_refined,long_term_frcst_fixed,long_term_frcst_var_refined],names = ['revenue','fixed','variable'], axis = 1)

forecast_x.columns =["Revenue","Fixed costs","Variable costs"]

forecast_x = forecast_x[forecast_x.index >max(px['DateTime'])]


#%%
forecast_y = pd.concat([long_term_frcst_y,fixed_costs_y,variable_costs_y],names = ['revenue','fixed','variable'], axis = 1)

forecast_y.columns =["Revenue","Fixed costs","Variable costs"]

forecast_y = forecast_y[forecast_y.index >max(py['DateTime'])]

#%%
yearly_rate = 0.15
monthly_rate = 0.0125
monthly_rate_low = 0.0083
monthly_rate_high = 0.016


def classic_profit(data):
    return data['Revenue'] - data["Fixed costs"] - data["Variable costs"]

def npv_profit(rate,data):
    profit = classic_profit(data)
    return np_fin.npv(rate, profit.values).round(5)

print("Current NPV of product X:",npv_profit(0.0125,forecast_x))
print("Current NPV of product Y:",npv_profit(0.0125,forecast_y))

#%%
data = forecast_x
profit = data['Revenue'] - data["Fixed costs"] - data["Variable costs"]
np.npv(0.0125, profit)

#%%
rates = np.arange(monthly_rate_low , monthly_rate_high+0.001,0.001)

def npv_sensitivity(rates,data):
    profit = classic_profit(data)
    npv = [np_fin.npv(rate,profit) for rate in list(rates)]
    return npv

npv_x = npv_sensitivity(rates,forecast_x)
npv_y = npv_sensitivity(rates,forecast_y)


#%%
markers_on = [4]
fig, ax = plt.subplots(1,1,figsize = (8,5))
plt.plot(npv_x, label = "X",markevery=markers_on)
plt.plot(4.1, npv_profit(0.0125,forecast_x), '-o',color = 'C0')
plt.plot(npv_y,label = "Y",color = 'red',markevery=markers_on)
plt.plot(4.1, npv_profit(0.0125,forecast_y), '-o',color = 'red')
ax.set_xlim(left = 0,right = 8)
ax.set_xticks(np.arange(0,9,1))
ax.set_xticklabels(["10%"] + [""]*7+ ["20%"])

ax.set_title("Sensitivity of NPV to discount rate")
ax.legend(loc = 'upper right')
plt.show()


#%%
px["Total costs"] = px["Fixed costs"] + px["Variable costs"]

plot_forecast_comparison(px, {
    "Estimated costs": long_term_frcst_rev_refined*0.38,

},
                         metrics="Total costs")

#%%
print("Base profit of X:",np_fin.npv(0.0125, 0.62*forecast_x["Revenue"]))
plot_forecast_comparison(px, {
    "Gross profit": long_term_frcst_rev_refined*0.62,
    "Costs": long_term_frcst_rev_refined*0.38,
    "Revenue": long_term_frcst_rev_refined
},
                         metrics="Gross Profit")

#%%

_, ax1 = plt.subplots(1,1,figsize = (8,5))


cost_percentage = np.arange(0.7,0.59,-0.01)
cost_sensitivity = [np_fin.npv(0.0125,x*forecast_x["Revenue"].values) for x in cost_percentage]

plt.plot(cost_sensitivity, label = "Profit",)
plt.plot(8, np_fin.npv(0.0125,forecast_x["Revenue"].values*0.62), '-o',color = 'C0')
ax1.set_xlim(left = 0,right = 10)
ax1.set_xticks(np.arange(0,11,1))
ax1.set_xticklabels(["30%"] + [""]*4+ ["35%"] + [""]*4+ ["40%"])
ax1.set_title("Sensitivity of NPV to cost proportion")
plt.show()


#%%
#"Fixed costs": long_term_frcst_fixed,
#"Variable costs": long_term_frcst_var_refined


plot_forecast_comparison(py,{
    'Base prediction': long_term_frcst_y - fixed_costs_y - variable_costs_y,
    'Optimistic' : long_term_frcst_y*1.1- fixed_costs_y - variable_costs_y,
    'Pessimistic' : long_term_frcst_y*0.9 - fixed_costs_y - variable_costs_y
}, metrics="Gross Profit")

#%%
gross_profit_y = pd.Series(long_term_frcst_y - 2*fixed_costs_y, index = forecast_dates_y)
gross_profit_y_low = pd.Series(long_term_frcst_y*0.9 - 2*fixed_costs_y, index = forecast_dates_y)
gross_profit_y_high = pd.Series(long_term_frcst_y*1.1 - 2*fixed_costs_y, index = forecast_dates_y)

fig, ax = plt.subplots(figsize=(15, 6))
plt.plot(py[["Gross Profit", "DateTime"]].set_index("DateTime"), color='black', marker='o', markersize=2.5,
         label='True')
plt.plot(gross_profit_y)
ax.fill_between(gross_profit_y.index, gross_profit_y_low, gross_profit_y_high, alpha = 0.3)
plt.title("Profit with 10% deviation from forecasted revenue")
plt.tight_layout()
plt.show()

#%%
forecast_y = pd.concat([long_term_frcst_y,fixed_costs_y,variable_costs_y],names = ['revenue','fixed','variable'], axis = 1)

forecast_y.columns =["Revenue","Fixed costs","Variable costs"]

forecast_y = forecast_y[forecast_y.index >max(py['DateTime'])]

forecast_y_factual = forecast_y[forecast_y.index >pd.to_datetime('2021-01-01')]

profit_base = forecast_y["Revenue"] - 2*forecast_y["Fixed costs"]
profit_base_factual = forecast_y_factual["Revenue"] - 2*forecast_y_factual["Fixed costs"]

profit_base_p = forecast_y["Revenue"]*0.9 - 2*forecast_y["Fixed costs"]
profit_base_factual_p = forecast_y_factual["Revenue"] - 2*forecast_y_factual["Fixed costs"]

