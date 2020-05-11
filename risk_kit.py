import pandas as pd
import scipy
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize 

#importing data
def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by Market Cap
    """
    
    returns = pd.read_csv('./Data/Portfolios_Formed_on_ME_monthly_EW.csv', header=0, index_col=0, parse_dates= True, na_values = -99.99)
    returns = returns[['Lo 10', 'Hi 10']]
    returns.columns=['SmallCap', 'LargeCap']
    returns = returns/100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period('M')
    
    return returns

def get_hfi_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by Market Cap
    """
    
    returns = pd.read_csv('./Data/edhec-hedgefundindices.csv', header=0, index_col=0, parse_dates= True, na_values = -99.99)
    
    returns = returns/100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period('M')
    
    return returns

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by Market Cap
    """
    
    returns = pd.read_csv('./Data/Portfolios_Formed_on_ME_monthly_EW.csv', header=0, index_col=0, parse_dates= True, na_values = -99.99)
    returns = returns[['Lo 10', 'Hi 10']]
    returns.columns=['SmallCap', 'LargeCap']
    returns = returns/100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period('M')
    
    return returns

def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    ind = pd.read_csv('data/ind30_m_vw_rets.csv', header=0, index_col=0, parse_dates = True)/100
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period('M')
    ind.columns= ind.columns.str.strip()
    return ind


#Preprocessing data

def annualize_rets(r, periods_per_year):
    """
    Annualize a set of returns
    
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol (r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    
    """
    return r.std()*(periods_per_year**0.5)


#Basic Evaluation

def drawdown(return_series:pd.Series, index_value = 1):
    """
    inputs:
    -return_series: pandas series, %of variation periodicaly
    -index_value (value set to 1), value of portifolio
    Returns: 
    pandas DataFrame with ['wealth_index','Peaks','Drawdown'] as columns
    """
    
    wealth_index = index_value*(return_series+1).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index-previous_peaks) /previous_peaks
    
    return pd.DataFrame({"Wealth":wealth_index, "Peaks":previous_peaks,"Drawdown": drawdowns})

#Statistic Evaluation

def neg_semideviation(r):
    """
    Returns the negative semideviation of r
    r must be a series or dataframe
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series of DataFrame
    Returns a float or a Series 
    """
    demeaned_r = r - r.mean()
    #use the population std, setting dof = 0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis(), but gives excess kurtosis (3-x)
    Computes the kurtosis of the supplied Series of DataFrame
    Returns a float or a Series 
    """
    demeaned_r = r - r.mean()
    #use the population std, setting dof = 0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r,level = 0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level of default
    Return True if the hypotesis of  normality is accepted, False Otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r, level = 5):
    """
    Returs a historic Value at Risc at specified Level
    Returns the Value to lose with a "level" of change
    ex. "have 5% to lose y%"
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    
    elif isinstance(r, pd.Series):
        return -np.percentile(r,level)
    
    else:
        raise TypeError("Expecet r to be Series or Dataframe")

def var_gaussian(r, level = 5, modified = False):
    """
    Returns a Gaussian Value at Risc at specified Level
    if modified == True, returns a Modified Cornish-Fischer adjusting the z factor
    """
    z = norm.ppf(level/100)
    
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
                 (z**2 -1)*s/6 +
                 (z**3 -3*z)*(k-3)/24 -
                 (2*z**3 -5*z)*(s**2)/36
            )
    return -(r.mean()+z*r.std(ddof=0))

def cvar_historic(r, level = 5):
    
    """
    Computes the conditional VaR of Series or Dataframe
    worst drop in "level" % of chance
    """
    
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r,level = level)
        return -r[is_beyond].mean()
    
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level = level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
#finance Evaluation   

def sharpe_ratio(r, risk_free_rate, periods_per_year):
    """
    computes the annualized sharpe ratio of a set of returns
    """
    rf_per_period = (1+risk_free_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


#efficient Frontier

def portfolio_return(weights, returns):
    """
    weights -> returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Weights -> volt
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov, style="-"):
    """
    plot 2 asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2- asset frontiers")
    weights = [np.array([w,1-w]) for w in np.linspace(0,1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef= pd.DataFrame({'Returns':rets, 'Vol': vols})
    return ef.plot.line(x='Vol',y='Returns',  style = '.-', figsize=(12,8))


def minimize_vol(target_r, er, cov):
    """
    target return -> weight vector
    """
    
    n = er.shape[0]
    
    #init guess
    init_guess = np.repeat(1/n, n)
    
    #bounds minimum of 0 maximun of 1, for n assets
    bounds = ((0.0,1),)*n
    #constrain to stop the function when met the espected return
    return_is_target = {
        'type':'eq',
        'args': (er,),
        'fun': lambda weights, er: target_r - portfolio_return(weights,er) #function to test if its the er is equal to er searched
    }
    #constrain to set the sum of weights to 1
    weights_sum_to_1= {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1        
    }
    
    results = minimize(portfolio_vol, init_guess ,args=(cov,),
                       method='SLSQP',
                       options={'disp': False},
                       constraints =(return_is_target, weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x