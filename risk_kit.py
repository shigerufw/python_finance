import pandas as pd
import scipy
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize 
import math

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

def get_ind_size():
    """
    get industry size
    """
    ind = pd.read_csv('data/ind30_m_size.csv', header=0, index_col=0, parse_dates = True)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period('M')
    ind.columns= ind.columns.str.strip()
    return ind

def get_ind_n():
    """
    get industry n firms
    """
    ind = pd.read_csv('data/ind30_m_nfirms.csv', header=0, index_col=0, parse_dates = True)
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

def optimal_weights(n_points, er, cov):
    """
    -> list of weights to run the optimizer on to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    
    return weights

def gmv(cov):
    """
    Returns the weights of the Global Minimun Vol portfolio
    given the covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1,n), cov)

def plot_ef( n_points, er, cov, show_cml= False, style = '.-', riskfree_rate = 0, show_ew = False, show_gmv= False):
    """
    Plots the N-asset efficient frontier
    """
    
    weights = optimal_weights(n_points, er, cov)    
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]    
    ef = pd.DataFrame({'Returns':rets,'Volatility':vols})
    
    ax = ef.plot.line(x='Volatility', y = 'Returns', style=style)
    
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew= portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        
        ax.plot([vol_ew],[r_ew], color = 'goldenrod', marker='o', markersize = 6)
        
    if show_gmv:
        w_gmv = gmv(cov)              
        r_gmv= portfolio_return(w_gmv, er)        
        vol_gmv = portfolio_vol(w_gmv, cov)
        
        ax.plot([vol_gmv],[r_gmv], color = 'midnightblue', marker='o', markersize = 12)
        
        
    
    if show_cml == True:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        #return of msr (y)
        r_msr = portfolio_return(w_msr, er)
        #volatility of msr (X)
        vol_msr = portfolio_vol(w_msr, cov)

        #draw capital market line - CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color = 'green', marker = 'o', linestyle = 'dashed', markersize = 6, linewidth = 2)
        
        return ax


def msr(riskfree_rate,er, cov):
    """
    RiskFree rate + ER + COV -> weight vector
    """
    
    n = er.shape[0]
    
    #init guess
    init_guess = np.repeat(1/n, n)
    
    #bounds minimum of 0 maximun of 1, for n assets
    bounds = ((0.0,1),)*n
    #constrain to stop the function when met the espected return
   
    #constrain to set the sum of weights to 1
    weights_sum_to_1= {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1        
    }
    
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        returns the negative of the sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio, init_guess ,
                       args=(riskfree_rate, er,cov,),
                       method='SLSQP',
                       options={'disp': False},
                       constraints =(weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x



# CPPI



def run_cppi(risky_r, safe_r=None, start = 1000, floor=0.8, drawdown= None, riskfree_rate=0.03,m = 3):
    """
    Runs a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk budged History, Risky Weight History
    """
    #set up CPPI Parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak =  start
    m = m
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns["R"])
    
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 #fast way to fill with numbers
    
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximun(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value-floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w,1)#wont leverage
        risky_w = np.maximum(risky_w,0) #wont go short
        safe_w = 1-risky_w

        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w

        ##update the account value for this timestamp

        account_value = (risky_alloc*(1+risky_r.iloc[step])) + (safe_alloc*(1+safe_r.iloc[step]))
        ## save the current values

        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value   
        
    risky_wealth = start*(1+risky_r).cumprod()
    
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    return backtest_result


def summary_stats(r, riskfree_rate = 0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    
    ann_r = r.aggregate(annualize_rets, periods_per_year = 12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year = 12)
    ann_sr = r.aggregate(sharpe_ratio, risk_free_rate = riskfree_rate, periods_per_year = 12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())                                          
    skew = r.aggregate(skewness)                     
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified = True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher Value at Risk (5%)": cf_var5,
        "Historic CVar (5%)": hist_cvar5,
        "Sharpe Ratio":ann_sr,
        
        "Annualized Sharp": dd,
        "Max Drawdown": dd
    })
        






#GMB


def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val




#Present value

def discount (t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time t, given interest rate r
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts



def pv(flows,r):
    """
    Computes a present value of a sequence of liabilities
    l is indexed by the time and the values are the amounts of each liability
    returns the present value of the sequence
    """  
    
    dates = flows.index
    discounts = discount(dates,r)
    return discounts.multiply(flows, axis = 'rows').sum()



def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on a interest rate and current value of assets
    """
    return pv(assets,r)/pv(liabilities,r)
        


def show_funding_ratio(assets, r):
    fr = funding_ratio(assets, liabilities, r)
    print(f'{fr*100:.2f}' " %")
    
# CIR Model
def inst_to_ann(r):
    """
    Converts short rate to ann rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Converts ann rate to short rate 
    """
    return np.log1p(r)


def cir(n_years = 10, n_scenarios = 1, a = 0.05, b = 0.03, sigma = 0.05, steps_per_year = 12, r_0 = None):
    """
    Generates random interest rate evolution over time usin the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None:
        r_0 = b
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    
    #random numbers dWt
    num_steps = int(n_years*steps_per_year)+1
    
    shock = np.random.normal(0, scale= np.sqrt(dt), size = (num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    
    #for price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    
    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h+(h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    
    
    #simulating changes in rates
    for step in range(1, num_steps):
        r_t = rates[step-1]        
        d_r_t = a*(b-r_t)*dt+ sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        #generating Prices
        prices[step] = price(n_years-step*dt, rates[step])
    
    rates = pd.DataFrame(data = inst_to_ann(rates), index = range(num_steps))
    prices = pd.DataFrame(data = prices, index = range(num_steps))
        
    return rates, prices




# Bonds
def zc_cashflow(maturity, pct_pay,face_value = 100):
    """
    returns a zero coupon cashflow given the face value, maturity and the pct_pay per year 
    """
    initial_value = bond_price(maturity,face_value,0,1,pct_pay)
    ret = np.repeat(pct_pay,maturity)
    arr_ret = pd.DataFrame((ret+1).cumprod())

    cashflow = arr_ret*initial_value


    return cashflow

def bond_cash_flows(maturity, principal = 100, coupon_rate = 0.03, coupons_per_year = 12):
    """
    Return a series of cash flows generated by a bond,
    indexed by a coupin number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1,n_coupons+1)
    cash_flows = pd.Series(data = coupon_amt, index = coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows

def bond_price(maturity, principal = 100, coupon_rate=0.03, coupons_per_year=12, discount_rate = 0.03):
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index = pricing_dates, columns = discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate,coupons_per_year, discount_rate.loc[t])
            
        return prices
    else:
        if maturity <=0: return principal+principal*coupon_rate/coupons_per_year
        
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)
    
def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows
    """
    discounted_flows = discount(flows.index, discount_rate)[0]*flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights = weights)

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Return the weight W in cf_s that, along with (1-w) in cf_l will have an effective duration that matches cf_t
    t = target
    s = short
    l = long
    cf = cashflows 
    """
    
    d_t= macaulay_duration(cf_t, discount_rate)
    d_s= macaulay_duration(cf_s, discount_rate)
    d_l= macaulay_duration(cf_l, discount_rate)
    return (d_l -d_t)/(d_l-d_s)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    computes the total return of a bond based on monthly bond prices and coupons payments
    """
    coupons = pd.DataFrame(data = 0, index = monthly_prices.index, columns = monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max , int(coupons_per_year*t_max/12), dtype= int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()

# mixing 


def bt_mix(r1,r2, allocator, **kwargs):
    """
    Run a backtest between 2 sets
    """
    
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be in the same shape")
    weights = allocator(r1,r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError('Allocator returned weights that dont match r1')
    r_mix = weights*r1 +(1-weights)*r2
    return r_mix


def fixedmix_allocator(r1,r2,w1,**kwargs):
    """
    produces a time series
    """
    return pd.DataFrame(data=w1, index= r1.index, columns = r1.columns)

def terminal_values(rets):
    """
    Returns the final value of a dollar at the end of the return period
    """
    
    return (rets+1).prod()


def terminal_stats(rets, floor = 0.8, cap = np.inf, name="Stats"):
    """
    Produce Summary Statistics on terminal Values per invested Dollar across a range of N scenarios
    rets is a T x N DataFrame of returns, where, T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Sumamary Stats indexed by stat name
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = breach.mean() if reach.sum() >0 else np.nan
    e_short = (floor - terminal_wealth[breach]).mean() if breach.sum() >0 else np.nan
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        'mean': terminal_wealth.mean(),
        'std': terminal_wealth.std(),
        'p_breach': p_breach,
        'e_short': e_short,
        'p_reach': p_reach,
        'e_surplus': e_surplus
    }, orient = 'index', columns=[name])
    return sum_stats


def glidepath_allocator(r1,r2, start_glide = 1, end_glide = 0):
    """
    Simulates a target-Date-Fund Style gradual move from r1 to r2
    """
    
    n_points = r1.shape[0]
    n_columns = r1.shape[1]
    path = pd.Series(data = np.linspace(start_glide, end_glide, num = n_points))
    paths = pd.concat([path]*n_columns, axis = 'columns')
    paths.index = r1.index
    paths.columns = r1.columns
    return paths




def floor_allocator(psp_r, ghp_r, floor, zc_prices, m = 3):
    """
    Allocate between p* seeking protifolio and global hedging portifolio (GHP) with the goal to provide exposure to the upside of the PSP without violating the floor.
    Uses CPPI-Stle dynamic risk budeting algorithm by investing a multiple of the cushion in PSP
    Returns a DataFrame with the same sahpe as the psps/ghp representating the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError('PSP and ZC Prices must have the same shape')
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index = psp_r.index, columns = psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ##PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0,1)
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        ##recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
        
    return w_history

def drawndown_allocator(psp_r, ghp_r, maxdd, m = 3):
    """
    Allocate between p* seeking protifolio and global hedging portifolio (GHP) with the goal to provide exposure to the upside of the PSP without violating the floor.
    Uses CPPI-Stle dynamic risk budeting algorithm by investing a multiple of the cushion in PSP
    Returns a DataFrame with the same sahpe as the psps/ghp representating the weights in the PSP
    """

    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1,n_scenarios)
    w_history = pd.DataFrame(index = psp_r.index, columns = psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ##PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0,1)
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        ##recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
        
    return w_history
    
    


