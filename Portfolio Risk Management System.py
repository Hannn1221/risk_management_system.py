# Working imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Create sample financial data for our risk management system
np.random.seed(42)

# Generate sample portfolio data
def create_sample_data():
    # 3 years of daily data
    dates = pd.date_range('2021-01-01', '2024-01-01', freq='D')
    
    # Asset tickers
    tickers = ['SPY', 'TLT', 'GLD', 'EFA', 'VNQ']
    
    # Generate correlated returns (more realistic)
    n_assets = len(tickers)
    n_days = len(dates)
    
    # Create correlation matrix
    corr_matrix = np.array([
        [1.00, -0.20, -0.10, 0.80, 0.60],  # SPY
        [-0.20, 1.00, 0.30, -0.15, -0.10], # TLT  
        [-0.10, 0.30, 1.00, -0.05, 0.20],  # GLD
        [0.80, -0.15, -0.05, 1.00, 0.50],  # EFA
        [0.60, -0.10, 0.20, 0.50, 1.00]    # VNQ
    ])
    
    # Generate correlated returns
    mean_returns = np.array([0.0008, 0.0002, 0.0003, 0.0006, 0.0005])  # Daily
    volatilities = np.array([0.016, 0.012, 0.018, 0.018, 0.020])       # Daily
    
    # Create covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    
    # Generate returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    # Convert to prices (starting at 100)
    prices = pd.DataFrame(index=dates, columns=tickers)
    prices.iloc[0] = 100  # Starting prices
    
    for i in range(1, len(dates)):
        prices.iloc[i] = prices.iloc[i-1] * (1 + returns[i])
    
    return_df = pd.DataFrame(returns, index=dates, columns=tickers)
    
    return prices, return_df

# Create the sample data
prices, returns = create_sample_data()

print("✓ Sample financial data created!")
print(f"Prices shape: {prices.shape}")
print(f"Returns shape: {returns.shape}")
print("\nFirst 5 rows of prices:")
print(prices.head())

print("\nBasic statistics:")
annual_returns = returns.mean() * 252
annual_vol = returns.std() * np.sqrt(252)
sharpe = annual_returns / annual_vol

stats = pd.DataFrame({
    'Annual Return': annual_returns,
    'Annual Volatility': annual_vol, 
    'Sharpe Ratio': sharpe
})
print(stats)


# VaR and Risk Metrics Functions
def calculate_portfolio_returns(returns, weights):
    """Calculate portfolio returns given asset returns and weights"""
    return (returns * weights).sum(axis=1)

def historical_var(returns, confidence_level=0.05):
    """Calculate Historical VaR"""
    return returns.quantile(confidence_level)

def parametric_var(returns, confidence_level=0.05):
    """Calculate Parametric VaR (assuming normal distribution)"""
    from scipy.stats import norm
    mean = returns.mean()
    std = returns.std()
    return mean + norm.ppf(confidence_level) * std

def monte_carlo_var(returns, confidence_level=0.05, n_simulations=10000):
    """Calculate Monte Carlo VaR"""
    mean = returns.mean()
    std = returns.std()
    
    # Generate random scenarios
    simulated_returns = np.random.normal(mean, std, n_simulations)
    return np.percentile(simulated_returns, confidence_level * 100)

def expected_shortfall(returns, confidence_level=0.05):
    """Calculate Expected Shortfall (Conditional VaR)"""
    var = historical_var(returns, confidence_level)
    return returns[returns <= var].mean()

# Portfolio weights (you can modify these)
portfolio_weights = pd.Series({
    'SPY': 0.40,  # 40% US Stocks
    'TLT': 0.20,  # 20% Bonds  
    'GLD': 0.15,  # 15% Gold
    'EFA': 0.15,  # 15% International Stocks
    'VNQ': 0.10   # 10% REITs
})

print("Portfolio Weights:")
print(portfolio_weights)
print(f"Total weight: {portfolio_weights.sum():.1%}")


# Calculate portfolio returns
portfolio_returns = calculate_portfolio_returns(returns, portfolio_weights)

print(f"\nPortfolio Statistics:")
print(f"Daily Mean Return: {portfolio_returns.mean():.4f}")
print(f"Daily Volatility: {portfolio_returns.std():.4f}")
print(f"Annual Return: {portfolio_returns.mean() * 252:.2%}")
print(f"Annual Volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
print(f"Sharpe Ratio: {(portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)):.3f}")

# Calculate VaR using different methods
confidence_levels = [0.01, 0.05, 0.10]  # 99%, 95%, 90% confidence

print(f"\nValue at Risk Analysis:")
print("=" * 50)

for conf in confidence_levels:
    hist_var = historical_var(portfolio_returns, conf)
    param_var = parametric_var(portfolio_returns, conf)
    mc_var = monte_carlo_var(portfolio_returns, conf)
    es = expected_shortfall(portfolio_returns, conf)
    
    print(f"\n{(1-conf)*100:.0f}% Confidence Level:")
    print(f"  Historical VaR:   {hist_var:.4f} ({hist_var:.2%})")
    print(f"  Parametric VaR:   {param_var:.4f} ({param_var:.2%})")
    print(f"  Monte Carlo VaR:  {mc_var:.4f} ({mc_var:.2%})")
    print(f"  Expected Shortfall: {es:.4f} ({es:.2%})")
    
    
    # Create comprehensive risk charts
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Portfolio return distribution
axes[0,0].hist(portfolio_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(historical_var(portfolio_returns, 0.05), color='red', linestyle='--', 
                  label=f'95% VaR: {historical_var(portfolio_returns, 0.05):.3f}')
axes[0,0].axvline(portfolio_returns.mean(), color='green', linestyle='-', 
                  label=f'Mean: {portfolio_returns.mean():.3f}')
axes[0,0].set_title('Portfolio Return Distribution')
axes[0,0].set_xlabel('Daily Returns')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# 2. Cumulative portfolio returns
portfolio_cumulative = (1 + portfolio_returns).cumprod()
axes[0,1].plot(portfolio_cumulative.index, portfolio_cumulative, linewidth=2, color='navy')
axes[0,1].set_title('Portfolio Cumulative Performance')
axes[0,1].set_xlabel('Date')
axes[0,1].set_ylabel('Cumulative Return')
axes[0,1].grid(True, alpha=0.3)

# 3. Rolling volatility
rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252)
axes[1,0].plot(rolling_vol.index, rolling_vol, color='orange', linewidth=2)
axes[1,0].set_title('30-Day Rolling Volatility (Annualized)')
axes[1,0].set_xlabel('Date')
axes[1,0].set_ylabel('Volatility')
axes[1,0].grid(True, alpha=0.3)

# 4. Asset correlation heatmap
corr_matrix = returns.corr()
im = axes[1,1].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
axes[1,1].set_xticks(range(len(corr_matrix.columns)))
axes[1,1].set_yticks(range(len(corr_matrix.columns)))
axes[1,1].set_xticklabels(corr_matrix.columns)
axes[1,1].set_yticklabels(corr_matrix.columns)
axes[1,1].set_title('Asset Correlation Matrix')

# Add correlation values to heatmap
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = axes[1,1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=axes[1,1])
plt.tight_layout()
plt.show()

print("Risk analysis complete! 📊")

# Create a summary table comparing VaR methods
var_comparison = pd.DataFrame()

for conf in [0.01, 0.05, 0.10]:
    var_comparison[f'{(1-conf)*100:.0f}% Confidence'] = {
        'Historical VaR': historical_var(portfolio_returns, conf),
        'Parametric VaR': parametric_var(portfolio_returns, conf), 
        'Monte Carlo VaR': monte_carlo_var(portfolio_returns, conf),
        'Expected Shortfall': expected_shortfall(portfolio_returns, conf)
    }

print("\nVaR Comparison Table:")
print("=" * 60)
print(var_comparison.round(4))

# Convert to percentage for easier reading
print("\nVaR Comparison Table (Percentage):")
print("=" * 60)
print((var_comparison * 100).round(2).astype(str) + '%')


# VaR Backtesting Functions
def backtest_var(returns, var_method='historical', confidence_level=0.05, window_size=252):
    """
    Backtest VaR model using rolling window approach
    """
    results = []
    
    for i in range(window_size, len(returns)):
        # Get historical data for VaR calculation
        historical_data = returns.iloc[i-window_size:i]
        
        # Calculate VaR using specified method
        if var_method == 'historical':
            var_estimate = historical_var(historical_data, confidence_level)
        elif var_method == 'parametric':
            var_estimate = parametric_var(historical_data, confidence_level)
        elif var_method == 'monte_carlo':
            var_estimate = monte_carlo_var(historical_data, confidence_level)
        
        # Get actual return for next day
        actual_return = returns.iloc[i]
        
        # Check if VaR was violated (actual return worse than VaR estimate)
        violation = actual_return <= var_estimate
        
        results.append({
            'date': returns.index[i],
            'actual_return': actual_return,
            'var_estimate': var_estimate,
            'violation': violation
        })
    
    return pd.DataFrame(results)

# Kupiec Test for VaR Backtesting
def kupiec_test(violations, total_observations, confidence_level=0.05):
    """
    Kupiec Proportion of Failures (POF) test for VaR backtesting
    """
    from scipy.stats import chi2
    
    expected_violations = total_observations * confidence_level
    actual_violations = sum(violations)
    
    if actual_violations == 0 or actual_violations == total_observations:
        return None, None, "Cannot calculate test statistic"
    
    # Calculate likelihood ratio test statistic
    lr_stat = 2 * (
        actual_violations * np.log(actual_violations / expected_violations) +
        (total_observations - actual_violations) * 
        np.log((total_observations - actual_violations) / (total_observations - expected_violations))
    )
    
    # Critical value for 95% confidence (chi-square with 1 degree of freedom)
    critical_value = chi2.ppf(0.95, df=1)
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    
    return lr_stat, p_value, "Pass" if lr_stat < critical_value else "Fail"

# Perform backtesting
print("VaR Backtesting Analysis")
print("=" * 50)

methods = ['historical', 'parametric', 'monte_carlo']
confidence_levels = [0.01, 0.05, 0.10]

backtest_results = {}

for method in methods:
    backtest_results[method] = {}
    for conf in confidence_levels:
        # Backtest the method
        bt_result = backtest_var(portfolio_returns, method, conf, window_size=252)
        
        # Calculate statistics
        total_obs = len(bt_result)
        violations = bt_result['violation'].sum()
        violation_rate = violations / total_obs
        expected_rate = conf
        
        # Kupiec test
        lr_stat, p_value, test_result = kupiec_test(bt_result['violation'], total_obs, conf)
        
        backtest_results[method][conf] = {
            'total_observations': total_obs,
            'violations': violations,
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'kupiec_stat': lr_stat,
            'kupiec_p_value': p_value,
            'kupiec_result': test_result
        }
        
        print(f"\n{method.title()} VaR at {(1-conf)*100:.0f}% confidence:")
        print(f"  Expected violations: {expected_rate:.1%}")
        print(f"  Actual violations: {violation_rate:.1%} ({violations}/{total_obs})")
        if lr_stat is not None:
            print(f"  Kupiec test: {test_result} (p-value: {p_value:.4f})")
            
            
            # Stress Testing Functions
def stress_test_scenarios():
    """Define historical stress scenarios"""
    scenarios = {
        'COVID-19 Crash (Mar 2020)': {
            'SPY': -0.12, 'TLT': 0.03, 'GLD': -0.02, 'EFA': -0.13, 'VNQ': -0.15
        },
        '2008 Financial Crisis': {
            'SPY': -0.09, 'TLT': 0.02, 'GLD': 0.01, 'EFA': -0.10, 'VNQ': -0.12
        },
        'Black Monday (1987)': {
            'SPY': -0.22, 'TLT': 0.05, 'GLD': 0.03, 'EFA': -0.20, 'VNQ': -0.18
        },
        'Tech Bubble Burst (2000)': {
            'SPY': -0.07, 'TLT': 0.04, 'GLD': 0.02, 'EFA': -0.08, 'VNQ': -0.06
        },
        'Interest Rate Shock': {
            'SPY': -0.03, 'TLT': -0.08, 'GLD': -0.01, 'EFA': -0.04, 'VNQ': -0.05
        },
        'Inflation Shock': {
            'SPY': -0.05, 'TLT': -0.06, 'GLD': 0.08, 'EFA': -0.06, 'VNQ': 0.02
        }
    }
    return scenarios

def calculate_stress_impact(scenarios, weights):
    """Calculate portfolio impact under stress scenarios"""
    results = {}
    
    for scenario_name, asset_shocks in scenarios.items():
        # Calculate portfolio impact
        portfolio_impact = sum(weights[asset] * shock for asset, shock in asset_shocks.items())
        
        results[scenario_name] = {
            'portfolio_impact': portfolio_impact,
            'asset_contributions': {asset: weights[asset] * shock 
                                  for asset, shock in asset_shocks.items()}
        }
    
    return results

# Run stress tests
print("\n\nStress Testing Analysis")
print("=" * 50)

scenarios = stress_test_scenarios()
stress_results = calculate_stress_impact(scenarios, portfolio_weights)

for scenario, result in stress_results.items():
    impact = result['portfolio_impact']
    print(f"\n{scenario}:")
    print(f"  Portfolio Impact: {impact:.2%}")
    print("  Asset Contributions:")
    for asset, contribution in result['asset_contributions'].items():
        print(f"    {asset}: {contribution:.2%}")

# Create stress test visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Portfolio impact by scenario
scenario_names = list(stress_results.keys())
portfolio_impacts = [stress_results[s]['portfolio_impact'] for s in scenario_names]

bars1 = ax1.barh(scenario_names, portfolio_impacts, color=['red' if x < 0 else 'green' for x in portfolio_impacts])
ax1.set_xlabel('Portfolio Impact (%)')
ax1.set_title('Stress Test: Portfolio Impact by Scenario')
ax1.grid(True, alpha=0.3)

# Format x-axis as percentage
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

# Asset contribution breakdown for worst scenario
worst_scenario = min(stress_results.items(), key=lambda x: x[1]['portfolio_impact'])
worst_scenario_name = worst_scenario[0]
worst_contributions = worst_scenario[1]['asset_contributions']

assets = list(worst_contributions.keys())
contributions = list(worst_contributions.values())
colors = ['red' if x < 0 else 'green' for x in contributions]

bars2 = ax2.bar(assets, contributions, color=colors)
ax2.set_ylabel('Contribution to Portfolio Loss (%)')
ax2.set_title(f'Asset Contributions - {worst_scenario_name}')
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

plt.tight_layout()
plt.show()

# Risk Attribution Functions
def calculate_marginal_var(returns, weights, confidence_level=0.05, delta=0.01):
    """Calculate Marginal VaR for each asset"""
    base_portfolio_returns = calculate_portfolio_returns(returns, weights)
    base_var = historical_var(base_portfolio_returns, confidence_level)
    
    marginal_vars = {}
    
    for asset in weights.index:
        # Create modified weights (increase asset weight by delta)
        modified_weights = weights.copy()
        modified_weights[asset] += delta
        modified_weights = modified_weights / modified_weights.sum()  # Renormalize
        
        # Calculate new portfolio VaR
        modified_portfolio_returns = calculate_portfolio_returns(returns, modified_weights)
        modified_var = historical_var(modified_portfolio_returns, confidence_level)
        
        # Marginal VaR = change in VaR / change in weight
        marginal_vars[asset] = (modified_var - base_var) / delta
    
    return pd.Series(marginal_vars)

def calculate_component_var(returns, weights, confidence_level=0.05):
    """Calculate Component VaR (risk contribution of each asset)"""
    marginal_vars = calculate_marginal_var(returns, weights, confidence_level)
    component_vars = weights * marginal_vars
    return component_vars

def risk_attribution_analysis(returns, weights, confidence_level=0.05):
    """Comprehensive risk attribution analysis"""
    
    # Portfolio metrics
    portfolio_returns = calculate_portfolio_returns(returns, weights)
    portfolio_var = historical_var(portfolio_returns, confidence_level)
    portfolio_vol = portfolio_returns.std()
    
    # Individual asset metrics
    asset_vars = returns.apply(lambda x: historical_var(x, confidence_level))
    asset_vols = returns.std()
    
    # Risk contributions
    marginal_vars = calculate_marginal_var(returns, weights, confidence_level)
    component_vars = calculate_component_var(returns, weights, confidence_level)
    
    # Risk decomposition
    risk_attribution = pd.DataFrame({
        'Weight': weights,
        'Asset_VaR': asset_vars,
        'Asset_Vol': asset_vols,
        'Marginal_VaR': marginal_vars,
        'Component_VaR': component_vars,
        'Risk_Contribution_%': (component_vars / portfolio_var) * 100
    })
    
    return risk_attribution, portfolio_var

# Perform risk attribution analysis
print("\n\nRisk Attribution Analysis")
print("=" * 50)

risk_attr, port_var = risk_attribution_analysis(returns, portfolio_weights, confidence_level=0.05)

print(f"Portfolio 95% VaR: {port_var:.4f} ({port_var:.2%})")
print(f"Sum of Component VaRs: {risk_attr['Component_VaR'].sum():.4f}")
print("\nRisk Attribution by Asset:")
print(risk_attr.round(4))

# Risk attribution visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Weight vs Risk Contribution
ax1.scatter(risk_attr['Weight'] * 100, risk_attr['Risk_Contribution_%'], s=100, alpha=0.7)
for i, asset in enumerate(risk_attr.index):
    ax1.annotate(asset, (risk_attr['Weight'][i] * 100, risk_attr['Risk_Contribution_%'][i]), 
                xytext=(5, 5), textcoords='offset points')
ax1.plot([0, 50], [0, 50], 'r--', alpha=0.5, label='Equal Weight = Equal Risk')
ax1.set_xlabel('Portfolio Weight (%)')
ax1.set_ylabel('Risk Contribution (%)')
ax1.set_title('Weight vs Risk Contribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Component VaR by asset
component_var_pct = (risk_attr['Component_VaR'] / port_var) * 100
bars = ax2.bar(risk_attr.index, component_var_pct, color='lightcoral')
ax2.set_ylabel('Risk Contribution (%)')
ax2.set_title('Component VaR by Asset')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, component_var_pct):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{value:.1f}%', ha='center', va='bottom')

# 3. Marginal vs Component VaR
ax3.scatter(risk_attr['Marginal_VaR'], risk_attr['Component_VaR'], s=100, alpha=0.7)
for i, asset in enumerate(risk_attr.index):
    ax3.annotate(asset, (risk_attr['Marginal_VaR'][i], risk_attr['Component_VaR'][i]), 
                xytext=(5, 5), textcoords='offset points')
ax3.set_xlabel('Marginal VaR')
ax3.set_ylabel('Component VaR')
ax3.set_title('Marginal vs Component VaR')
ax3.grid(True, alpha=0.3)

# 4. Risk-adjusted returns (Sharpe ratios)
sharpe_ratios = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
ax4.bar(sharpe_ratios.index, sharpe_ratios, color='steelblue')
ax4.set_ylabel('Sharpe Ratio')
ax4.set_title('Risk-Adjusted Returns by Asset')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n🎯 Risk Management System Complete!")
print("\nSummary of Analysis:")
print(f"• Portfolio 95% VaR: {port_var:.2%}")
print(f"• Highest risk contributor: {risk_attr['Risk_Contribution_%'].idxmax()} ({risk_attr['Risk_Contribution_%'].max():.1f}%)")
print(f"• Most efficient asset (highest Sharpe): {sharpe_ratios.idxmax()} ({sharpe_ratios.max():.3f})")


# Generate comprehensive risk report
def generate_risk_report():
    """Generate a comprehensive risk management report"""
    
    report = f"""
    
    MULTI-ASSET PORTFOLIO RISK MANAGEMENT REPORT
    ============================================
    
    PORTFOLIO COMPOSITION:
    {portfolio_weights.apply(lambda x: f"{x:.1%}").to_string()}
    
    PORTFOLIO PERFORMANCE:
    • Annual Return: {portfolio_returns.mean() * 252:.2%}
    • Annual Volatility: {portfolio_returns.std() * np.sqrt(252):.2%}
    • Sharpe Ratio: {(portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)):.3f}
    
    VALUE AT RISK (95% Confidence):
    • Historical VaR: {historical_var(portfolio_returns, 0.05):.2%}
    • Parametric VaR: {parametric_var(portfolio_returns, 0.05):.2%}
    • Monte Carlo VaR: {monte_carlo_var(portfolio_returns, 0.05):.2%}
    • Expected Shortfall: {expected_shortfall(portfolio_returns, 0.05):.2%}
    
    RISK ATTRIBUTION:
    Top Risk Contributors:
    {risk_attr.nlargest(3, 'Risk_Contribution_%')[['Weight', 'Risk_Contribution_%']].apply(lambda x: f"Weight: {x['Weight']:.1%}, Risk: {x['Risk_Contribution_%']:.1f}%", axis=1).to_string()}
    
    STRESS TEST RESULTS:
    Worst Case Scenario: {worst_scenario_name}
    Portfolio Impact: {worst_scenario[1]['portfolio_impact']:.2%}
    
    """
    
    return report

print(generate_risk_report())

# Create images directory
import os
if not os.path.exists('images'):
    os.makedirs('images')

# Save all your plots with high DPI
plt.figure(figsize=(16, 12))
# [Your risk dashboard code here]
plt.savefig('images/portfolio_dashboard.png', dpi=300, bbox_inches='tight')

# Save individual plots
plt.figure(figsize=(12, 8))
# [Your VaR distribution plot]
plt.savefig('images/var_distribution.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 8))
# [Your correlation heatmap]
plt.savefig('images/correlation_heatmap.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 8))
# [Your stress test results]
plt.savefig('images/stress_test_results.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 8))
# [Your risk attribution chart]
plt.savefig('images/risk_attribution.png', dpi=300, bbox_inches='tight')
