Portflio Risk Management System
# Portfolio Risk Management System

A comprehensive VaR analysis system built in Python demonstrating quantitative finance risk management concepts.

## 🎯 Project Overview
This project implements a multi-asset portfolio risk management system with:
- **VaR Calculations**: Historical, Parametric, and Monte Carlo methods
- **Model Backtesting**: Kupiec test validation 
- **Stress Testing**: Historical crisis scenarios
- **Risk Attribution**: Component VaR analysis

## 📊 Key Results
Prices shape: (1096, 5)
Returns shape: (1096, 5)

First 5 rows of prices:
                   SPY         TLT         GLD         EFA         VNQ
2021-01-01         100         100         100         100         100
2021-01-02   99.792698  100.632141  103.189506  100.704065  100.628489
2021-01-03  101.739857   98.562863  103.409081  101.284161    100.3361
2021-01-04  103.889233   97.217693  102.274148    102.3164  100.103286
2021-01-05  102.212037   96.149233  102.386979  100.070403   97.102784

Basic statistics:
     Annual Return  Annual Volatility  Sharpe Ratio
SPY       0.129243           0.251565      0.513755
TLT      -0.017341           0.183219     -0.094647
GLD      -0.024122           0.292462     -0.082480
EFA       0.182592           0.280303      0.651409
VNQ       0.062088           0.324555      0.191303
Portfolio Weights:
SPY    0.40
TLT    0.20
GLD    0.15
EFA    0.15
VNQ    0.10
dtype: float64
Total weight: 100.0%

Portfolio Statistics:
Daily Mean Return: 0.0003
Daily Volatility: 0.0103
Annual Return: 7.82%
Annual Volatility: 16.28%
Sharpe Ratio: 0.481

Value at Risk Analysis:
==================================================

99% Confidence Level:
  Historical VaR:   -0.0239 (-2.39%)
  Parametric VaR:   -0.0235 (-2.35%)
  Monte Carlo VaR:  -0.0240 (-2.40%)
  Expected Shortfall: -0.0278 (-2.78%)

95% Confidence Level:
  Historical VaR:   -0.0168 (-1.68%)
  Parametric VaR:   -0.0166 (-1.66%)
  Monte Carlo VaR:  -0.0161 (-1.61%)
  Expected Shortfall: -0.0214 (-2.14%)

90% Confidence Level:
  Historical VaR:   -0.0125 (-1.25%)
  Parametric VaR:   -0.0128 (-1.28%)
  Monte Carlo VaR:  -0.0130 (-1.30%)
  Expected Shortfall: -0.0180 (-1.80%)

 
 Risk analysis complete! 📊

VaR Comparison Table:
============================================================
                    99% Confidence  95% Confidence  90% Confidence
Historical VaR             -0.0239         -0.0168         -0.0125
Parametric VaR             -0.0235         -0.0166         -0.0128
Monte Carlo VaR            -0.0231         -0.0168         -0.0127
Expected Shortfall         -0.0278         -0.0214         -0.0180

VaR Comparison Table (Percentage):
============================================================
                   99% Confidence 95% Confidence 90% Confidence
Historical VaR             -2.39%         -1.68%         -1.25%
Parametric VaR             -2.35%         -1.66%         -1.28%
Monte Carlo VaR            -2.31%         -1.68%         -1.27%
Expected Shortfall         -2.78%         -2.14%          -1.8%
VaR Backtesting Analysis
==================================================

Historical VaR at 99% confidence:
  Expected violations: 1.0%
  Actual violations: 1.7% (14/844)
  Kupiec test: Pass (p-value: 0.0789)

Historical VaR at 95% confidence:
  Expected violations: 5.0%
  Actual violations: 5.6% (47/844)
  Kupiec test: Pass (p-value: 0.4562)

Historical VaR at 90% confidence:
  Expected violations: 10.0%
  Actual violations: 11.3% (95/844)
  Kupiec test: Pass (p-value: 0.2322)

Parametric VaR at 99% confidence:
  Expected violations: 1.0%
  Actual violations: 1.1% (9/844)
  Kupiec test: Pass (p-value: 0.8480)

Parametric VaR at 95% confidence:
  Expected violations: 5.0%
  Actual violations: 5.5% (46/844)
  Kupiec test: Pass (p-value: 0.5539)

Parametric VaR at 90% confidence:
  Expected violations: 10.0%
  Actual violations: 10.3% (87/844)
  Kupiec test: Pass (p-value: 0.7665)

Monte_Carlo VaR at 99% confidence:
  Expected violations: 1.0%
  Actual violations: 1.1% (9/844)
  Kupiec test: Pass (p-value: 0.8480)

Monte_Carlo VaR at 95% confidence:
  Expected violations: 5.0%
  Actual violations: 5.1% (43/844)
  Kupiec test: Pass (p-value: 0.8998)

Monte_Carlo VaR at 90% confidence:
  Expected violations: 10.0%
  Actual violations: 10.3% (87/844)
  Kupiec test: Pass (p-value: 0.7665)


Stress Testing Analysis
==================================================

COVID-19 Crash (Mar 2020):
  Portfolio Impact: -7.95%
  Asset Contributions:
    SPY: -4.80%
    TLT: 0.60%
    GLD: -0.30%
    EFA: -1.95%
    VNQ: -1.50%

2008 Financial Crisis:
  Portfolio Impact: -5.75%
  Asset Contributions:
    SPY: -3.60%
    TLT: 0.40%
    GLD: 0.15%
    EFA: -1.50%
    VNQ: -1.20%

Black Monday (1987):
  Portfolio Impact: -12.15%
  Asset Contributions:
    SPY: -8.80%
    TLT: 1.00%
    GLD: 0.45%
    EFA: -3.00%
    VNQ: -1.80%

Tech Bubble Burst (2000):
  Portfolio Impact: -3.50%
  Asset Contributions:
    SPY: -2.80%
    TLT: 0.80%
    GLD: 0.30%
    EFA: -1.20%
    VNQ: -0.60%

Interest Rate Shock:
  Portfolio Impact: -4.05%
  Asset Contributions:
    SPY: -1.20%
    TLT: -1.60%
    GLD: -0.15%
    EFA: -0.60%
    VNQ: -0.50%

Inflation Shock:
  Portfolio Impact: -2.70%
  Asset Contributions:
    SPY: -2.00%
    TLT: -1.20%
    GLD: 1.20%
    EFA: -0.90%
    VNQ: 0.20%


Risk Attribution Analysis
==================================================
Portfolio 95% VaR: -0.0168 (-1.68%)
Sum of Component VaRs: 0.0027

Risk Attribution by Asset:
     Weight  Asset_VaR  ...  Component_VaR  Risk_Contribution_%
SPY    0.40    -0.0260  ...        -0.0007               4.3888
TLT    0.20    -0.0187  ...         0.0048             -28.6045
GLD    0.15    -0.0294  ...         0.0000              -0.2802
EFA    0.15    -0.0287  ...        -0.0005               3.2508
VNQ    0.10    -0.0328  ...        -0.0009               5.2980

[5 rows x 6 columns]

🎯 Risk Management System Complete!

Summary of Analysis:
• Portfolio 95% VaR: -1.68%
• Highest risk contributor: VNQ (5.3%)
• Most efficient asset (highest Sharpe): EFA (0.651)

    
    MULTI-ASSET PORTFOLIO RISK MANAGEMENT REPORT
    ============================================
    
    PORTFOLIO COMPOSITION:
    SPY    40.0%
TLT    20.0%
GLD    15.0%
EFA    15.0%
VNQ    10.0%
    
    PORTFOLIO PERFORMANCE:
    • Annual Return: 7.82%
    • Annual Volatility: 16.28%
    • Sharpe Ratio: 0.481
    
    VALUE AT RISK (95% Confidence):
    • Historical VaR: -1.68%
    • Parametric VaR: -1.66%
    • Monte Carlo VaR: -1.69%
    • Expected Shortfall: -2.14%
    
    RISK ATTRIBUTION:
    Top Risk Contributors:
    VNQ    Weight: 10.0%, Risk: 5.3%
SPY    Weight: 40.0%, Risk: 4.4%
EFA    Weight: 15.0%, Risk: 3.3%
    
    STRESS TEST RESULTS:
    Worst Case Scenario: Black Monday (1987)
    Portfolio Impact: -12.15%

## 🛠️ Technical Skills Demonstrated
- Advanced Python programming (pandas, numpy, scipy)
- Quantitative finance theory and risk management
- Statistical hypothesis testing and model validation
- Data visualization and financial modeling

## 📈 Usage
```python
python portfolio_risk_management.py
