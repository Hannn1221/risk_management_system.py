# 📊 Portfolio Risk Management System  

A comprehensive **Python-based multi-asset portfolio risk management system** demonstrating advanced **quantitative finance and risk modeling concepts**.  

---

## 🎯 Project Overview  
This system implements a full portfolio risk management pipeline with:  

- **Value at Risk (VaR):** Historical, Parametric, Monte Carlo  
- **Expected Shortfall (CVaR):** Tail-risk measurement  
- **Backtesting:** Kupiec test for model validation  
- **Stress Testing:** Historical crisis scenarios (COVID-19, 2008, Black Monday, etc.)  
- **Risk Attribution:** Component VaR & marginal contributions  

---

## 📈 Portfolio Highlights  

**Composition**  
- SPY **40%**  
- TLT **20%**  
- GLD **15%**  
- EFA **15%**  
- VNQ **10%**  

**Performance**  
- Annual Return: **7.82%**  
- Annual Volatility: **16.28%**  
- Sharpe Ratio: **0.48**  

**Risk (95% Confidence)**  
- Historical VaR: **-1.68%**  
- Parametric VaR: **-1.66%**  
- Monte Carlo VaR: **-1.69%**  
- Expected Shortfall: **-2.14%**  

**Stress Testing (Worst Case)**  
- Black Monday (1987): **-12.15% portfolio impact**  

**Risk Attribution (95% VaR)**  
- Top Contributor: **VNQ (5.3%)**  
- SPY & EFA also significant contributors  

---

## 🛠️ Technical Skills Demonstrated  
- **Python Programming:** pandas, numpy, scipy, matplotlib  
- **Risk Modeling:** VaR, Expected Shortfall, Stress Testing  
- **Model Validation:** Kupiec test backtesting  
- **Quantitative Finance:** portfolio statistics & risk attribution  
- **Data Visualization & Statistical Analysis**  

---

## 🚀 Usage  

You can interact with the project in different ways:  

### Run the Full Pipeline (Single Script)  
```bash
python portfolio_risk_management.py
