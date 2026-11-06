"""Reproduce cryptocurrency hedge and safe-haven tests from original Bloomberg exports."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def load_prices():
    frames = []
    for ticker in ('BTC', 'ETH', 'SPX'):
        df = pd.read_excel(ROOT / 'data' / 'raw' / f'{ticker}_Bloomberg.xlsx', usecols=[0, 1])
        df.columns = ['Date', ticker]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna().drop_duplicates('Date').set_index('Date').sort_index()
        frames.append(df)
    return pd.concat(frames, axis=1).sort_index()

def make_returns(prices):
    # Match the thesis/EViews: align cryptocurrency prices to SPX trading dates FIRST.
    aligned = prices.loc[prices['SPX'].notna()]
    return np.log(aligned / aligned.shift(1)).iloc[1:]

def fit_models(returns, ticker, threshold=None):
    sample = returns[['SPX', ticker]].dropna()
    baseline = sm.OLS(sample[ticker], sm.add_constant(sample[['SPX']])).fit()
    if threshold is None:
        threshold = sample['SPX'].quantile(0.05)
    stress = (sample['SPX'] <= threshold).astype(int)
    design = pd.DataFrame({'SPX': sample['SPX'], 'SPX_x_stress': sample['SPX'] * stress}, index=sample.index)
    haven = sm.OLS(sample[ticker], sm.add_constant(design)).fit()
    return sample, baseline, haven, threshold

def main():
    (ROOT / 'figures').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    prices = load_prices()
    returns = make_returns(prices)
    returns.to_csv(ROOT / 'results' / 'aligned_log_returns.csv', float_format='%.12g')
    # Correlation matrix for common ETH/SPX/BTC dates, as in thesis Table 2.
    common = returns[['SPX', 'BTC', 'ETH']].dropna()
    common.corr().to_csv(ROOT / 'results' / 'correlations.csv')
    print('Common-sample correlations:')
    print(common.corr().round(6))
    # Shared full-SPX stress threshold; transparent and consistent across assets.
    threshold = returns['SPX'].dropna().quantile(0.05)
    records = []
    for ticker in ('BTC', 'ETH'):
        sample, baseline, haven, _ = fit_models(returns, ticker, threshold)
        for name, model in [('hedge', baseline), ('safe_haven', haven)]:
            records.append({'asset': ticker, 'model': name, 'n': len(sample), 'r_squared': model.rsquared, 'intercept': model.params['const'], 'beta_spx': model.params['SPX'], 'gamma_stress': model.params.get('SPX_x_stress', float('nan')), 'stress_beta': model.params['SPX'] + model.params.get('SPX_x_stress', 0), 'p_beta': model.pvalues['SPX'], 'p_gamma': model.pvalues.get('SPX_x_stress', float('nan'))})
            (ROOT / 'results' / f'{ticker.lower()}_{name}_ols.txt').write_text(model.summary().as_text(), encoding='utf-8')
        rolling = sample['SPX'].rolling(60).corr(sample[ticker])
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(rolling.index, rolling, lw=1)
        ax.axhline(0, color='gray', lw=.8)
        ax.set(title=f'{ticker} vs S&P 500: 60-trading-day rolling correlation', ylabel='Pearson correlation', xlabel='Date', ylim=(-1,1))
        fig.tight_layout();fig.savefig(ROOT / 'figures' / f'{ticker.lower()}_rolling_correlation.png', dpi=160);plt.close(fig)
    pd.DataFrame(records).to_csv(ROOT / 'results' / 'regression_results.csv', index=False)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    for ax, ticker in zip(axes, ('SPX', 'BTC', 'ETH')):
        prices[ticker].dropna().plot(ax=ax, title=f'{ticker} price', lw=1)
        ax.set_xlabel('')
    fig.tight_layout();fig.savefig(ROOT / 'figures' / 'price_history.png', dpi=160);plt.close(fig)
    print('Stress cutoff:', threshold)
    print(pd.DataFrame(records).round(6).to_string(index=False))

if __name__ == '__main__':
    main()
