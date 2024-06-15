<div align="center">
_______

# Research Project [*HICSS 2024*]
# ⏰📈 - Utilizing Data Fingerprints for Privacy-Preserving Algorithm Selection in Time Series Classification: Performance and Uncertainty Estimation on Unseen Datasets
</div>

## 📌  Introduction & Description

The selection of algorithms is a crucial step in designing AI services for real-world time series classification use cases. Traditional methods such as neural architecture search, automated machine learning, combined algorithm selection, and hyperparameter optimizations are effective but require considerable computational resources and necessitate access to all data points to run their optimizations. In this work, we introduce a novel data fingerprint that describes any time series classification dataset in a privacy-preserving manner and provides insight into the algorithm selection problem without requiring training on the (unseen) dataset. By decomposing the multi-target regression problem, only our data fingerprints are used to estimate algorithm performance and uncertainty in a scalable and adaptable manner. Our approach is evaluated on the 112 University of California riverside benchmark datasets, demonstrating its effectiveness in predicting the performance of 35 state-of-the-art algorithms and providing valuable insights for effective algorithm selection in time series classification service systems, improving a naive baseline by 7.32% on average in estimating the mean performance and 15.81% in estimating the uncertainty. 

## 🚀  Quickstart - How to run the code 
The code was developed and tested on Python 3.11.0.

1. download project from the anonymous link:
```bash
https://anonymous.4open.science/r/data_centric_estimation_time_series/
```

1. Install dependencies:
```bash
cd data_centric_ts
pip install -r requirements.txt
```

1. Run the experiments in the notebooks, starting with the data exploration:
```bash
/notebooks/a_explore_data.ipynb
```

## 📦 Datasets [optional]
If you want to recalculate the fingerprints, download the raw univariate data set:
http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip
and place it here: datasets/Univariat_ts
then you can delete datasets/data_centric/data_centric_num_None_norm_True.csv, such that the fingerprints are recalculated on the next run.



### 📈 Statistical Measures for Characterizing Individual Time Series Instances

| Description                    | Formula                                                      |
|--------------------------------|--------------------------------------------------------------|
| Mean ($\bar{x}^{i,d}$)         | $\frac{1}{T} \sum\limits_{t=1}^{T} x^{i,d}_{t}$               |
| Maximum change ($\max(\Delta x^{i,d})$) | $\max\limits_{t=1,\ldots,T-1} \left( x^{i,d}_{t+1} - x^{i,d}_t \right)$ |
| Standard deviation ($\sigma^{i,d}$)     | $\sqrt{\frac{1}{T-1} \sum\limits_{t=1}^{T} \left( x^{i,d}_{t} - \overline{x^{i,d}} \right)^2}$ |
| Deviation of change ($\sigma(\Delta x^{i,d})$) | $\sqrt{\frac{1}{T-2} \sum\limits_{t=1}^{T-1} \left( x^{i,d}_{t+1} - x^{i,d}_t - \overline{\Delta x^{i,d}} \right)^2}$ |
| Minimum value ($\min(x^{i,d})$)  | $\min\limits_{t=1,\ldots,T} x^{i,d}_{t}$                     |
| Coefficient of variation CV | $\frac{\sigma(x^{i,d})}{\bar{x}^{i,d}}$                      |
| Maximum value ($\max(x^{i,d})$)  | $\max\limits_{t=1,\ldots,T} x^{i,d}_{t}$                     |
| Skewness ($\gamma_{1}$)      | $\frac{\text{E}\left[ \left( x^{i,d} - \overline{x^{i,d}} \right)^3 \right]}{\sigma^3}$ |
| Percentile ($\text{P}_{th}$) | $x^{i,d}$ at the ${p\%}$-percentile                          |
| Kurtosis ($\text{Kurt}[x^{i,d}]$) | $\frac{\text{E}\left[ \left( x^{i,d} - \overline{x^{i,d}} \right)^4 \right]}{\sigma^4} - 3$ |
| Interquartile range (IQR)    | $\text{P}_{75}(x^{i,d}) - \text{P}_{25}(x^{i,d})$            |
| Autocorrelation ($R_{XX}$)   | $\frac{\sum\limits_{t=1}^{T-1} \left( x^{i,d}_t - \overline{x^{i,d}} \right)\left( x^{i,d}_{t+1} - \overline{x^{i,d}} \right)}{\sum\limits_{t=1}^{T} \left( x^{i,d}_t - \overline{x^{i,d}} \right)^2}$ |
| Mean change ($\overline{\Delta x^{i,d}}$) | $\frac{1}{T-1} \sum\limits_{t=1}^{T-1} \left( x^{i,d}_{t+1} - x^{i,d}_t \right)$ |

**Table**: Statistical measures for characterizing individual time series instances over time.
