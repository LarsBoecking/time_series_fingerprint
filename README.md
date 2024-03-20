<div align="center">
_______

# Research Project
# ⏰📈 - Data-centric Algorithm Selection for Time Series Classification: Estimating Performance and Uncertainty
</div>

## 📌  Introduction & Description

The selection of algorithms for real-world time series classification is a frequently debated topic. 
Traditional methods such as neural architecture search, automated machine learning, as well as combined algorithm selection and hyperparameter optimizations are effective but require considerable computational resources.
In this work, we introduce a novel data-centric fingerprint that describes any time series classification dataset and provides insight into the algorithm selection problem without requiring training on the (unseen) dataset. 
Our selection strategy integrates model-centric AI advancements by leveraging existing benchmarks on time series datasets, building upon previously performed architecture and hyperparameter optimization.
By decomposing the multi-target regression problem, our data-centric fingerprints are used to estimate algorithm performance and uncertainty in a scalable and adaptable manner. 
Our approach is evaluated on the 112 UCR-benchmark datasets, demonstrating its effectiveness in predicting the performance of 35 state-of-the-art algorithms and providing valuable insights for effective algorithm selection, improving a naive baseline by 7.32\% on average in estimating the mean performance and 15.81\% in estimating the uncertainty. 


## 🚀  Quickstart - How to run the code 
The code was developed and tested on Python 3.11.0.

1. download project from the anonymous link
```bash
https://anonymous.4open.science/r/data_centric_estimation_time_series/
```

1. Install dependencies
```bash
cd data_centric_ts
pip install -r requirements.txt
```

1. Run the experiments in the notebooks
```bash
/notebooks/a_explore_data.ipynb
```

## 📦 Datasets [optional]
If you want to recalculate the fingerprints, download the raw univariate data set:
http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip
and place it here: datasets/Univariat_ts
then you can delete datasets/data_centric/data_centric_num_None_norm_True.csv, such that the fingerprints are recalculated on the next run.