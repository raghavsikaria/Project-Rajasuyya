# Project-Rajasuyya
Contains my attempt at predicting NIFTYBank Index returns &amp; generic modelling utilities

## [Please visit my site for this project here -> it contains entire content for the project, analysis & description, has better UI/UX and interactive plots for your convenience!](https://raghavsikaria.github.io/posts/2020-06-20-time-series-analysis-and-prediction)

[github_repository]: https://github.com/raghavsikaria/Project-Rajasuyya
[kriti_mahajan_profile]: https://www.linkedin.com/in/kriti-mahajan-174101b9/
[shreenivas_kunte_profile]: https://www.linkedin.com/in/shreenivas-kunte-cfa-cipm-0795897/
[cfa_india_soc_profile]: https://www.linkedin.com/company/cfasocietyindia/
[cfa_session_youtube_link]: https://www.youtube.com/watch?v=Rr-ztgKuaSA

**DISCLAIMER** - _What this project is not_:

1. This project does not give any investment advice & is purely for research and academic purposes
2. The analysis is based on data spanning 2 decades and the macro-economic scenario changes alone in that span, make it impractical for us to draw any inference in short term from this analysis
3. The code is not meant for production, error-catching and logging mechanism have been skipped intentionally

**AND IFF**

+ You like the project -> please star it on GitHub
+ You find anything wrong -> please raise an issue in GitHub repository [here][github_repository]
+ You want to enhance/improve/help or contribute in any capacity -> please raise a PR in GitHub repository [here][github_repository]
+ You use any of the codes -> kindly provide a link back to this project or my profile on LinkedIn/GitHub (courtsey in Open Source)
...there's always scope in improving the code design!

> I can't recall ever once having seen the name of a market timer on Forbes' annual list of the richest people in the world. If it were truly possible to predict corrections, you'd think somebody would have made billions by doing it.                               
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Peter Lynch

Hello reader, this project is completetly about Time Series - its analysis, prediction modelling and hyperparameter optimization(HPO) - specifically Bayesian HPO using Gausian Processes. I have made sure that the code is reusable. You can find some cool coding hacks and extensive API usage of libraries in the code. What's more?! Since the code is generic, you can also try it out on different data sets as well! We'll employ ARIMA, Deep Learning - CNNs, LSTMs & the likes. It's going to be extremely technical, all code has been open sourced [here][github_repository] and I have tried my best to aid you with explicit code comments and annotations.
The project has been implemented in Python & following are the libraries utilised:

+ Deep Learning library - tf.keras (yes, there's difference between this and Keras)
+ Visualization library - Bokeh
+ Hyperparameter Optimization library - Skopt (scikit-optimize)
+ Data processing library - Pandas, Numpy, sklearn & statsmodels

Will also be writing in detail in upcoming weeks about underlying math and theory of the concepts involved in the project!

I attended [CFA Society, India's][cfa_india_soc_profile] webinar on 11th April '20 by [Kriti Mahajan][kriti_mahajan_profile] moderated by [Shreenivas Kunte, CFA, CIPM][shreenivas_kunte_profile] on the topic- _"Practitioners' Insights: Using a Neural Network to Predict Stock Index Prices"_ which gave me the idea & inspiration to invest my time in this direction and this project was born!

P.S: Like always all code can be found [here][github_repository].

## Objective

+ To analyse a Time Series dataset (in this project - NIFTYBANK specifically, but you can use any dataset)
+ To conduct Data exploration for understanding the data
+ To process time-series data, normalise it and conduct data-stationarity testing
+ To prepare benchmark in predictive modelling using traditional algorithms like ARIMA
+ To apply Deep Learning techniques & try and improve on predictive modelling performance
  + To automate modelling process
  + To automate & apply Bayesian HPO techniques and find optimal hyperparameters for the models
  + To automate & generate markdown document for each model which summarizes the model in its entirety - from architecture to HPO process to performance

## Contents

1. [Data Exploration, Analysis & Processing](#data-exploration-analysis--processing)
    + [NIFTYBANK - A glance](#niftybank---a-glance)
    + [Checking for Covariance Stationarity](#checking-for-covariance-stationarity)
    + [Normalizing the Data](#normalizing-the-data)
    + [Correlation between all NIFTYBANK Index stocks](#correlation-between-all-niftybank-index-stocks)
    + [NIFTYBANK Index Stocks - Daily Lognormalized Returns](#niftybank-index-stocks---daily-lognormalized-returns)
    + [NIFTYBANK Index Stocks - Normalized Cumulative Returns](#niftybank-index-stocks---normalized-cumulative-returns)
2. [Generating Benchmark using ARIMA](#generating-benchmark-using-arima)
3. [Deep Neural Networks](#deep-neural-networks)
    + [Simple Convolutional Neural Network](#simple-convolutional-neural-network)
    + [Simple LSTM Network](#simple-lstm-network)
4. [Understanding the Code](#understanding-the-code)
5. [Acknowledgements](#acknowledgements)
6. [References](#references)

## Understanding the Code

Here is quick primer to understand repository and structure of code:

~~~txt
.
├── assets
│   ├── simple_cnn
│   └── simple_lstm
├── config
│   └── model_configurations.py
├── initiate_predictive_modelling.py
├── LICENSE
├── models
│   └── neural_network_models.py
├── README.md
└── utils
    ├── data_processing_utils.py
    ├── data_stationarity_utils.py
    ├── data_visualization_utils.py
    └── model_utilities.py
~~~

+ assets: You can find all plots, html files, markdown documents here
+ config/model_configurations.py: You can find all config variables set here for the entire project
+ initiate_predictive_modelling.py: A temporary piece of code I created over other utilities so that one can carry out end to end process and get a feel of things
+ utils/data_processing_utils.py: You can find all Data processing tools here - reading CSV, dealing with NA, interpolation of data, saving df, computing lagged features, splitting time series data, data standardization, data normalization, etc
+ utils/data_stationarity_utils.py: You can find functions here to check for covariance stationarity over given data
+ utils/data_visualization_utils.py: You can find all visualization related functions here - bokeh utils, skopt plots, etc
+ utils/model_utilities.py: You can find all model related utilities here - saving your model, generating predictions, conducting HPO, etc
+ models/neural_network_models.py: Contains interfaces for all models that we prepare

If I've had your interest so far, let's collaborate over [GitHub][github_repository] and make this better.
You can also reach out to me incase you have any queries pertaining to anything. Hope this helps!

## Acknowledgements

My sincere thanks to [Kriti Mahajan][kriti_mahajan_profile], [Shreenivas Kunte, CFA, CIPM][shreenivas_kunte_profile] & [CFA Society, India][cfa_india_soc_profile] to conduct and host the webinar. It was quite insightful & gave me all the inspritaion I needed to take this project up. The entire session can be found [here][cfa_session_youtube_link] on Youtube.

## References

#### Time Series modelling & tf.keras Resources

+ Kriti Mahajan's Colab Notebook <https://colab.research.google.com/drive/1PYj_6Y8W275ficMkmZdjQBOY7P0T2B3g#scrollTo=mCTtyN5OP22z>
+ <https://www.tensorflow.org/tutorials/structured_data/time_series>
+ <https://www.tensorflow.org/api_docs/python/tf/keras>
+ <https://www.tensorflow.org/guide/keras/sequential_model>
+ <https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/>
+ <https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/>

#### Hyperparameter Optimization Resources

+ <https://www.kdnuggets.com/2018/12/keras-hyperparameter-tuning-google-colab-hyperas.html>
+ <https://medium.com/@crawftv/parameter-hyperparameter-tuning-with-bayesian-optimization-7acf42d348e1>
+ <https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb>
+ <https://github.com/crawftv/Skopt-hyperparameter-tutorial/blob/master/scikit_optimize_tutorial.ipynb>
+ <https://towardsdatascience.com/hyperparameter-optimization-in-python-part-1-scikit-optimize-754e485d24fe>
+ <https://scikit-optimize.github.io/stable/>
+ <https://www.kaggle.com/schlerp/tuning-hyper-parameters-with-scikit-optimize>
+ <https://www.kdnuggets.com/2019/06/automate-hyperparameter-optimization.html>
+ <https://towardsdatascience.com/hyperparameter-optimization-in-python-part-0-introduction-c4b66791614b>

#### NIFTYBank Data Resources

+ <https://www.niftyindices.com/indices/equity/sectoral-indices/nifty-bank>
+ <https://economictimes.indiatimes.com/markets/nifty-bank/indexsummary/indexid-1913,exchange-50.cms>
+ <https://www.moneycontrol.com/stocks/marketstats/indexcomp.php?optex=NSE&opttopic=indexcomp&index=23>