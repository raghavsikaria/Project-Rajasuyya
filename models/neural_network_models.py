##########################################################
# Author: Raghav Sikaria
# LinkedIn: https://www.linkedin.com/in/raghavsikaria/
# Github: https://github.com/raghavsikaria
# Last Update: 23-6-2020
# Project: Project-Rajasuyya
# Description: Contains all different Model interfaces
# Personal Comments: Planning to introduce ABC classes
# for a better code design & dealing with code redundancy
##########################################################

# library imports
import pandas as pd
import numpy as np
import random as rn
import os

# Machine learning library imports
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import BatchNormalization,MaxPooling1D,Conv1D,Activation,Dropout,Flatten,Input,Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error

# Hyper-parameter optimization library Imports
from skopt import gp_minimize
from skopt.utils import use_named_args, dump, load
from skopt.space import Real, Categorical, Integer

# Project imports
from utils.data_visualization_utils import DataVisualizationUtils as dvu
from utils.model_utilities import ModelUtils as mu

# SETTING SESSION PARAMETERS
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

class SimpleCNN:
    """Interface for SimpleCNN model."""

    # Setting seeds
    seed_num = 1
    np.random.seed(seed_num)
    rn.seed(seed_num)
    tf.random.set_seed(seed_num)
    os.environ['PYTHONHASHSEED'] = '0'

    def __init__(self, config: dict, number_of_features: int):
        self.model_name = config['MODEL_NAME']
        self.input_shape = (config['NUMBER_OF_OBS_FROM_PAST'],number_of_features)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto',restore_best_weights=True)
        self.tensor_board = TensorBoard(log_dir=config['TS_BOARD_LOGS'])
        self.path_to_save_hyperparameters = config['MODEL_OPTIMISED_HYPERPARAMETERS']
        self.path_to_save_hpo_plots = config['MODEL_HPO_PLOT_PATHS']
        self.hpo_result = None

    def generate_model(self, use_optimised_hyperparameters: bool = True, dropout: float = 0.1, number_of_kernels: int = 5) -> 'tf.keras Model object':
        """Initializes and returns SimpleCNN model."""

        # Load hyperparameters & use them if desired
        if use_optimised_hyperparameters:
            _, number_of_kernels, dropout, _ = mu.get_optimised_hyperparameters(self.path_to_save_hyperparameters)

        # Inner function for initializing Conv layer
        def convolutional_layer(x):
            x = Conv1D(filters=number_of_kernels,kernel_size=2,strides=2)(x)
            x = Activation("relu")(x)
            x = MaxPooling1D(pool_size=1)(x)
            return x
        
        inputs = Input(shape=self.input_shape)
        x = convolutional_layer(inputs)
        x = Dropout(dropout)(x)
        x = Flatten()(x)
        x = Dense(1)(x)
        x = Activation("tanh")(x)

        model = Model(inputs, x, name=self.model_name)
        return model
    
    def compile_and_fit_model(self, model: 'tf.keras Model object', train_x: 'np.array', train_y: 'np.array', val_x: 'np.array', val_y: 'np.array', use_optimised_hyperparameters: bool = True, learning_rate: float = 0.01, batch_size: int = 100, epochs: int = 100) -> 'tf.keras Model object':
        """Compiles & fits the given model on given data and carries out the training process."""

        # Load hyperparameters & use them if desired
        if use_optimised_hyperparameters:
            learning_rate, _, _, batch_size = mu.get_optimised_hyperparameters(self.path_to_save_hyperparameters)

        # Initialize Optimizer for model
        optimizer = SGD(learning_rate=learning_rate)
        # Compile & fit the model
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, callbacks=[self.early_stopping, self.tensor_board], verbose=2, shuffle=False, validation_data = (val_x, val_y))

        return model

    def hyper_parameter_optimization_process(self, train_x: 'np.array', train_y: 'np.array', val_x: 'np.array', val_y: 'np.array', model_epochs: int = 10, n_calls: int = 11, acq_func: str = 'gp_hedge', verbose: bool = True, kappa: float = 1.96, noise: float = 0.01, n_jobs: int = -1) -> 'Skopt HPO Object':
        """Conducts hyperparameter optimization process & return its results"""

        # Initializing concerned hyper-parameters
        dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
        dim_num_conv_filters = Integer(low=1, high=28, name='num_conv_filters')
        dim_dropout = Real(low=0.1, high=0.9, prior='log-uniform', name='dropout')
        dim_batch_size = Integer(low=1, high=128, name='batch_size')

        # Setting default values for the concerned hyperparameters
        default_parameters = [1e-3, 1, 0.1, 64]
        dimensions = [dim_learning_rate, dim_num_conv_filters, dim_dropout, dim_batch_size]

        # Fitness function as objective function served to HPO process
        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, num_conv_filters, dropout, batch_size):

            # Setting seed and clearing model graphs in backend    
            tf.random.set_seed(SimpleCNN.seed_num)
            K.clear_session()
            tf.compat.v1.reset_default_graph()

            # Initializing model, compiling & training it
            model = self.generate_model(use_optimised_hyperparameters=False, dropout=dropout, number_of_kernels=num_conv_filters)
            optimizer = SGD(learning_rate=learning_rate)
            model.compile(loss='mean_squared_error', optimizer=optimizer)
            model.fit(train_x, train_y, batch_size=batch_size, epochs=model_epochs, verbose=2, shuffle=False, validation_data = (val_x, val_y))

            # Generating prediction on Validation data
            validation_data_prediction = model.predict(val_x)
            # Calculating MSE for the model trained with candidate Hyperparameters of this iteration
            mse_validation = mean_squared_error(val_y,validation_data_prediction)

            # Deleting created model
            del model
            return mse_validation
        
        hpo_result = gp_minimize(
            func=fitness,
            dimensions=dimensions,
            acq_func=acq_func,
            n_calls=n_calls,
            noise=noise,
            n_jobs=n_jobs,
            kappa = kappa,
            x0=default_parameters,
            verbose=verbose
        )

        self.hpo_result = hpo_result
        # Storing optimised Hyperparameters
        dump(hpo_result, self.path_to_save_hyperparameters, store_objective=False)
        # Generating Dataframe of all HPO process candidates and their MSE
        hpo_iterations_df = mu.generate_hyperparameter_optimization_iterations_df(hpo_result=hpo_result, columns=["Learning Rate","# Conv Filters","Dropout","Batch Size"])
        # Saving all HPO plots
        dvu.save_all_hpo_plots(hpo_result = hpo_result, hpo_iterations_df = hpo_iterations_df, path_map = self.path_to_save_hpo_plots)

        return hpo_result

class SimpleLSTM:
    """Interface for SimpleLSTM model."""

    # Setting seeds
    seed_num = 1
    np.random.seed(seed_num)
    rn.seed(seed_num)
    tf.random.set_seed(seed_num)
    os.environ['PYTHONHASHSEED'] = '0'

    def __init__(self, config: dict, number_of_features: int):
        self.model_name = config['MODEL_NAME']
        self.input_shape = (config['NUMBER_OF_OBS_FROM_PAST'],number_of_features)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto',restore_best_weights=True)
        self.tensor_board = TensorBoard(log_dir=config['TS_BOARD_LOGS'])
        self.path_to_save_hyperparameters = config['MODEL_OPTIMISED_HYPERPARAMETERS']
        self.path_to_save_hpo_plots = config['MODEL_HPO_PLOT_PATHS']
        self.hpo_result = None

    def generate_model(self, use_optimised_hyperparameters: bool = True, dropout: float = 0.1, number_of_lstm_nodes: int = 20) -> 'tf.keras Model object':
        """Initializes and returns SimpleCNN model."""

        # Load hyperparameters & use them if desired
        if use_optimised_hyperparameters:
            _, number_of_lstm_nodes, dropout, _ = mu.get_optimised_hyperparameters(self.path_to_save_hyperparameters)

        # Inner function for initializing Conv layer
        def lstm_layer(x):
            x = LSTM(units=number_of_lstm_nodes)(x)
            x = Activation("tanh")(x)
            return x
        
        inputs = Input(shape=self.input_shape)
        x = lstm_layer(inputs)
        x = Dropout(dropout)(x)
        x = Dense(1)(x)
        x = Activation("tanh")(x)

        model = Model(inputs, x, name=self.model_name)
        return model
    
    def compile_and_fit_model(self, model: 'tf.keras Model object', train_x: 'np.array', train_y: 'np.array', val_x: 'np.array', val_y: 'np.array', use_optimised_hyperparameters: bool = True, learning_rate: float = 0.01, batch_size: int = 100, epochs: int = 100) -> 'tf.keras Model object':
        """Compiles & fits the given model on given data and carries out the training process."""

        # Load hyperparameters & use them if desired
        if use_optimised_hyperparameters:
            learning_rate, _, _, batch_size = mu.get_optimised_hyperparameters(self.path_to_save_hyperparameters)

        # Initialize Optimizer for model
        optimizer = SGD(learning_rate=learning_rate)
        # Compile & fit the model
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, callbacks=[self.early_stopping, self.tensor_board], verbose=2, shuffle=False, validation_data = (val_x, val_y))

        return model

    def hyper_parameter_optimization_process(self, train_x: 'np.array', train_y: 'np.array', val_x: 'np.array', val_y: 'np.array', model_epochs: int = 10, n_calls: int = 11, acq_func: str = 'gp_hedge', verbose: bool = True, kappa: float = 1.96, noise: float = 0.01, n_jobs: int = -1) -> 'Skopt HPO Object':
        """Conducts hyperparameter optimization process & return its results"""

        # Initializing concerned hyper-parameters
        dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
        dim_num_lstm_nodes = Integer(low=20, high=300, name='num_lstm_nodes')
        dim_dropout = Real(low=0.1, high=0.9, prior='log-uniform', name='dropout')
        dim_batch_size = Integer(low=1, high=128, name='batch_size')

        # Setting default values for the concerned hyperparameters
        default_parameters = [1e-3, 50, 0.1, 64]
        dimensions = [dim_learning_rate, dim_num_lstm_nodes, dim_dropout, dim_batch_size]

        # Fitness function as objective function served to HPO process
        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, num_lstm_nodes, dropout, batch_size):

            # Setting seed and clearing model graphs in backend    
            tf.random.set_seed(SimpleCNN.seed_num)
            K.clear_session()
            tf.compat.v1.reset_default_graph()

            # Initializing model, compiling & training it
            model = self.generate_model(use_optimised_hyperparameters=False, dropout=dropout, number_of_lstm_nodes=num_lstm_nodes)
            optimizer = SGD(learning_rate=learning_rate)
            model.compile(loss='mean_squared_error', optimizer=optimizer)
            model.fit(train_x, train_y, batch_size=batch_size, epochs=model_epochs, verbose=2, shuffle=False, validation_data = (val_x, val_y))

            # Generating prediction on Validation data
            validation_data_prediction = model.predict(val_x)
            # Calculating MSE for the model trained with candidate Hyperparameters of this iteration
            mse_validation = mean_squared_error(val_y,validation_data_prediction)

            # Deleting created model
            del model
            return mse_validation
        
        hpo_result = gp_minimize(
            func=fitness,
            dimensions=dimensions,
            acq_func=acq_func,
            n_calls=n_calls,
            noise=noise,
            n_jobs=n_jobs,
            kappa = kappa,
            x0=default_parameters,
            verbose=verbose
        )

        self.hpo_result = hpo_result
        # Storing optimised Hyperparameters
        dump(hpo_result, self.path_to_save_hyperparameters, store_objective=False)
        # Generating Dataframe of all HPO process candidates and their MSE
        hpo_iterations_df = mu.generate_hyperparameter_optimization_iterations_df(hpo_result=hpo_result, columns=["Learning Rate","# LSTM Nodes","Dropout","Batch Size"])
        # Saving all HPO plots
        dvu.save_all_hpo_plots(hpo_result = hpo_result, hpo_iterations_df = hpo_iterations_df, path_map = self.path_to_save_hpo_plots)

        return hpo_result