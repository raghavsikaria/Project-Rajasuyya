##########################################################
# Author: Raghav Sikaria
# LinkedIn: https://www.linkedin.com/in/raghavsikaria/
# Github: https://github.com/raghavsikaria
# Last Update: 23-6-2020
# Project: Project-Rajasuyya
# Description: A high level wrapper for interacting with
# all utilities and carrying out end-to-end process
# Personal Comments: Please feel free to interact/modify
# this script as desired!
##########################################################

# Project imports
from config import model_configurations as conf
from models.neural_network_models import SimpleCNN, SimpleLSTM
from utils.data_processing_utils import DataCleaningAndProcessingUtils
from utils.model_utilities import ModelUtils

class InitiatePredictiveModelling:
    """A high level wrapper for interacting with all utilities and carrying out end-to-end process.
    
    Personal Comments: Please feel free to alter this script as desired!
    """

    INTERFACE_MAPPING = {'simple_cnn': {'interface': SimpleCNN}, 'simple_lstm': {'interface': SimpleLSTM}}

    def __init__(self,model_type: str):
        self.config = conf.CONFIG[f'{model_type.upper()}']
        self.model_interface = InitiatePredictiveModelling.INTERFACE_MAPPING[model_type]['interface']
        self.data_proc_obj = DataCleaningAndProcessingUtils()
        self.model_utils = ModelUtils(self.config)
        self.number_of_features = None
        self.model_object = None
    
    def process_and_normalize_data_feed_for_model(self, test_percentage: float = 0.08, validation_percentage: 'Float, as a percentage of Test%' = 0.5) -> 'Dataframe:training_data, Dataframe:validation_data, Dataframe:testing_data':
        """An umbrella function to read the data from CSV file, split the data into train, val & test components, scale & normalize it and finally return all data."""

        # Read CSV file
        df = self.data_proc_obj.read_csv_data(conf.DATA_PATH,index_column='date')
        # Push dependent variable to last position in dataframe, will help in splitting later
        df = self.data_proc_obj.adjust_df_for_dependent_variable(df, conf.DEPENDENT_VARIABLE)
        # Split the data into train, validation & test dataframes
        training_data, validation_data, testing_data = self.data_proc_obj.train_val_test_split_for_time_series(df=df, test_percentage=test_percentage, validation_percentage=validation_percentage)
        # Standardize the data - center the data
        stdz_training_data, stdz_validation_data, stdz_testing_data = self.data_proc_obj.data_z_score_standardization(training_data, validation_data, testing_data)
        # Scale the data to specific feature range - default [-1,1]
        scaled_training_data, scaled_validation_data, scaled_testing_data = self.data_proc_obj.data_minmax_scaler(stdz_training_data, stdz_validation_data, stdz_testing_data)

        return scaled_training_data, scaled_validation_data, scaled_testing_data
    
    def get_dependent_and_independent_features(self, training_data: 'DataFrame', validation_data: 'DataFrame', testing_data: 'DataFrame') -> 'np.array:train_x,np.array:train_y,np.array:val_x,np.array:val_y,np.array:test_x,np.array:test_y':
        """Reshapes all data for incorporating past data which decides forecast, splits for independent & dependent features and returns all data."""

        train_x, train_y = self.data_proc_obj.split_df(training_data, self.config['NUMBER_OF_OBS_FROM_PAST'],conf.FORECAST_PERIOD)
        val_x, val_y = self.data_proc_obj.split_df(validation_data, self.config['NUMBER_OF_OBS_FROM_PAST'],conf.FORECAST_PERIOD)
        test_x, test_y = self.data_proc_obj.split_df(testing_data, self.config['NUMBER_OF_OBS_FROM_PAST'],conf.FORECAST_PERIOD)
        
        # Initializing number of features to be used later
        self.number_of_features = train_x.shape[2]
        
        return train_x, train_y, val_x, val_y, test_x, test_y
    
    def initialise_model_interface(self) -> 'tf.keras model object':
        """Initializes chosen model interface and returns a model object."""

        self.model_object = self.model_interface(self.config, self.number_of_features)
        return self.model_object

    def optimise_hyperparameters_and_save(self, train_x: 'np.array', train_y: 'np.array', val_x: 'np.array', val_y: 'np.array', model_epochs: int = 10, n_calls: int = 11, acq_func: str = 'gp_hedge', verbose: bool = True, kappa: float = 1.96, noise: float = 0.01, n_jobs: int = -1) -> None:
        """Commences hyperparameter optimization process for the chosen model."""

        self.model_object.hyper_parameter_optimization_process(train_x, train_y, val_x, val_y, model_epochs=model_epochs, n_calls=n_calls, acq_func=acq_func, verbose=verbose, kappa=kappa, noise=noise, n_jobs=n_jobs)
    
    def generate_and_train_model(self, train_x: 'np.array', train_y: 'np.array', val_x: 'np.array', val_y: 'np.array', use_optimised_hyperparameters: bool = True, epochs: int = 10):
        """Compiles & fits [carries out training process] on the given tf.keras model object."""

        model = self.model_object.generate_model(use_optimised_hyperparameters=use_optimised_hyperparameters)
        model = self.model_object.compile_and_fit_model(model, train_x, train_y, val_x, val_y, use_optimised_hyperparameters=use_optimised_hyperparameters, epochs=epochs)
        return model

    def get_predictions_from_model(self, model: 'tf.keras model object', val_x: 'np.array', val_y: 'np.array', test_x: 'np.array', test_y: 'np.array') -> 'Bokeh Figure, list, list':
        """Carries out generating prediction on validation & test data, return predictions and plots of the same."""

        graph, validation_data_prediction, testing_data_prediction = self.model_utils.generate_predictions(model, val_x, val_y, test_x, test_y)
        return graph, validation_data_prediction, testing_data_prediction

    def save_model(self, model: 'tf.keras model object', generate_markdown: bool = True) -> None:
        """Saves the given model, its relevant plots, prepares markdown if desired."""

        self.model_utils.save_model(model, generate_markdown)