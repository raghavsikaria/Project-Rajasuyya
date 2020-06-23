##########################################################
# Author: Raghav Sikaria
# LinkedIn: https://www.linkedin.com/in/raghavsikaria/
# Github: https://github.com/raghavsikaria
# Last Update: 23-6-2020
# Project: Project-Rajasuyya
# Description: Contains all Model utilities
##########################################################

# Library imports
from skopt.utils import load
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error
from datetime import datetime as dt
from contextlib import redirect_stdout
import fileinput
import pandas as pd

# Project imports
from utils.data_visualization_utils import DataVisualizationUtils

class ModelUtils:
    """Contains all Model utilities."""

    def __init__(self, config: dict):
        self.model_name = config['MODEL_NAME_REPORT']
        self.model_plot_path = config['MODEL_PLOT_PATH']
        self.model_summary_path = config['MODEL_SUMMARY_PATH']
        self.model_save_path = config['MODEL_SAVE']
        self.markdown_path = config['MARKDOWN_PATH']
        self.model_prediction_plot_path_html = config['MODEL_PREDICTION_PLOT_PATH_HTML']
        self.model_prediction_plot_path_image = config['MODEL_PREDICTION_PLOT_PATH_PNG']
        self.model_training_history_plot_path_html = config['MODEL_TRAINING_HISTORY_PLOT_PATH_HTML']
        self.model_training_history_plot_path_image = config['MODEL_TRAINING_HISTORY_PLOT_PATH_PNG']
        self.model_prediction_markdown_key = config['MARKDOWN_MODEL_PREDICTIONS_KEY']
        self.markdown_template = config['MARKDOWN_TEMPLATE']
        self.model_hpo_plot_keys = config['MARKDOWN_MODEL_HPO_PLOT_KEYS']
        self.model_hpo_plot_paths = config['MODEL_HPO_PLOT_PATHS']
        self.time_stamp = None
        self.data_viz_obj = DataVisualizationUtils(config)

    get_time_stamp = lambda: dt.now().strftime("%d-%m-%Y_%H-%M-%S-%p")

    @staticmethod
    def add_element_to_markdown(markdown_path: str, element_key: str, element_payload: str) -> None:
        """Adds payload to it's respective path key variable in the given Markdown document."""

        try:
            with fileinput.FileInput(markdown_path, inplace=True) as file:
                for line in file:
                    print(line.replace(element_key, element_payload), end='')
        except BaseException:
            pass

    @staticmethod
    def add_hpo_plots_to_markdown(markdown_path: str, plot_keys: list, plot_paths: list) -> None:
        """Adds payload to it's respective path key variable in the given Markdown document for all keys given in a list."""

        for i in range(len(plot_keys)):
            ModelUtils.add_element_to_markdown(markdown_path, plot_keys[i], plot_paths[i].split('/')[-1])
    
    @staticmethod
    def get_optimised_hyperparameters(hpo_path: str) -> 'list, Skopt optimised hyperparameters':
        """Loads saved optimised Hyperparameters and returns only the optimal params as a list."""

        return load(hpo_path)['x']

    @staticmethod
    def save_model_summary(model: 'tf.keras Model object', path: str) -> 'tf.keras Model summary':
        """Saves the summary of a model as a text file."""

        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)

        with open(path, 'w') as f:
            with redirect_stdout(f):
                model.summary()
        
        return model_summary

    @staticmethod
    def generate_hyperparameter_optimization_iterations_df(hpo_result: 'Skopt HPO Object', columns: 'List, names to be given for Hyperparameters') -> 'DataFrame':
        """Prepares a DataFrame with Iteration wise data of all candidate hyperparameters and returns it."""

        hpo_process_df = pd.concat([pd.DataFrame(hpo_result.x_iters, columns = columns),(pd.Series(hpo_result.func_vals, name="MSE"))], axis=1)
        hpo_process_df.index.name = "Iteration"
        hpo_process_df.index += 1
        return hpo_process_df

    def save_model(self, model: 'tf.keras Model object', generate_markdown: bool = True) -> None:
        """Saves the given tf.keras model.
        
        Here are all the model assets prepared & saved:
        1-Model Plot
        2-Model summary
        3-Save model - state, objective, variables
        4-Training history graph
        5-IF generate_markdown == TRUE
            5-a-Initialize markdown template
            5-b-Write markdown document
            5-c-Add HPO Plots to markdown document
        """

        # Initialising timestamp and formatting in all paths
        self.time_stamp = ModelUtils.get_time_stamp()
        self.model_plot_path = self.model_plot_path.format(self.time_stamp)
        self.model_summary_path = self.model_summary_path.format(self.time_stamp)
        self.model_save_path = self.model_save_path.format(self.time_stamp)
        self.model_training_history_plot_path_html = self.model_training_history_plot_path_html.format(self.time_stamp)
        self.model_training_history_plot_path_image = self.model_training_history_plot_path_image.format(self.time_stamp)

        # Preparing & saving model plot
        plot_model(model, to_file=self.model_plot_path, show_shapes=True, show_layer_names=True, dpi=96)
        # Save model summary
        model_summary = ModelUtils.save_model_summary(model,self.model_summary_path)
        # Save model        
        model.save(self.model_save_path)
        # Generate & save Training history graph
        graph = self.data_viz_obj.generate_training_history_graph(
            graph_save_path = self.model_training_history_plot_path_html,
            graph_image_save_path = self.model_training_history_plot_path_image,
            training_loss = model.history.history['loss'],
            validation_loss = model.history.history['val_loss']
        )
        # Proceed with Markdown generation if desired
        if generate_markdown:
            self.markdown_path = self.markdown_path.format(self.time_stamp)
            markdown_payload = self.markdown_template.format(
                MODEL_NAME=self.model_name,
                TIMESTAMP=self.time_stamp,
                MODEL_SUMMARY=model_summary,
                MODEL_PLOT_PATH=self.model_plot_path.split('/')[-1],
                MODEL_TRAINING_HIST_PATH=self.model_training_history_plot_path_image.split('/')[-1]
            )

            # Saving Markdown document
            with open(self.markdown_path, "w") as md_gen: 
                md_gen.write(markdown_payload)
            # Adding HPO plots to Markdown document
            ModelUtils.add_hpo_plots_to_markdown(self.markdown_path, self.model_hpo_plot_keys, self.model_hpo_plot_paths)
    
    def generate_predictions(self, model: 'tf.keras Model object', val_x: 'np.array', val_y: 'np.array', test_x: 'np.array', test_y: 'np.array') -> 'Prediction Plot:Bokeh figure, Validation Prediction: list, Test prediction:list':
        """Generates predictions on validation and test data from the given model, calculates MSE, creates & saves Prediction plot and adds it to markdown document."""
        
        # Initialize model plots
        self.model_prediction_plot_path_html = self.model_prediction_plot_path_html.format(ModelUtils.get_time_stamp())
        self.model_prediction_plot_path_image = self.model_prediction_plot_path_image.format(ModelUtils.get_time_stamp())

        # Generate prediction on validation & testing data
        validation_data_prediction = model.predict(val_x)
        testing_data_prediction = model.predict(test_x)

        # Calculate MSE on validation & test predictions
        mse_validation = mean_squared_error(val_y,validation_data_prediction)
        mse_test = mean_squared_error(test_y,testing_data_prediction)

        number_of_days_for_validation_prediction = range(len(val_x))
        number_of_days_for_testing_prediction = range(len(test_x))

        # Generate and save Prediction plot
        graph = self.data_viz_obj.generate_prediction_graph(
            graph_save_path = self.model_prediction_plot_path_html,
            graph_image_save_path = self.model_prediction_plot_path_image,
            mse_validation = mse_validation, 
            prediction_days_for_validation = number_of_days_for_validation_prediction, 
            val_y = val_y, 
            pred_val_y = validation_data_prediction, 
            mse_test = mse_test, 
            prediction_days_for_test = number_of_days_for_testing_prediction, 
            test_y = test_y, 
            pred_test_y = testing_data_prediction
        )

        # Add plot to markdown document
        ModelUtils.add_element_to_markdown(self.markdown_path, self.model_prediction_markdown_key, self.model_prediction_plot_path_image.split('/')[-1])
        return graph, validation_data_prediction, testing_data_prediction