##########################################################
# Author: Raghav Sikaria
# LinkedIn: https://www.linkedin.com/in/raghavsikaria/
# Github: https://github.com/raghavsikaria
# Last Update: 23-6-2020
# Project: Project-Rajasuyya
# Description: Contains all Data Visualization utilities
##########################################################

# Library imports
# Bokeh Imports
from bokeh.io import export_png, export_svgs
from bokeh.plotting import figure, output_file, show, save
from bokeh.layouts import grid, column, gridplot, layout, row
from bokeh.models import ColumnDataSource, Title, Legend, Circle, Line, DataTable, TableColumn

# SKOPT Visualization imports
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from skopt.plots import plot_convergence, plot_evaluations, plot_objective, plot_regret

class DataVisualizationUtils:
    """Contains all Data Visualization utilities."""

    def __init__(self, config: 'dict, visualization config'):
        """Initializes class object with visualization configurations."""

        self.model_name = config['MODEL_NAME_REPORT']
        self.plot_width = config['BOKEH']['PLOT_WIDTH']
        self.plot_height = config['BOKEH']['PLOT_HEIGHT']

        # Validation and Testing Data Prediction Plot configs
        self.validation_prediction_plot_title = config['BOKEH']['VALIDATION_PREDICTION_TITLE']
        self.testing_prediction_plot_title = config['BOKEH']['TESTING_PREDICTION_TITLE']
        self.prediction_plot_axes_labels = config['BOKEH']['PREDICTION_AXES_LABELS']
        self.prediction_plot_legend = (config['BOKEH']['PREDICTION_LEGEND_ACTUAL_RETURNS'],config['BOKEH']['PREDICTION_LEGEND_PREDICTED_RETURNS'])

        # Training History Plot configs
        self.training_history_plot_title = config['BOKEH']['TRAINING_HISTORY_TITLE']
        self.training_history_plot_axes_labels = config['BOKEH']['TRAINING_HISTORY_AXES_LABELS']
        self.training_history_plot_legend = (config['BOKEH']['TRAINING_HISTORY_LEGEND_TRAINING_LOSS'],config['BOKEH']['TRAINING_HISTORY_LEGEND_VALIDATION_LOSS'])

    get_column_data_source = lambda x, y1, y2: ColumnDataSource(data=dict(x = x, y1 = y1, y2 = y2))

    @staticmethod
    def save_hpo_plot_convergence(hpo_result: 'Skopt HPO Object', path: str) -> None:
        """Plots and saves convergence from HPO process"""

        plt.figure(figsize=(18,10))
        plot_convergence(hpo_result, size=1.1)
        plt.savefig(path, bbox_inches='tight', pad_inches=1)
    
    @staticmethod
    def save_hpo_plot_evaluations(hpo_result: 'Skopt HPO Object', path: str) -> None:
        """Plots and saves evaluations from HPO process"""

        plt.figure(figsize=(20,20))
        plot_evaluations(hpo_result)
        plt.savefig(path, bbox_inches='tight', pad_inches=1)
    
    @staticmethod
    def save_hpo_plot_objective(hpo_result: 'Skopt HPO Object', path: str) -> None:
        """Plots and saves objective from HPO process"""

        plot_objective(hpo_result, size = 4)
        plt.savefig(path, bbox_inches='tight', pad_inches=1)
    
    @staticmethod
    def save_hpo_plot_regret(hpo_result: 'Skopt HPO Object', path: str) -> None:
        """Plots and saves regret from HPO process"""

        plt.figure(figsize=(18,10))
        plot_regret(hpo_result)
        plt.savefig(path, bbox_inches='tight', pad_inches=1)

    @staticmethod
    def save_df_as_image(df: 'DataFrame', path: str, height: int = 1400, width: int = 700) -> None:
        """Saves the given DataFrame as a PNG image"""

        source = ColumnDataSource(df)
        df_columns = [df.index.name]
        df_columns.extend(df.columns.values)
        columns_for_table=[]
        for column in df_columns:
            # Adding all TableColumn props
            columns_for_table.append(TableColumn(field=column, title=column))

        # Preparing Bokeh table with all TableColumns
        data_table = DataTable(source=source, columns=columns_for_table, height = height, width = width, height_policy="auto", width_policy="auto", index_position=None)
        export_png(data_table, filename = path)

    @staticmethod
    def save_all_hpo_plots(hpo_result: 'Skopt HPO Object', hpo_iterations_df: 'Dataframe', path_map: list) -> None:
        """Umbrella utility function to save all HPO result plots."""

        DataVisualizationUtils.save_df_as_image(df=hpo_iterations_df, path= path_map[0])
        DataVisualizationUtils.save_hpo_plot_convergence(hpo_result=hpo_result, path= path_map[1])
        DataVisualizationUtils.save_hpo_plot_evaluations(hpo_result=hpo_result, path= path_map[2])
        DataVisualizationUtils.save_hpo_plot_objective(hpo_result=hpo_result, path= path_map[3])
        DataVisualizationUtils.save_hpo_plot_regret(hpo_result=hpo_result, path= path_map[4])

    @staticmethod
    def initialise_legend_properties(figure: 'Bokeh Figure Object', location: 'str, Legend Location' = 'top_right', background_fill_alpha: float = 0.5, background_fill_color: str = 'black', label_text_color: str = 'white', click_policy: str = 'mute') -> 'Bokeh Figure':
        """Initializes LEGEND properties in a given Bokeh plot & returns the same."""

        figure.legend.location = location
        figure.legend.background_fill_alpha = background_fill_alpha
        figure.legend.background_fill_color = background_fill_color
        figure.legend.label_text_color = label_text_color
        figure.legend.click_policy = click_policy

        return figure

    @staticmethod
    def initialize_graph_figure(title: str, axes_labels: 'tuple, (X-axis, Y-axis)', plot_height: int = 500, plot_width: int = 1000) -> 'Bokeh Figure':
        """Initializes Bokeh figure & returns the same."""

        # Initializing figure
        p = figure(plot_width=plot_width, plot_height=plot_height, toolbar_location='right', x_axis_label=axes_labels[0], y_axis_label=axes_labels[1])

        # Giving title to figure
        p.add_layout(Title(text=title, text_font_size="16pt"), 'above')
        p.title.text_font_size = '20pt'

        # Setting background color & opacity
        p.background_fill_color = "black"
        p.background_fill_alpha = 0.9
        return p
    
    @staticmethod
    def save_bokeh_figure(figure: 'Bokeh Figure', figure_html_save_path: str, figure_png_save_path: str) -> None:
        """Sets output path for Bokeh figure, saves it as HTML and as a PNG image."""

        # Set output HTML file path
        output_file(figure_html_save_path)
        # Save graph as HTML
        save(figure)
        # Save graph as PNG image
        export_png(figure, filename = figure_png_save_path)


    @staticmethod
    def add_multiline_to_plot(figure: 'Bokeh Figure', legend: tuple, column_data_source: 'CDS for Bokeh figure', line_width: int=2, line_1_color: str='#CC0000', line_2_color: str='orange') -> 'Bokeh Figure':
        """Adds multiple lines on Bokeh figure, initializes legend and return Bokeh figure."""

        figure.line(x='x', y='y1', line_width=line_width, line_color=line_1_color, legend=legend[0], source=column_data_source)
        figure.line(x='x', y='y2', line_width=line_width, line_color=line_2_color, legend=legend[1], source=column_data_source)
        figure = DataVisualizationUtils.initialise_legend_properties(figure)
        return figure

    @staticmethod
    def add_multiline_and_circle_glyph_to_plot(figure: 'Bokeh Figure', legend: tuple, column_data_source: 'CDS for Bokeh figure', circle_size: int = 8, line_1_color: str='#CC0000', line_2_color: str='orange') -> 'Bokeh Figure':
        """Adds multiple lines + circle glyphs on Bokeh figure, initializes legend and return Bokeh figure."""

        figure = DataVisualizationUtils.add_multiline_to_plot(figure, legend, column_data_source)
        figure.circle(x='x', y='y1', fill_color=line_1_color,line_color=line_1_color, size=circle_size, source=column_data_source)
        figure.circle(x='x', y='y2', fill_color=line_2_color, line_color=line_2_color, size=circle_size, source=column_data_source)
        return figure

    def generate_training_history_graph(self, graph_save_path: 'str, for graph HTML', graph_image_save_path: 'str, for graph PNG', training_loss: list, validation_loss: list) -> 'Bokeh Figure':
        """Generates training history plot, save it as HTML & PNG image and returns Bokeh Figure."""

        self.training_history_plot_title = self.training_history_plot_title.format(self.model_name)
        epochs = range(len(training_loss))

        # Initialize the graph
        training_history_plot = DataVisualizationUtils.initialize_graph_figure(
            title = self.training_history_plot_title, 
            axes_labels = self.training_history_plot_axes_labels, 
            plot_height = self.plot_height, 
            plot_width = self.plot_width
        )

        # Prepare CDS for Bokeh figure feed
        column_data_source = DataVisualizationUtils.get_column_data_source(epochs, training_loss, validation_loss)
        # Add multiline plot and circle glyphs to graph
        training_history_plot = DataVisualizationUtils.add_multiline_and_circle_glyph_to_plot(training_history_plot, self.training_history_plot_legend, column_data_source)
        # Save graph as HTML and PNG image
        DataVisualizationUtils.save_bokeh_figure(training_history_plot, graph_save_path, graph_image_save_path)

        return training_history_plot

    def generate_prediction_graph(self, graph_save_path: 'str, for graph HTML', graph_image_save_path: 'str, for graph PNG', mse_validation: float, prediction_days_for_validation: int, val_y: list, pred_val_y: list, mse_test: float, prediction_days_for_test: int, test_y: list, pred_test_y: list) -> 'Bokeh Figure':
        """Generates Prediction plot over Validation & Testing data, combines them, saves as HTML & PNG image and returns Bokeh Figure.
        
        Needs optimization, code is redundant and can further be reduced!
        """

        self.validation_prediction_plot_title = self.validation_prediction_plot_title.format(self.model_name, mse_validation)
        self.testing_prediction_plot_title = self.testing_prediction_plot_title.format(self.model_name, mse_test)

        # Initialize the graph for validation data predictions
        validation_plot = DataVisualizationUtils.initialize_graph_figure(
            title=self.validation_prediction_plot_title, 
            axes_labels = self.prediction_plot_axes_labels, 
            plot_height = self.plot_height, 
            plot_width = self.plot_width
        )
        # Initialize the graph for testing data predictions
        testing_plot = DataVisualizationUtils.initialize_graph_figure(
            title=self.testing_prediction_plot_title, 
            axes_labels = self.prediction_plot_axes_labels, 
            plot_height = self.plot_height, 
            plot_width = self.plot_width
        )

        # Prepare CDS for both data
        validation_source = DataVisualizationUtils.get_column_data_source(prediction_days_for_validation, val_y, pred_val_y)
        test_source = DataVisualizationUtils.get_column_data_source(prediction_days_for_test, test_y, pred_test_y)

        # Prepare plots for both data
        validation_plot = DataVisualizationUtils.add_multiline_and_circle_glyph_to_plot(validation_plot, self.prediction_plot_legend, validation_source)
        testing_plot = DataVisualizationUtils.add_multiline_and_circle_glyph_to_plot(testing_plot, self.prediction_plot_legend, test_source)

        # Combine both plots
        combined_graph_layout = column(validation_plot, testing_plot)
        # Save graph as HTML and PNG image
        DataVisualizationUtils.save_bokeh_figure(combined_graph_layout, graph_save_path, graph_image_save_path)

        return combined_graph_layout