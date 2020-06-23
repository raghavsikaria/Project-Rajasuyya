DATA_PATH = "data/<PATH_TO_DATA>"
DEPENDENT_VARIABLE = "National Stock Exchange: Index: Nifty Bank returns"
FORECAST_PERIOD = 1
CONFIG = {
    "SIMPLE_CNN" : {
        "MODEL_NAME": "simple_convolutional_network",
        "MODEL_NAME_REPORT": "Simple Convolutional Network",
        "NUMBER_OF_OBS_FROM_PAST": 30,
        "TS_BOARD_LOGS": "./assets/simple_cnn/logs",
        "MODEL_PLOT_PATH": "assets/simple_cnn/simple_cnn_model_plot_TS_{}.png",
        "MODEL_SUMMARY_PATH": "assets/simple_cnn/simple_cnn_model_summary_TS_{}.txt",
        "MODEL_SAVE": "assets/simple_cnn/simple_cnn_model_{}",
        "MODEL_OPTIMISED_HYPERPARAMETERS": "assets/simple_cnn/simple_cnn_hpo_result",
        "MODEL_HPO_PLOT_PATHS": ["assets/simple_cnn/hpo_iterations.png", "assets/simple_cnn/hpo_plot_convergence.png", "assets/simple_cnn/hpo_plot_evaluation.png", "assets/simple_cnn/hpo_plot_objective.png", "assets/simple_cnn/hpo_plot_regret.png"],
        "MODEL_PREDICTION_PLOT_PATH_HTML": "assets/simple_cnn/simple_cnn_model_prediction_{}.html",
        "MODEL_PREDICTION_PLOT_PATH_PNG": "assets/simple_cnn/simple_cnn_model_prediction_{}.png",
        "MODEL_TRAINING_HISTORY_PLOT_PATH_HTML": "assets/simple_cnn/simple_cnn_model_training_hist_{}.html",
        "MODEL_TRAINING_HISTORY_PLOT_PATH_PNG": "assets/simple_cnn/simple_cnn_model_training_hist_{}.png",
        "MARKDOWN_PATH": "assets/simple_cnn/simple_cnn_descr_md_{}.md",
        "MARKDOWN_TEMPLATE": """[model_plot_path]: {MODEL_PLOT_PATH}\n[model_training_hist_path]: {MODEL_TRAINING_HIST_PATH}\n[model_prediction_path]: model_prediction_path_key\n[hpo_plot_convergence_path]: model_hpo_plot_convergence_path_key\n[hpo_plot_evaluation_path]: model_hpo_plot_evaluation_path_key\n[hpo_plot_objective_path]: model_hpo_plot_objective_path_key\n[hpo_plot_regret_path]: model_hpo_plot_regret_path_key\n[model_hpo_iterations_path]: model_hpo_iterations_path_key\n# {MODEL_NAME} \n\n#### TIMESTAMP: {TIMESTAMP} \n\n## Model Summary \n\n```txt \n\n{MODEL_SUMMARY}\n\n``` \n\n## Model Plot\n\n<span style="display:block;text-align:center">![{MODEL_NAME}][model_plot_path]</span>\n\n## Hyperparameter Optimization\n\n### Iterations & Results\n\n<span style="display:block;text-align:center">![HPO Iterations Table to be added when HPO is conducted][model_hpo_iterations_path]</span>\n\n### Convergence Plot\n\n<span style="display:block;text-align:center">![convergence plot to be added when HPO is conducted][hpo_plot_convergence_path]</span>\n\n### Evaluation Plot\n\n<span style="display:block;text-align:center">![evaluation plot to be added when HPO is conducted][hpo_plot_evaluation_path]</span>\n\n### Objective Plot\n\n<span style="display:block;text-align:center">![objective plot to be added when HPO is conducted][hpo_plot_objective_path]</span>\n\n### Regret Plot\n\n<span style="display:block;text-align:center">![regret plot to be added when HPO is conducted][hpo_plot_regret_path]</span>\n\n## Model Training History\n\n<span style="display:block;text-align:center">![{MODEL_NAME}][model_training_hist_path]</span>\n\n## Model Predictions & MSE\n\nInteractive plots of the below graphs can be found in HTML files in this model's assets.\n\n<span style="display:block;text-align:center">![{MODEL_NAME} - Predictions - Will be added when predictions are generated][model_prediction_path]</span>""",
        "MARKDOWN_MODEL_PREDICTIONS_KEY": "model_prediction_path_key",
        "MARKDOWN_MODEL_HPO_PLOT_KEYS": ["model_hpo_iterations_path_key", "model_hpo_plot_convergence_path_key", "model_hpo_plot_evaluation_path_key", "model_hpo_plot_objective_path_key", "model_hpo_plot_regret_path_key"],
        "BOKEH":{
            "CREDIT":"raghavsikaria9@gmail.com | https://www.linkedin.com/in/raghavsikaria/",
            "VALIDATION_PREDICTION_TITLE":"{} | Validation Data Prediction - MSE: {}",
            "TESTING_PREDICTION_TITLE":"{} | Test Data Prediction - MSE: {}",
            "PREDICTION_AXES_LABELS":("Days into the future","Actual + Predicted Daily Returns"),
            "PREDICTION_LEGEND_ACTUAL_RETURNS": "Actual Returns",
            "PREDICTION_LEGEND_PREDICTED_RETURNS": "Predicted Returns",
            "TRAINING_HISTORY_TITLE":"{} | Training History",
            "TRAINING_HISTORY_AXES_LABELS":("Epochs","Training + Validation Loss"),
            "TRAINING_HISTORY_LEGEND_TRAINING_LOSS": "Training Loss",
            "TRAINING_HISTORY_LEGEND_VALIDATION_LOSS": "Validation Loss",
            "PLOT_WIDTH":1200,
            "PLOT_HEIGHT":500
        }
    },
    "SIMPLE_LSTM" : {
        "MODEL_NAME": "simple_lstm_network",
        "MODEL_NAME_REPORT": "Simple LSTM Network",
        "NUMBER_OF_OBS_FROM_PAST": 30,
        "TS_BOARD_LOGS": "./assets/simple_lstm/logs",
        "MODEL_PLOT_PATH": "assets/simple_lstm/simple_lstm_model_plot_TS_{}.png",
        "MODEL_SUMMARY_PATH": "assets/simple_lstm/simple_lstm_model_summary_TS_{}.txt",
        "MODEL_SAVE": "assets/simple_lstm/simple_lstm_model_{}",
        "MODEL_OPTIMISED_HYPERPARAMETERS": "assets/simple_lstm/simple_lstm_hpo_result",
        "MODEL_HPO_PLOT_PATHS": ["assets/simple_lstm/hpo_iterations.png", "assets/simple_lstm/hpo_plot_convergence.png", "assets/simple_lstm/hpo_plot_evaluation.png", "assets/simple_lstm/hpo_plot_objective.png", "assets/simple_lstm/hpo_plot_regret.png"],
        "MODEL_PREDICTION_PLOT_PATH_HTML": "assets/simple_lstm/simple_lstm_model_prediction_{}.html",
        "MODEL_PREDICTION_PLOT_PATH_PNG": "assets/simple_lstm/simple_lstm_model_prediction_{}.png",
        "MODEL_TRAINING_HISTORY_PLOT_PATH_HTML": "assets/simple_lstm/simple_lstm_model_training_hist_{}.html",
        "MODEL_TRAINING_HISTORY_PLOT_PATH_PNG": "assets/simple_lstm/simple_lstm_model_training_hist_{}.png",
        "MARKDOWN_PATH": "assets/simple_lstm/simple_lstm_descr_md_{}.md",
        "MARKDOWN_TEMPLATE": """[model_plot_path]: {MODEL_PLOT_PATH}\n[model_training_hist_path]: {MODEL_TRAINING_HIST_PATH}\n[model_prediction_path]: model_prediction_path_key\n[hpo_plot_convergence_path]: model_hpo_plot_convergence_path_key\n[hpo_plot_evaluation_path]: model_hpo_plot_evaluation_path_key\n[hpo_plot_objective_path]: model_hpo_plot_objective_path_key\n[hpo_plot_regret_path]: model_hpo_plot_regret_path_key\n[model_hpo_iterations_path]: model_hpo_iterations_path_key\n# {MODEL_NAME} \n\n#### TIMESTAMP: {TIMESTAMP} \n\n## Model Summary \n\n```txt \n\n{MODEL_SUMMARY}\n\n``` \n\n## Model Plot\n\n<span style="display:block;text-align:center">![{MODEL_NAME}][model_plot_path]</span>\n\n## Hyperparameter Optimization\n\n### Iterations & Results\n\n<span style="display:block;text-align:center">![HPO Iterations Table to be added when HPO is conducted][model_hpo_iterations_path]</span>\n\n### Convergence Plot\n\n<span style="display:block;text-align:center">![convergence plot to be added when HPO is conducted][hpo_plot_convergence_path]</span>\n\n### Evaluation Plot\n\n<span style="display:block;text-align:center">![evaluation plot to be added when HPO is conducted][hpo_plot_evaluation_path]</span>\n\n### Objective Plot\n\n<span style="display:block;text-align:center">![objective plot to be added when HPO is conducted][hpo_plot_objective_path]</span>\n\n### Regret Plot\n\n<span style="display:block;text-align:center">![regret plot to be added when HPO is conducted][hpo_plot_regret_path]</span>\n\n## Model Training History\n\n<span style="display:block;text-align:center">![{MODEL_NAME}][model_training_hist_path]</span>\n\n## Model Predictions & MSE\n\nInteractive plots of the below graphs can be found in HTML files in this model's assets.\n\n<span style="display:block;text-align:center">![{MODEL_NAME} - Predictions - Will be added when predictions are generated][model_prediction_path]</span>""",
        "MARKDOWN_MODEL_PREDICTIONS_KEY": "model_prediction_path_key",
        "MARKDOWN_MODEL_HPO_PLOT_KEYS": ["model_hpo_iterations_path_key", "model_hpo_plot_convergence_path_key", "model_hpo_plot_evaluation_path_key", "model_hpo_plot_objective_path_key", "model_hpo_plot_regret_path_key"],
        "BOKEH":{
            "CREDIT":"raghavsikaria9@gmail.com | https://www.linkedin.com/in/raghavsikaria/",
            "VALIDATION_PREDICTION_TITLE":"{} | Validation Data Prediction - MSE: {}",
            "TESTING_PREDICTION_TITLE":"{} | Test Data Prediction - MSE: {}",
            "PREDICTION_AXES_LABELS":("Days into the future","Actual + Predicted Daily Returns"),
            "PREDICTION_LEGEND_ACTUAL_RETURNS": "Actual Returns",
            "PREDICTION_LEGEND_PREDICTED_RETURNS": "Predicted Returns",
            "TRAINING_HISTORY_TITLE":"{} | Training History",
            "TRAINING_HISTORY_AXES_LABELS":("Epochs","Training + Validation Loss"),
            "TRAINING_HISTORY_LEGEND_TRAINING_LOSS": "Training Loss",
            "TRAINING_HISTORY_LEGEND_VALIDATION_LOSS": "Validation Loss",
            "PLOT_WIDTH":1200,
            "PLOT_HEIGHT":500
        }
    }
}