[model_plot_path]: simple_lstm_model_plot_TS_23-06-2020_11-53-35-AM.png
[model_training_hist_path]: simple_lstm_model_training_hist_23-06-2020_11-53-35-AM.png
[model_prediction_path]: simple_lstm_model_prediction_23-06-2020_11-54-56-AM.png
[hpo_plot_convergence_path]: hpo_plot_convergence.png
[hpo_plot_evaluation_path]: hpo_plot_evaluation.png
[hpo_plot_objective_path]: hpo_plot_objective.png
[hpo_plot_regret_path]: hpo_plot_regret.png
[model_hpo_iterations_path]: hpo_iterations.png
# Simple LSTM Network 

#### TIMESTAMP: 23-06-2020_11-53-35-AM 

## Model Summary 

```txt 

Model: "simple_lstm_network"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 30, 600)]         0         
_________________________________________________________________
lstm (LSTM)                  (None, 300)               1081200   
_________________________________________________________________
activation (Activation)      (None, 300)               0         
_________________________________________________________________
dropout (Dropout)            (None, 300)               0         
_________________________________________________________________
dense (Dense)                (None, 1)                 301       
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0         
=================================================================
Total params: 1,081,501
Trainable params: 1,081,501
Non-trainable params: 0
_________________________________________________________________

``` 

## Model Plot

<span style="display:block;text-align:center">![Simple LSTM Network][model_plot_path]</span>

## Hyperparameter Optimization

### Iterations & Results

<span style="display:block;text-align:center">![HPO Iterations Table to be added when HPO is conducted][model_hpo_iterations_path]</span>

### Convergence Plot

<span style="display:block;text-align:center">![convergence plot to be added when HPO is conducted][hpo_plot_convergence_path]</span>

### Evaluation Plot

<span style="display:block;text-align:center">![evaluation plot to be added when HPO is conducted][hpo_plot_evaluation_path]</span>

### Objective Plot

<span style="display:block;text-align:center">![objective plot to be added when HPO is conducted][hpo_plot_objective_path]</span>

### Regret Plot

<span style="display:block;text-align:center">![regret plot to be added when HPO is conducted][hpo_plot_regret_path]</span>

## Model Training History

<span style="display:block;text-align:center">![Simple LSTM Network][model_training_hist_path]</span>

## Model Predictions & MSE

Interactive plots of the below graphs can be found in HTML files in this model's assets.

<span style="display:block;text-align:center">![Simple LSTM Network - Predictions - Will be added when predictions are generated][model_prediction_path]</span>