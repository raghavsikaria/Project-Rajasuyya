[model_plot_path]: simple_cnn_model_plot_TS_22-06-2020_13-57-20-PM.png
[model_training_hist_path]: simple_cnn_model_training_hist_22-06-2020_13-57-20-PM.png
[model_prediction_path]: simple_cnn_model_prediction_22-06-2020_13-57-52-PM.png
[hpo_plot_convergence_path]: hpo_plot_convergence.png
[hpo_plot_evaluation_path]: hpo_plot_evaluation.png
[hpo_plot_objective_path]: hpo_plot_objective.png
[hpo_plot_regret_path]: hpo_plot_regret.png
[model_hpo_iterations_path]: hpo_iterations.png
# Simple Convolutional Network 

#### TIMESTAMP: 22-06-2020_13-57-20-PM 

## Model Summary 

```txt 
Model: "simple_convolutional_network"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 30, 600)]         0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 15, 24)            28824     
_________________________________________________________________
activation_2 (Activation)    (None, 15, 24)            0         
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 15, 24)            0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 15, 24)            0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 360)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 361       
_________________________________________________________________
activation_3 (Activation)    (None, 1)                 0         
=================================================================
Total params: 29,185
Trainable params: 29,185
Non-trainable params: 0
```

## Model Plot

<span style="display:block;text-align:center">![Simple Convolutional Network][model_plot_path]</span>

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

<span style="display:block;text-align:center">![Simple Convolutional Network][model_training_hist_path]</span>

## Model Predictions & MSE

Interactive plots of the below graphs can be found in HTML files in this model's assets.

<span style="display:block;text-align:center">![Simple Convolutional Network - Predictions - Will be added when predictions are generated][model_prediction_path]</span>