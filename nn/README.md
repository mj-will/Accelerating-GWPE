# Running the function approximator

## Import function approximator

```
from gwfa.function_approximator import FunctionApproximator
```

## Initialising the approximator

To intialise the approximator need a `.json` file that contains the network configuration. It should look something like this:

```
{
     "layers": 8,
     "mixed_layers": 0,
     "neurons": 500,
     "mixed_neurons": 100,
     "activation": "elu",
     "dropout": 0.0,
     "mixed_dropout": 0.0,
     "batch norm": false,
     "regularization": false,
     "lambda": 0.0001,
     "learning_rate": 0.001,
     "lr_decay": 1e-3,
     "batch_size": 100,
     "epochs": 500,
     "patience": 50,
     "loss": "mean_squared_error",
     "block_size": 10000,
     "blocks": "all",
     "accumulate": "all",
     "notes": "no input split",
     "save": true,
     "datapath": "./data/iota_psi_dist_marg/",
     "outdir": "./outdir/iota_psi_dist_marg/"
 }
```

You also need to specify `input_shape`. This detemines whether the network will split the parameters and have seperate blocks of neurons or simply use a single block. The function expects an `int` or list of ints.

Optionally, you can specifiy the names of parameters (`parameter_names`). This is recommened as they will be used for plotting and aid in understanding the output.

```
FA = FunctionApproximator(input_shape=input_shape, json_file=json_file, parameter_names=parameter_names)
```

## Setting up the normalisation

The function approximator uses the range of the prior values to normalise inputs to $[0, 1]$. To setup the normalisation:

```
FA.setup_normalisation(priors)
```
where `priors` is an array of the prior values.

## Training the approximator

To train the approximator, simply use:

```
FA.train_on_data(x, y, accumulate=True, plot=False)
```

where `x` is an array of sample points, `y` is the target loglikelihood values, `accumulate` is a boolean that enables or disables data accumulation and `plot` enables or disables plotting. Plotting slightly slows down the training process but can help in understanding the networks performance.

## Saving the results and approximator

Once all training has been complete the results can be saved using:

```
FA.save_results()
```

This has two optional arguments:
* `fname`: (string) name of saved file, default: `results.h5`
* `save`: (bool) force save even if false in the .json config file

To save the approximator use:

```
FA.save_approximator(fname="fa.pkl")
```

This saves the attributes dictionary of the class (without the network weights)

## Loading a trained approximator

Once trained the approximator can be loaded from the save file using:

```
FA = FunctionApproximator(attr_dict="/path/to/saved/fa.pkl")
```

Before getting predictions, the weights file must be loaded:

```
FA.load_weights("/path/to/weights/file")
```

## Getting predictions from a trained approximator

The function approximator and its weights must be loaded before prediciting. To predict the value of the function at a given point use:

```
normalised_samples, predictions = FA.predict(samples)
```
