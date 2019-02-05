#!/usr/bin/env python

# Iterate through a set of model configurations and run nn with that configuration

import os
import json

def update_model(params, model_path='auto_model.json'):
    """
    Update a model file with a new architecture

    Args:
      params (dict): Dictionary of parameters to update
    """

    with open(model_path, 'r') as model_file:
        model_params = json.load(model_file)
    # check specified parameters are model parameters
    for key in params.keys():
        if key in model_params.keys():
            model_params[key] = params[key]
        else:
            print('Key: ' + key + 'is not a parameter in the model file')
    # write changes
    with open(model_path, 'w') as model_file:
        json.dump(model_params, model_file)

def main():
    # parameters to test
    parameters = {'neurons': [600, 750, 900, 1050, 1200], 'layers': [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 21, 24, 27, 30]}
    model_path = 'auto_model.json'
    for n in parameters['neurons']:
        for l in parameters['layers']:
            # update model
            d = {'neurons': n, 'layers': l}
            update_model(d, model_path)
            print('Running nn for {} neurons - {} layers'.format(n, l))
            # run model
            os.system('python nn.py ' +  model_path)

if __name__ == '__main__':
    main()

