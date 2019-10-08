
def get_prior_ranges(priors, bilby=True):
    """
    Get the minimium and maximum values of the priors for different input types
    """
    parameters = []
    values = []
    if bilby:
        for parameter, prior in priors.items():
            if not parameter == "mass_ratio":
                parameters.append(parameter)
                if type(prior) is float:
                    values.append((prior, prior))
                else:
                    values.append((prior.minimum, prior.maximum))
    else:
        print("Not implemented")

    return parameters, values
