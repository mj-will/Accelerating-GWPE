
def setup_function_approximator(run_path, device="gpu"):
    """Setup an instace of the function approximator"""
    import sys
    sys.path.append("../nn")
    import gwfa
    gwfa.set_keras_device(device)
    from gwfa.function_approximator import FunctionApproximator
    FA = FunctionApproximator(attr_dict=run_path + "fa.pkl", verbose=0)
    return FA

def setup_bilby_likelihood(duration=2.):
    """Setup a instance of likelihood class for timing"""
    import numpy as np
    from bilby.core.sampler import Sampler, Dynesty
    import bilby
    # Setup
    sampling_frequency = 2048.
    outdir = 'tmp'
    label = 'test'
    np.random.seed(88170235)

    # Set up injection parameters
    injection_parameters = dict(
        mass_1=16., mass_2=15., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
        phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., iota=0.4, psi=2.659,
        phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

    # Fixed arguments passed into the source model
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                              reference_frequency=50., minimum_frequency=20.)

    # Create the waveform_generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments)

    # Set up interferometers
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=injection_parameters['geocent_time'] - 3)
    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=injection_parameters)
    # Set up a PriorDict, which inherits from dict.
    priors = bilby.gw.prior.BBHPriorDict()
    # Add prior for geocent time
    priors['geocent_time'] = bilby.core.prior.Uniform(
        minimum=injection_parameters['geocent_time'] - 1,
        maximum=injection_parameters['geocent_time'] + 1,
        name='geocent_time', latex_label='$t_c$', unit='$s$')
    # Fix parameters
    fixed_parameters =['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'geocent_time']
    for key in fixed_parameters:
        priors[key] = injection_parameters[key]
        #pass

    parameters_to_sample = value = { k : injection_parameters[k] for k in set(injection_parameters) - set(fixed_parameters) }
    # Initialise the likelihood
    likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
            time_marginalization=False,
            distance_marginalization=False,
            phase_marginalization=True)
    priors.fill_priors(likelihood, None)
    sampler = Dynesty(likelihood, priors, outdir, label, injection_parameters=injection_parameters, skip_import_verfication=True, external_sampler="dynesty")


    return sampler, priors, parameters_to_sample
