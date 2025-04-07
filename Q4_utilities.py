import numpy as np
from tudatpy import constants, numerical_simulation
from tudatpy.astro import element_conversion, two_body_dynamics
from tudatpy.data import save2txt
from tudatpy.interface import spice
from tudatpy.numerical_simulation import (
    environment,
    environment_setup,
    estimation_setup,
    propagation,
    propagation_setup,
)


global_frame_orientation = "J2000"
fixed_step_size = 3600.0

def get_lambert_problem_result(
    bodies: environment.SystemOfBodies,
    initial_state: np.array,
    final_state: np.array,
    flight_time: float,
    departure_epoch: float
) -> environment.Ephemeris:
    """
    This function solved Lambert's problem

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    initial_state : numpy array

    final_state : numpy array

    flight_time : float
        Time of flight bewtween the two states

    departure_epoch : float

    Return
    ------
    Ephemeris object defining a purely Keplerian trajectory.
    """

    # Gravitational parameter of the Sun
    central_body_gravitational_parameter = bodies.get_body(
        "Earth"
    ).gravitational_parameter


    # Create Lambert targeter
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3],
        final_state[:3],
        flight_time,
        central_body_gravitational_parameter,
    )

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = np.hstack([initial_state, np.zeros(3)])

    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()

    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(
        lambert_arc_initial_state, central_body_gravitational_parameter
    )

    # Setup Keplerian ephemeris model that describes the Lambert arc
    kepler_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(
            lambert_arc_keplerian_elements,
            departure_epoch,
            central_body_gravitational_parameter,
        ),
        "",  
    )

    return kepler_ephemeris

def propagate_variational_equations(
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    bodies: environment.SystemOfBodies,
    lambert_arc_ephemeris: environment.Ephemeris,
    initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
) -> numerical_simulation.SingleArcVariationalSimulator:
    """
    Propagates the variational equations for a given range of epochs for a perturbed trajectory.

    Parameters
    ----------
    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    initial_state_correction : np.ndarray, default=np.array([0, 0, 0, 0, 0, 0])
        Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    Return
    ------
    Variational equations solver object, from which the state-, state transition matrix-, and
    sensitivity matrix history can be extracted.
    """

    # Compute initial state along Lambert arc
    lambert_arc_initial_state = (
        lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction
    )

    # Get propagator settings
    propagator_settings = get_perturbed_propagator_settings(
        bodies,
        lambert_arc_initial_state,
        initial_time,
        termination_condition,
    )

    # Define parameters for variational equations
    sensitivity_parameters = get_sensitivity_parameter_set(propagator_settings, bodies)

    # Propagate variational equations
    variational_equations_solver = (
        numerical_simulation.create_variational_equations_solver(
            bodies, propagator_settings, sensitivity_parameters
        )
    )

    return variational_equations_solver


def get_sensitivity_parameter_set(
    propagator_settings: propagation_setup.propagator.PropagatorSettings,
    bodies: environment.SystemOfBodies,
) -> numerical_simulation.estimation.EstimatableParameterSet:
    """
    Function creating the parameters for which the variational equations are to be solved.

    Parameters
    ----------
    propagator_settings : propagation_setup.propagator.PropagatorSettings
        Settings used for the propagation of the dynamics

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    Return
    ------
    Propagation settings of the unperturbed trajectory.
    """
    parameter_settings = estimation_setup.parameter.initial_states(
        propagator_settings, bodies
    )

    return estimation_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )

def get_perturbed_propagator_settings(
    bodies: environment.SystemOfBodies,
    initial_state: np.ndarray,
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Creates the propagator settings for a perturbed trajectory.

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    initial_state : np.ndarray
        Cartesian initial state of the vehicle in the simulation

    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    Return
    ------
    Propagation settings of the perturbed trajectory.
    """
    # Define bodies that are propagated, and their central bodies of propagation.
    bodies_to_propagate = ["Spacecraft"]
    central_bodies = ["Earth"]

    # Define accelerations acting on vehicle.
    acceleration_settings_on_vehicle = dict(
        Sun =
        [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.radiation_pressure()
        ],
        Earth=[propagation_setup.acceleration.point_mass_gravity()],
        Venus=[propagation_setup.acceleration.point_mass_gravity()],
        Moon=[propagation_setup.acceleration.point_mass_gravity()],
        Saturn=[propagation_setup.acceleration.point_mass_gravity()],
        Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
        Mars=[propagation_setup.acceleration.point_mass_gravity()],
    )

    # Create global accelerations dictionary.
    acceleration_settings = {"Spacecraft": acceleration_settings_on_vehicle}

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    # Create numerical integrator settings.
    fixed_step_size = 1.0
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78
    )

    dependent_variables_to_save = dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("Spacecraft")
    ]

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        initial_time,
        integrator_settings,
        termination_condition,
        output_variables=dependent_variables_to_save
    )


    return propagator_settings

def create_simulation_bodies() -> environment.SystemOfBodies:
    """
    Creates the body objects required for the simulation, using the
    environment_setup.create_system_of_bodies for natural bodies,
    and manual definition for vehicles

    Parameters
    ----------
    none

    Return
    ------
    Body objects required for the simulation.

    """
    bodies_to_create = ["Earth", "Mars", "Venus", "Sun", "Saturn", "Jupiter", "Moon"]
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
    )   
    body_settings.add_empty_settings("Spacecraft")
    reference_area_radiation = 2.028448  
    radiation_pressure_coefficient = 1.2
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(reference_area_radiation, radiation_pressure_coefficient)
    body_settings.get("Spacecraft").radiation_pressure_target_settings = radiation_pressure_settings
    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies.get_body("Spacecraft").mass = 547.268267 # Mass 547.268267 kg

    return bodies

def compute_bootstrap_CI(data, num_samples):
    # Ensure data is 1-dimensional
    data = data.flatten()

    # Generate bootstrap resamples for each confidence level
    boot_samples_68_low = np.percentile(np.random.choice(data, (num_samples, len(data)), replace=True), 16, axis=1)
    boot_samples_68_high = np.percentile(np.random.choice(data, (num_samples, len(data)), replace=True), 84, axis=1)
    boot_samples_95_low = np.percentile(np.random.choice(data, (num_samples, len(data)), replace=True), 2.5, axis=1)
    boot_samples_95_high = np.percentile(np.random.choice(data, (num_samples, len(data)), replace=True), 97.5, axis=1)
    boot_samples_997_low = np.percentile(np.random.choice(data, (num_samples, len(data)), replace=True), 0.135, axis=1)
    boot_samples_997_high = np.percentile(np.random.choice(data, (num_samples, len(data)), replace=True), 99.865, axis=1)

    # Compute final bootstrapped confidence intervals
    boot_CI = np.zeros((3, 2))
    boot_CI[0, 0] = np.percentile(boot_samples_68_low, 16)   # Lower bound of 68% CI (1-sigma)
    boot_CI[0, 1] = np.percentile(boot_samples_68_high, 84)  # Upper bound of 68% CI (1-sigma)
    boot_CI[1, 0] = np.percentile(boot_samples_95_low, 2.5)    # Lower bound of 95% CI (2-sigma)
    boot_CI[1, 1] = np.percentile(boot_samples_95_high, 97.5)  # Upper bound of 95% CI (2-sigma)
    boot_CI[2, 0] = np.percentile(boot_samples_997_low, 0.135)   # Lower bound of 99.7% CI (3-sigma)
    boot_CI[2, 1] = np.percentile(boot_samples_997_high, 99.865) # Upper bound of 99.7% CI (3-sigma)

    return boot_CI