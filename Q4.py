import numpy as np
from ConjunctionUtilities import *
from EstimationUtilities import *
from Q4_utilities import *
from scipy.stats import gaussian_kde
from tudatpy.astro import element_conversion

_, measurements_data, sensor_parameters = read_measurement_file("data/group3/q4_meas_iod_99003.pkl")

bodies = create_simulation_bodies()

range_measurements_nominal = np.array([Yk[0] for Yk in measurements_data['Yk_list']])
RA_measurements_nominal = np.array([Yk[1] for Yk in measurements_data['Yk_list']])
DEC_measurements_nominal = np.array([Yk[2] for Yk in measurements_data['Yk_list']])
times = np.array([tk for tk in measurements_data['tk_list']])

radar_position = np.array([Yk[:3] for Yk in sensor_parameters['sensor_ecef']])
earth_rotation_model = bodies.get("Earth").rotation_model
rotation_initial = earth_rotation_model.body_fixed_to_inertial_rotation(times[0])
rotation_final = earth_rotation_model.body_fixed_to_inertial_rotation(times[1])
radar_position_eci_1 = rotation_initial @ radar_position
radar_position_eci_2 = rotation_initial @ radar_position
sigma_range = 10
sigma_RA = 0.0017453292519943296
sigma_DEC = 0.0017453292519943296
range_limit = np.array([Yk for Yk in sensor_parameters['rg_lim']])
RA_limit = np.array([Yk for Yk in sensor_parameters['az_lim']])
DEC_limit = np.array([Yk for Yk in sensor_parameters['el_lim']])

# Initialize variables as DA Variables (given the uncertainty)
range_measurements = np.empty_like(range_measurements_nominal)
RA_measurements = np.empty_like(RA_measurements_nominal)
DEC_measurements = np.empty_like(DEC_measurements_nominal)

nb_montecarlo_iteration = 10000

montecarlo_initial_state = np.zeros((nb_montecarlo_iteration,6))
montecarlo_final_state = np.zeros((nb_montecarlo_iteration,6))

for montecarlo_iteration in range(nb_montecarlo_iteration):

    range_measurements = np.empty_like(range_measurements_nominal)
    RA_measurements = np.empty_like(RA_measurements_nominal)
    DEC_measurements = np.empty_like(DEC_measurements_nominal)
    uniform = False
    normal = True
    if uniform:
        for i in range(len(range_measurements_nominal)):
            range_measurements[i] = np.random.uniform(range_measurements_nominal[i] - sigma_range, range_measurements_nominal[i] + sigma_range)
            RA_measurements[i] = np.random.uniform(RA_measurements_nominal[i] - sigma_RA, RA_measurements_nominal[i] + sigma_RA)
            DEC_measurements[i] = np.random.uniform(DEC_measurements_nominal[i] - sigma_DEC, DEC_measurements_nominal[i] + sigma_DEC)
            if range_measurements[i] < range_limit[0] or RA_measurements[i] < RA_limit[0] or DEC_measurements[i] < DEC_limit[0]:
                continue
            if range_measurements[i] > range_limit[1] or RA_measurements[i] > RA_limit[1] or DEC_measurements[i] > DEC_limit[1]:
                continue
    
    if normal:
        for i in range(len(range_measurements_nominal)):
            range_measurements[i] = np.random.normal(range_measurements_nominal[i], sigma_range)
            RA_measurements[i] = np.random.normal(RA_measurements_nominal[i], sigma_RA)
            DEC_measurements[i] = np.random.normal(DEC_measurements_nominal[i],  sigma_DEC)
            if range_measurements[i] < range_limit[0] or RA_measurements[i] < RA_limit[0] or DEC_measurements[i] < DEC_limit[0]:
                continue
            if range_measurements[i] > range_limit[1] or RA_measurements[i] > RA_limit[1] or DEC_measurements[i] > DEC_limit[1]:
                continue
        

    # Retrieve the corresponding initial and final state


    x1_ecef = radar_position_eci_1[0] + range_measurements[0] * np.cos(RA_measurements[0]) * np.cos(DEC_measurements[0])
    x2_ecef = radar_position_eci_2[0] + range_measurements[1] * np.cos(RA_measurements[1]) * np.cos(DEC_measurements[1])
    y1_ecef = radar_position_eci_1[1] + range_measurements[0] * np.sin(RA_measurements[0]) * np.cos(DEC_measurements[0])
    y2_ecef = radar_position_eci_2[1] + range_measurements[1] * np.sin(RA_measurements[1]) * np.cos(DEC_measurements[1])
    z1_ecef = radar_position_eci_1[2] + range_measurements[0] * np.sin(DEC_measurements[0])
    z2_ecef = radar_position_eci_2[2] + range_measurements[1] * np.sin(DEC_measurements[1])

    initial_state_ecef = np.array([x1_ecef, y1_ecef, z1_ecef]).flatten()
    final_state_ecef = np.array([x2_ecef, y2_ecef, z2_ecef]).flatten()

    initial_state = initial_state_ecef
    final_state = final_state_ecef

    # Solve the Lambert Problem

    lambert_solution = get_lambert_problem_result(bodies, initial_state, final_state, times[1]-times[0], times[0])
    lambert_arc_states = dict()
    for epoch in times:
        lambert_arc_states[epoch] = lambert_solution.cartesian_state(epoch)

    initial_determined_state = np.array(list(lambert_arc_states.values()))[0,:]
    final_determined_state = np.array(list(lambert_arc_states.values()))[1,:]

    # Propagate Variational equations
    termination_settings = propagation_setup.propagator.time_termination(
        times[1], terminate_exactly_on_final_condition=True
    )
    variational_equations_solver = propagate_variational_equations(times[0], termination_settings, bodies, lambert_solution)
    state_transition_matrix_history = (
    variational_equations_solver.state_transition_matrix_history
    )
    state_history = variational_equations_solver.state_history

    propagated_final_state = np.array(list(state_history.values()))[-1,:]

    final_state_transition_matrix = state_transition_matrix_history[times[1]]
    phi_r_v = final_state_transition_matrix[0:3, 3:6]
    final_state_deviation = propagated_final_state - final_determined_state
    delta_v_i = np.linalg.inv(phi_r_v) @ (-final_state_deviation[:3])
    initial_state_correction = np.zeros(6)
    initial_state_correction[3:] = delta_v_i

    lambert_arc_initial_state = (
        lambert_solution.cartesian_state(times[0]) + initial_state_correction
    )
    propagator_settings = get_perturbed_propagator_settings(
        bodies, lambert_arc_initial_state, times[0], termination_settings
    )
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )
    state_history_corrected = dynamics_simulator.propagation_results.state_history
    initial_corrected_state = np.array(list(state_history_corrected.values()))[0,:]
    final_corrected_state = np.array(list(state_history_corrected.values()))[-1,:]


    tolerance = 1e-7
    final_state_deviation = np.linalg.norm(final_corrected_state[:3] - final_determined_state[:3])

    iteration = 0

    while final_state_deviation>tolerance and iteration<100:

        variational_equations_solver = propagate_variational_equations(times[0], termination_settings, bodies, lambert_solution, initial_state_correction=initial_state_correction)
        state_transition_matrix_history = (
        variational_equations_solver.state_transition_matrix_history
        )
        state_history = variational_equations_solver.state_history

        propagated_final_state = np.array(list(state_history.values()))[-1,:]

        final_state_transition_matrix = state_transition_matrix_history[times[1]]
        phi_r_v = final_state_transition_matrix[0:3, 3:6]
        final_state_deviation = propagated_final_state - final_determined_state
        delta_v_i = np.linalg.inv(phi_r_v) @ (-final_state_deviation[:3])

        initial_state_correction[3:] = initial_state_correction[3:] + delta_v_i

        lambert_arc_initial_state = (
            lambert_solution.cartesian_state(times[0]) + initial_state_correction
        )

        propagator_settings = get_perturbed_propagator_settings(
            bodies, lambert_arc_initial_state, times[0], termination_settings
        )

        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings
        )

        state_history_corrected = dynamics_simulator.propagation_results.state_history
        initial_corrected_state = np.array(list(state_history_corrected.values()))[0,:]
        final_corrected_state = np.array(list(state_history_corrected.values()))[-1,:]

        final_state_deviation = np.linalg.norm(final_corrected_state[:3] - final_determined_state[:3])

        iteration += 1
    print(montecarlo_iteration)
    
    montecarlo_final_state[montecarlo_iteration,:] = final_corrected_state
    montecarlo_initial_state[montecarlo_iteration,:] = initial_corrected_state


def plot_histogram_with_boot_CI(data_list, boot_CI_list, xlabel_text, title_text):
    """
    Plots a single figure with 6 subplots (3x2), each showing a histogram with KDE and bootstrapped CIs.
    """
    # Create a single figure with 6 subplots (3x2)
    fig, axes = plt.subplots(3, 2, figsize=(11.7, 16.5/1.5)) 
    fig.suptitle(title_text, fontsize=24, fontweight='bold')
    axes = axes.flatten()  
    mean = []
    # Loop through each dataset and plot on a separate subplot
    for i, (data, boot_CI) in enumerate(zip(data_list, boot_CI_list)):

        ax = axes[i]

        # Plot the histogram
        ax.hist(data, bins=30, edgecolor='black', color='yellow', alpha=0.7, label='Data Histogram')

        # Kernel Density Estimation (KDE)
        kde = gaussian_kde(data)
        xi = np.linspace(min(data), max(data), 1000)
        f = kde(xi)

        # Histogram scaling factor
        hist, _ = np.histogram(data, bins=30)
        scale_factor = max(hist)

        # Normalize the KDE to the histogram
        f_scaled = f * scale_factor / max(f)
        ax.plot(xi, f_scaled, 'k-', linewidth=3, label='PDF (KDE)')

        # Plot the mean as a black dotted line
        mean_value = np.mean(data)
        mean.append(mean_value)
        ax.axvline(mean_value, color='k', linestyle=':', linewidth=3, label='Mean')

        # Bootstrapped Confidence Intervals (CIs)

        ax.axvline(boot_CI[0, 0], color='red', linestyle='-', linewidth=4, label='68% CI (1-$\sigma$)')
        ax.axvline(boot_CI[0, 1], color='red', linestyle='-', linewidth=4)
        ax.axvline(boot_CI[1, 0], color='green', linestyle='-.', linewidth=3, label='95% CI (2-$\sigma$)')
        ax.axvline(boot_CI[1, 1], color='green', linestyle='-.', linewidth=3)
        ax.axvline(boot_CI[2, 0], color='blue', linestyle='--', linewidth=3, label='99.7% CI (3-$\sigma$)')
        ax.axvline(boot_CI[2, 1], color='blue', linestyle='--', linewidth=3)

        # Adding labels and title for each subplot
        ax.set_xlabel(xlabel_text[i], fontsize=14, fontweight='bold')
        ax.set_ylabel('Nb of occurrences', fontsize=14, fontweight='bold')
        
        # Set tick parameters for better readability
        ax.tick_params(axis='both', which='both', direction='in', length=6, width=2, 
                      grid_alpha=0.5, labelsize=14)
        ax.xaxis.get_offset_text().set_fontsize(14)

        if i == 0:
            ax.legend(loc='best', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2, w_pad=2)
    plt.savefig('histograms.png')
    return mean

cov_matrix = np.cov(montecarlo_initial_state, rowvar=False)

for i in [0,1,2]:
    for j in [0,1,2]:
        cov_matrix[i,j] = cov_matrix[i,j] * (10**(-6))
for i in [0,1,2]:
    for j in [3,4,5]:
        cov_matrix[i,j] = cov_matrix[i,j] * (10**(-3))
for j in [0,1,2]:
    for i in [3,4,5]:
        cov_matrix[i,j] = cov_matrix[i,j] * (10**(-3))
    

np.savetxt('cov_matrix.txt', cov_matrix, fmt='%.9f', delimiter=',', header='Covariance Matrix', comments='')

data_list = [
    montecarlo_initial_state[:,0], 
    montecarlo_initial_state[:,1], 
    montecarlo_initial_state[:,2], 
    montecarlo_initial_state[:,3], 
    montecarlo_initial_state[:,4], 
    montecarlo_initial_state[:,5]
]

boot_CI_list = []
for i in range(len(montecarlo_initial_state[0, :])):
    boot_CI = compute_bootstrap_CI(montecarlo_initial_state[:, i], 100)
    boot_CI_list.append(boot_CI) 

xlabel_list = [
    'X [m]', 
    'Y [m]', 
    'Z [m]', 
    'Vx [m/s]', 
    'Vy [m/s]', 
    'Vz [m/s]'
]

mean = plot_histogram_with_boot_CI(data_list, boot_CI_list, xlabel_list, 'Histograms for the Initial Inertial Cartesian State components')

print('--------------------------------------')
print('--------------------------------------')
print('The covariance matrix for the initial inertial state is:')
print(cov_matrix)
print('--------------------------------------')
print('--------------------------------------')
print('--------------------------------------')
print('--------------------------------------')
print('The mean initial inertial state is:')
print(mean)
print('--------------------------------------')
print('--------------------------------------')
earth_mu = bodies.get_body('Earth').gravitational_parameter
mean_kepler_elements = element_conversion.cartesian_to_keplerian(mean, earth_mu)
for i in [2,3,4,5]:
    mean_kepler_elements[i] = np.rad2deg(mean_kepler_elements[i])
print('--------------------------------------')
print('--------------------------------------')
print('The mean initial Keplerian state is:')
for value in mean_kepler_elements:
    print(f'{value:.15f}')  # Printing with 15 digits of precision
print('--------------------------------------')
print('--------------------------------------')
