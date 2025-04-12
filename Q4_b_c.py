from ConjunctionUtilities import * 
from tudatpy.astro import element_conversion
from Q4_utilities import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ConjunctionUtilities import *
from EstimationUtilities import *
from TudatPropagator import *

bodies = create_simulation_bodies()



ff = read_catalog_file('data/group3/estimated_rso_catalog.pkl')


state_1 = np.array([2490065.1542269904,
       1130965.2261890918,
       6254058.5738026695,
       5872.782407936628,
       4158.268607880216,
       -2924.6268365521933])



swarm_covariance = np.array([
    [7.29171069e-01, -1.97840486e-01, 1.15174447e-01, 9.59242513e-05, 5.77758620e-05, 2.94814291e-04],
    [-1.97840486e-01, 8.08336940e-01, 1.01597981e-01, 7.28157111e-05, 4.24463927e-05, 2.45983192e-04],
    [1.15174447e-01, 1.01597981e-01, 7.96539605e-01, 4.85590119e-05, 5.15089488e-05, -2.15520624e-04],
    [9.59242513e-05, 7.28157111e-05, 4.85590119e-05, 8.76069501e-07, -7.39855501e-08, -8.97438046e-08],
    [5.77758620e-05, 4.24463927e-05, 5.15089488e-05, -7.39855501e-08, 9.21045087e-07, -4.54925911e-08],
    [2.94814291e-04, 2.45983192e-04, -2.15520624e-04, -8.97438046e-08, -4.54925911e-08, 5.61127806e-07]
])

object_covariance_1 = np.array([
    [4.55052686e6, -0.34458423e6, 1.5545267e6, -10.06884573e3, 0.54125231e3, -4.53303147e3],
    [-0.34458423e6, 3.7932542e6, 3.8343132e6, -0.36769873e3, -8.34636111e3, -9.909584e3],
    [1.5545267e6, 3.8343132e6, 4.67728211e6, -4.62569842e3, -8.52841422e3, -12.30611067e3],
    [-10.06884573e3, -0.36769873e3, -4.62569842e3, 36.71721689, -11.65944401, 33.95170352],
    [0.54125231e3, -8.34636111e3, -8.52841422e3, -11.65944401, 69.61481039, 25.69156412],
    [-4.53303147e3, -9.909584e3, -12.30611067e3, 33.95170352, 25.69156412, 76.43503262]
])



mu = bodies.get_body("Earth").gravitational_parameter
kepler_elements_unknown = element_conversion.cartesian_to_keplerian(state_1, mu)
state_SWARM = np.array([-4.89007093e+06,
       -4.01718012e+06,
       2.54384021e+06,
       2.45899194e+03,
       1.50937352e+03,
       7.08247789e+03])
object_dict = dict()
object_dict["mass"] = 260.931020
object_dict["area"] = 1.715124
object_dict["Cd"] = 2.2
object_dict["Cr"] = 1.3

keplerian_swarm = element_conversion.cartesian_to_keplerian(state_SWARM, mu)
keplerian_swarm[2:] = np.rad2deg(keplerian_swarm[2:])
print(keplerian_swarm)
initial_state_swarm_2 = state_SWARM
Cov_SWARM_2 = swarm_covariance

_, state, object_covariance = propagate_state_and_covar(state_1.reshape(-1, 1), object_covariance_1, [796787770.0, 796780800.0], object_dict, bodies=bodies, backward=True)

# Extend the screening window
time_unk, position_unknown = propagate_orbit(state, [796780800.0, 797400000], object_dict, bodies=bodies)
_, position_swarm = propagate_orbit(initial_state_swarm_2, [796780800.0, 797400000], ff[39452], bodies=bodies)

relative_position = position_unknown[:, :3] - position_swarm[:, :3]
relative_distance = np.linalg.norm(relative_position, axis=1)
print(min(relative_distance)) # miss distance in a screening window extended
relative_time = time_unk

# 48 hours screening window
time_T, min_distance = compute_TCA(state, initial_state_swarm_2, [796780800.0, 796780800.0+2*24*3600], object_dict, ff[39452], bodies=bodies)
time_TCA_array = np.array(time_T)
print(time_TCA_array)
min_dist = min(np.array(min_distance))
index = np.where(np.array(min_distance) == min_dist)[0]
time_TCA = time_TCA_array[index]

eigenvalues_SWARM, eigenvectors_SWARM = np.linalg.eig(Cov_SWARM_2)
eigenvalues_object, eigenvectors_object = np.linalg.eig(object_covariance)

Cov_SWARM_2_rem,_,_,_,_ = remediate_covariance(Cov_SWARM_2, 1e-9, eigenvalues_SWARM, eigenvectors_SWARM)
object_covariance_rem,_,_,_,_ = remediate_covariance(object_covariance, 1e-9, eigenvalues_object, eigenvectors_object)

t_S_TCA, state_swarm_TCA, Cov_SWARM_TCA = propagate_state_and_covar(initial_state_swarm_2.reshape(-1, 1), Cov_SWARM_2_rem, [796780800.0, time_TCA], ff[39452], bodies=bodies)
if time_TCA>796787770.0:
    t_U_TCA, state_unk_TCA, Cov_unk_TCA = propagate_state_and_covar(state_1.reshape(-1, 1), object_covariance_1, [796787770.0, time_TCA], object_dict, bodies)
else:
    t_U_TCA, state_unk_TCA, Cov_unk_TCA = propagate_state_and_covar(state_1.reshape(-1, 1), object_covariance_1, [time_TCA, 796787770.0], object_dict, bodies, backward=True)

print(t_S_TCA)
print(t_U_TCA)
print(Cov_SWARM_TCA)
print(Cov_unk_TCA)
Pc = Pc2D_Foster(state_swarm_TCA, Cov_SWARM_TCA, state_unk_TCA, Cov_unk_TCA, (np.sqrt((1.715124)/np.pi) + np.sqrt((2.648)/np.pi)))
print(Pc)
print(min_dist)

# Plot 3D ORBIT
def plot_orbits_with_closest_approach(position_unknown, position_SWARM):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the orbits with enhanced visibility
    ax.plot(position_unknown[:, 0], position_unknown[:, 1], position_unknown[:, 2], label='Unknown', color='blue', linewidth=2.5)
    ax.plot(position_SWARM[:, 0], position_SWARM[:, 1], position_SWARM[:, 2], label='SWARM-A', color='green', linewidth=2.5)


    # Set plot limits based on the maximum extent of both orbits
    max_extent = np.max(np.linalg.norm(np.concatenate((position_unknown, position_SWARM), axis=0), axis=1))
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])

    # Set plot attributes for better visualization
    ax.set_title(f"3D Orbits", fontsize=20, fontweight='bold')
    ax.set_xlabel('X [m]', fontsize=18, labelpad=15)
    ax.set_ylabel('Y [m]', fontsize=18, labelpad=15)
    ax.set_zlabel('Z [m]', fontsize=18, labelpad=15)
    
    # Customize ticks and labels
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_tick_params(direction='in', length=6, width=2, labelsize=14, pad=10, grid_alpha=0.5)
        axis.get_offset_text().set_fontsize(14)

    # Improve legend appearance and positioning
    ax.legend(loc='best', fontsize=17, frameon=True, facecolor='white', edgecolor='black')

    plt.show()


plot_orbits_with_closest_approach(position_unknown[:,:3], position_swarm[:,:3])

t_TCA_10_days = 797325909.0609727

fig = plt.figure(figsize=(14,7))
plt.plot(relative_time/3600, relative_distance/1000)
plt.xlabel('Time [hr]', fontsize = 16, fontweight = 'bold')
plt.ylabel('Relative Distance [km]', fontsize = 16, fontweight = 'bold')
plt.title('Relative distance between O/O Asset and Unknown Object', fontsize = 25, fontweight = 'bold')
plt.show()