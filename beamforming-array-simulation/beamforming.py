import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
# -------------------------------
# Antenna parameters
# -------------------------------

N = 8               # number of antenna elements
d = 0.5             # spacing (in wavelengths)
k = 2*np.pi         # wave number

theta = np.linspace(-np.pi/2, np.pi/2, 1000)


# -------------------------------
# Array Factor Function
# -------------------------------

def array_factor(theta, beta):

    AF = np.zeros(len(theta), dtype=complex)

    for n in range(N):
        AF += np.exp(1j*n*(k*d*np.cos(theta)+beta))

    AF = np.abs(AF)
    AF = AF / np.max(AF)

    return AF


# -------------------------------
# Beam Direction
# -------------------------------

theta0 = np.deg2rad(30)   # steer beam to 30 degrees

beta = -k*d*np.cos(theta0)

AF = array_factor(theta, beta)


# -------------------------------
# Plot Radiation Pattern
# -------------------------------

plt.figure()

plt.plot(np.rad2deg(theta), AF)

plt.xlabel("Angle (degrees)")
plt.ylabel("Normalized Array Factor")

plt.title("Beamforming Radiation Pattern")

plt.grid()

plt.savefig("results/beam_pattern.png")

plt.show()
# -------------------------------
# Polar Radiation Pattern
# -------------------------------

plt.figure()

ax = plt.subplot(111, polar=True)

ax.plot(theta, AF)

ax.set_title("Polar Radiation Pattern")

plt.savefig("results/polar_pattern.png")

plt.show() 
# -------------------------------
# Beam Steering Animation
# -------------------------------

fig, ax = plt.subplots()

line, = ax.plot([], [])

ax.set_xlim(-90, 90)
ax.set_ylim(0, 1)

ax.set_xlabel("Angle (degrees)")
ax.set_ylabel("Normalized Array Factor")

ax.set_title("Beam Steering Animation")


def update(frame):

    theta0 = np.deg2rad(frame)

    beta = -k*d*np.cos(theta0)

    AF = array_factor(theta, beta)

    line.set_data(np.rad2deg(theta), AF)

    return line,


ani = FuncAnimation(fig, update, frames=np.arange(-60,60,2), interval=100)
ani.save("results/beam_animation.gif", writer="pillow")

plt.show()

# -------------------------------
# Radiation Heatmap
# -------------------------------

theta_scan = np.linspace(-np.pi/2, np.pi/2, 200)

AF_map = []

for ang in theta_scan:

    beta = -k*d*np.cos(ang)

    AF = array_factor(theta, beta)

    AF_map.append(AF)

AF_map = np.array(AF_map)

plt.figure()

plt.imshow(AF_map,
           extent=[-90,90,-90,90],
           aspect='auto',
           cmap='inferno')

plt.xlabel("Observation Angle (deg)")
plt.ylabel("Beam Steering Angle (deg)")

plt.title("Beamforming Radiation Heatmap")

plt.colorbar(label="Normalized Power")

plt.savefig("results/heatmap.png")

plt.show()

# -------------------------------
# 3D Radiation Pattern
# -------------------------------

theta_3d = np.linspace(0, np.pi, 180)
phi_3d = np.linspace(0, 2*np.pi, 180)

THETA, PHI = np.meshgrid(theta_3d, phi_3d)

# Use array factor as radius
R = np.abs(np.sin(THETA))

# Convert spherical → cartesian
X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
Z = R * np.cos(THETA)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='turbo')

ax.set_title("3D Radiation Pattern")

plt.savefig("results/radiation_3d.png")

plt.show()
