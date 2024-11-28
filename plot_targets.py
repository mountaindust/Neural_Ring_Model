'''
Script to plot targets and perception from a given location
'''

import numpy as np
import matplotlib.pyplot as plt

import decision_model as model

focal_loc = np.array([5,10])

targets = model.Targets()
angles = targets.get_percep_angles(focal_loc)

fig = plt.figure(figsize=(12,6))
ax1 = plt.subplot(121)

if targets.geom_name is None:
    # delta functions
    ax1.plot(targets.locs[:,0],targets.locs[:,1],'.')
    # plot perception angles
    for n, theta in enumerate(angles):
        r = np.linalg.norm(targets.locs[n,:] - focal_loc)
        x = (focal_loc[0],focal_loc[0] + r*np.cos(theta))
        y = (focal_loc[1],focal_loc[1] + r*np.sin(theta))
        ax1.plot(x,y,'k')
else:
    raise NotImplementedError("This geometry still TBD")

ax1.set_aspect('equal')
ax1.set_title('Target Geometry')

ax2 = plt.subplot(122, projection='polar')

theta_mesh = np.linspace(0, 2*np.pi, 2000)
if targets.geom_name is None:
    p_func = np.zeros(2000)
    for theta in angles:
        if theta < 0:
            theta += 2*np.pi
        idx = np.searchsorted(theta_mesh,theta)
        if (theta-theta_mesh[idx-1]) < (theta_mesh[idx]-theta):
            p_func[idx-1] = 1
        else:
            p_func[idx] = 1
    ax2.plot(theta_mesh,p_func)

ax2.set_rmin(-0.5)
ax2.set_rmax(1.25)
ax2.set_rticks([0, 0.5, 1])
ax2.set_rlabel_position(0)
ax2.set_title('Perception Signal')
plt.show()