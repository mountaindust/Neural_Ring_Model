'''
Script to plot targets and perception from a given location
'''

import numpy as np
import matplotlib.pyplot as plt

import decision_model as model

focal_loc = np.array([5,10])

targets = model.Targets(geom_name='circle', r=1)
angles = targets.get_percep_angles(focal_loc)

fig = plt.figure(figsize=(12,6))

###### Target Geometry Plot ######
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
elif targets.geom_name == 'circle':
    # plot circle targets
    for n,pos in enumerate(targets.locs):
        try:
            circle = plt.Circle(pos, targets.r[n], color='b')
        except TypeError:
            circle = plt.Circle(pos, targets.r, color='b')
        ax1.add_patch(circle)
    # plot perception angles
    for n, thetas in enumerate(angles):
        r = np.linalg.norm(targets.locs[n,:] - focal_loc)
        for ii in range(2):
            x = (focal_loc[0],focal_loc[0] + r*np.cos(thetas[ii]))
            y = (focal_loc[1],focal_loc[1] + r*np.sin(thetas[ii]))
            ax1.plot(x,y,'k')
else:
    raise NotImplementedError("This geometry still TBD")

ax1.set_aspect('equal')
ax1.set_title('Target Geometry')

###### Perception Signal Plot ######
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
    
elif targets.geom_name == 'circle':
    p_func = np.zeros(2000)
    for thetas in angles:
        for ii in range(2):
            if thetas[ii] < 0:
                thetas[ii] += 2*np.pi
        # step function perception
        theta_bool = np.logical_and(thetas[0]<=theta_mesh,theta_mesh<=thetas[1])
        p_func[theta_bool] = 1

ax2.plot(theta_mesh,p_func)
ax2.set_rmin(-0.5)
ax2.set_rmax(1.25)
ax2.set_rticks([0, 0.5, 1])
ax2.set_rlabel_position(0)
ax2.set_title('Perception Signal')
plt.show()