import numpy as np
import vtk
"""
Calculating Vorticity with Smoothed Particle Hydrodynamics:

Interpolation of quantity $A$: (Smoothed Particle Hydrodynamics by J. J. Monoghan)
\begin{equation*}
    A\left( \, \vec{r} \, \right) \approx \sum_i A_i \frac{m_i}{\rho_i} W( \left|\vec{r}-\vec{r}_i \right|, h)
\end{equation*}
With density:
\begin{equation*}
    \rho_i = \sum_j m_j W( \left|\vec{r}_j-\vec{r}_i \right|, h)
\end{equation*}
We can rewrite vorticity to include density with calculus identity:
\begin{align*}
    \vec{\omega} &= \nabla \times \vec{v} = \frac{1}{\rho} \left[\nabla \times \left(\rho \vec{v}\right) - \left(\nabla \rho \right) \times \vec{v}\right]\\
\end{align*}
Evaluating some of these terms (for the $i^{th}$ particle located at $\vec{R}_i$):
\begin{align*}
    \nabla \times \left(\rho \vec{v}\right) &= \nabla_{\vec{r}_i} \times \left[ \sum_j \rho \left(\vec{r}_j\right) \vec{v}_j \frac{m_j}{\rho \left(\vec{r}_j \right)} W( \left|\vec{r}_i-\vec{r}_j \right|, h) \right] = \sum_j m_j \nabla_{\vec{r}_i} W( \left|\vec{r}_i-\vec{r}_j \right|, h) \times \vec{v}_j\\
    &= \sum_j m_j \left(\vec{r}_j-\vec{r}_i \right)  \times \vec{v}_j \frac{W( \left|\vec{r}_i-\vec{r}_j \right|, h)}{h^2}\\
    \nabla \rho &= \nabla_{\vec{r}_i} \sum_j m_j W( \left|\vec{r}_i-\vec{r}_j \right|, h)=\sum_j m_j (\vec{r}_j-\vec{r}_i)\frac{W( \left|\vec{r}_i-\vec{r}_j \right|, h)}{h^2}
\end{align*}
Therefore:
\begin{align*}
    \vec{\omega} \left(\vec{r}_i \right) \approx \frac{1}{\rho\left(\vec{r}_i\right)} \sum_j m_j (\vec{r}_j - \vec{r}_i) \times (\vec{v}_j-\vec{v}_i) \frac{W( \left|\vec{r}_i-\vec{r}_j \right|, h)}{h^2}
\end{align*}
Where Gaussian Kernal $W = exp \left(-\frac{1}{2h^2} \left|\vec{r}_i -\vec{r}_j\right|^2 \right) / (2 \pi h^2)$ is used. Note in this implementation, m=1.

To enforce periodic boundary conditions, use this formula for position difference:
\begin{align*}
    D_x = \left[ \left(x_2-x_1+\frac{3L_x}{2} \right)\% L_x \right] - \frac{L_x}{2}
\end{align*}
Where \% denotes the modulo operator. This can be verified by checking each case.
"""

def get_vorticity(exp_data, x, y, h, L, vec_field='velocity', include_density=False):
    """
    :param x: and :param y: should be equal to 1d np array of each axis.
    :param h: determines how wide to spread smoothing
    :param L: Length of the box
    :param vec_field: Either "velocity" or "director", determines which vector field to take vorticity of
    :param include_density: If True, multiplies vorticity by \rho.
    """
    X, Y = np.meshgrid(x, y) #shape: [grid_points_x, grid_points_y]
    """
    Reshape all arrays to [num_snapshots, grid_points_x, grid_points_y, num of particles]. (or any dimension may be 1)
    """
    X = X[None,:,:,None]
    Y = Y[None,:,:,None]
    exp_data_x = exp_data['x'][:,None,None,:]
    exp_data_y = exp_data['y'][:,None,None,:]
    if vec_field == 'velocity':
        exp_data_vx = exp_data['vx'][:,None,None,:]
        exp_data_vy = exp_data['vy'][:,None,None,:]
    elif vec_field == 'director':
        exp_data_vx = exp_data['nx'][:,None,None,:]
        exp_data_vy = exp_data['ny'][:,None,None,:]
    else:
        print('Invalid vector field')
    
    rj_minus_ri_x = exp_data_x-X #j index goes over particles, i index goes over grid points
    rj_minus_ri_y = exp_data_y-Y
  
    rj_minus_ri_x = ((rj_minus_ri_x + (3*L/2)) % L) - (L/2) #enforce periodic boundary conditions
    rj_minus_ri_y = ((rj_minus_ri_y + (3*L/2)) % L) - (L/2)
    
    """
    Start calculation
    """
    W = np.exp((-1/(2*h**2)) * ((rj_minus_ri_x)**2+(rj_minus_ri_y)**2))
    
    #vx and vy are the interpolated velocities at the grid points
    vx = np.sum(W*exp_data_vx, axis=-1) / np.sum(W, axis=-1) #shape: [num_snapshots, grid_points_x, grid_points_y] 
    vy = np.sum(W*exp_data_vy, axis=-1) / np.sum(W, axis=-1) # (Note: Summing over all the particles)
    
    rj_minus_ri = np.stack((rj_minus_ri_x, rj_minus_ri_y),axis=-1) #[num_snapshots, grid_points_x, grid_points_y, num of particles,2]
    vj_minus_vi = np.stack((exp_data_vx - vx[...,None], exp_data_vy - vy[...,None]), axis=-1)

    rj_minus_ri_cross_vj_minus_vi = np.cross(rj_minus_ri, vj_minus_vi) #shape: [num_snapshots, grid_points_x, grid_points_y, num of particles
    
    if not include_density:
        vorticity = h**(-2) * np.sum(rj_minus_ri_cross_vj_minus_vi*W, axis=-1)/ np.sum(W, axis=-1) #shape: [num_snapshots, grid_points_x, grid_points_y]
    elif include_density:
        vorticity = h**(-2) * np.sum(rj_minus_ri_cross_vj_minus_vi*W, axis=-1) #shape: [num_snapshots, grid_points_x, grid_points_y]
    
    return vorticity   
    

def dump_vtp_vorticity(vorticity):
    """
    Takes as input the output of get_vorticity and enables plotting in paraview
    """
    pass #Add code to do this

    
    
