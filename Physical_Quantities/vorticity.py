import numpy as np
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
"""

def get_vorticity(exp_data, x, y, h):
    """
    :param x: and :param y: should be equal to 1d np array of each axis.
    :param h: determines how wide to spread smoothing
    """

    X, Y = np.meshgrid(x, y) #shape: [grid_points_x, grid_points_y]
    
    W = np.exp( (-1/(2*h**2)) * ( (X[None,:,:,None] - exp_data['x'][:,None,None,:])**2 + (Y[None,:,:,None] - exp_data['y'][:,None,None,:])**2 ) ) #shape: [num_snapshots, grid_points_x, grid_points_y, num_particles]
    
    vx = np.sum(W * exp_data['vx'][:,None,None,:], axis=-1) / np.sum(W, axis=-1) #shape: [num_snapshots, grid_points_x, grid_points_y] (Note: Summing over all the particles)
    vy = np.sum(W * exp_data['vy'][:,None,None,:], axis=-1) / np.sum(W, axis=-1)
    
    r_j_minus_r_i = np.concatenate(
        (
            exp_data['x'][:,None,None,:,None] - X[None,:,:,None,None],
            exp_data['x'][:,None,None,:,None] - Y[None,:,:,None,None]
        ), axis=-1
    ) #shape: [num_snapshots, grid_points_x, grid_points_y, num_particles, 2]
    
    v_j_minus_v_i = np.concatenate(
        (
            exp_data['vx'][:,None,None,:,None] - vx[:,:,:,None,None],
            exp_data['vy'][:,None,None,:,None] - vy[:,:,:,None,None]
        ), axis=-1
    ) #shape: [num_snapshots, grid_points_x, grid_points_y, num_particles, 2]

    r_j_minus_r_i_cross_v_j_minus_v_i = np.cross(r_j_minus_r_i, v_j_minus_v_i) #shape: [num_snapshots, grid_points_x, grid_points_y, num_particles]
    
    vorticity = h**(-2) * np.sum(r_j_minus_r_i_cross_v_j_minus_v_i*W, axis=-1) / np.sum(W, axis=-1) #shape: [num_snapshots, grid_points_x, grid_points_y]
    
    return vorticity

#def dump_vtp_vorticity(vorticity):
    """
    Takes as input the output of get_vorticity and enables plotting in paraview
    """
    
    
