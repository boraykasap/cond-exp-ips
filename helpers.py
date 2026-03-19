import numpy as np

def compute_alpha(M, Mbar, Nc, X, L, dx):

    """
    Compute alpha_i for each cell.

    M: Np x 3 array (particle velocities)
    Mbar: Np x 3 array (particle velocities in the parallel process)
    X: Np x 1 array (particle positions)
    Nc: number of cells
    L: domain length
    dx: cell width
    """
    # M is Np x 3b
    alpha_before_derivative = np.zeros((Nc, 3))

    # Indicates which particle belongs to which cell
    cell_index = np.floor((X + L/2) / dx).astype(int)
    cell_index = np.clip(cell_index, 0, Nc-1)

    for c in range(Nc):
        # Access cell by cell
        mask = (cell_index == c)
        if mask.any():
            for i in range(3):
                # j is skipped we skipped those j not equal to 0 
                val = np.mean((M[mask, 0] - Mbar[mask, 0]) * Mbar[mask, i])
                # Update the value
                alpha_before_derivative[c, i] = val
    
    alpha = np.zeros((Nc, 3))
    
    for i in range(3):
        for c in range(1, Nc-1):
            alpha[c, i] = (alpha_before_derivative[c+1, i] - alpha_before_derivative[c-1, i]) / (2 * dx)
        alpha[0, i] = (alpha_before_derivative[1, i] - alpha_before_derivative[0, i]) / (dx)
        alpha[Nc - 1, i] = (alpha_before_derivative[Nc - 1, i] - alpha_before_derivative[Nc - 2, i]) / (dx)
    return alpha

def compute_cbar(M, Mbar, U, Nc, X, L, dx, m, k, Tbar, tau):
    """
    Compute cbar_ij for each cell.

    M: Np x 3 array (particle velocities)
    Mbar: Np x 3 array (particle velocities in the parallel process)
    U: Nc x 3
    X: Np x 1 array (particle positions)
    Nc: number of cells
    L: domain length
    dx: cell width
    m: 
    k: Boltzmann constant
    tau
    
    """
    
    term_before_derivative = np.zeros((Nc, 3, 3))

    # Indicates which particle belongs to which cell
    cell_index = np.floor((X + L/2) / dx).astype(int)
    cell_index = np.clip(cell_index, 0, Nc-1)

    for c in range(Nc):
        # Access cell by cell
        mask = (cell_index == c)
        if mask.any():
            for i in range(3):
                for j in range(3):
                    # l is skipped we skipped those l not equal to 0 
                    # I made it l so that it is not confused with the Boltzmann constant
                    term_before_derivative[c, i, j] = np.mean((M[mask, 0] - Mbar[mask, 0]) * Mbar[mask, i] * Mbar[mask, j])

    derivative = np.zeros_like(term_before_derivative)
    for i in range(3):
        for j in range(3):
            for c in range(1, Nc-1):
                derivative[c, i, j] = (term_before_derivative[c+1, i, j] - term_before_derivative[c-1, i, j]) / (2 * dx)
            derivative[0, i, j] = (term_before_derivative[1, i, j] - term_before_derivative[0, i, j]) / (dx)
            derivative[Nc-1, i, j] = (term_before_derivative[Nc-1, i, j] - term_before_derivative[Nc-2, i, j]) / (dx)

    # This does not depend on i or j
    dUdx = np.zeros(Nc)
    
    for c in range(1, Nc-1):
        dUdx[c] = (U[c+1, 0] - U[c-1, 0]) / (2 * dx)
    dUdx[0] = (U[1, 0] - U[0, 0]) / (dx)
    dUdx[Nc-1] = (U[Nc-1, 0] - U[Nc-2, 0]) / (dx)
    
        
    Lambdabar = - ((m / (2 * k * Tbar)) ** 2) / tau
    cbar = np.zeros((Nc, 3, 3))
    for c in range(Nc):
        for i in range(3):
            for j in range(3):
                cbar[c, i, j] = (m/(2 * k * Tbar)) * derivative[c, i, j] \
                + (i==j) * (  (1/2 * dUdx[c]) - (1 / tau) - 15 * m * Lambdabar / (k * Tbar))
    return cbar

def compute_gammabar(alpha, M, Mbar, Nc, X, L, dx, m, k, Tbar, tau):
    term_before_derivative = np.zeros((Nc, 3))

    cell_index = np.floor((X + L/2) / dx).astype(int)
    cell_index = np.clip(cell_index, 0, Nc-1)
    
    Mbar_sq = Mbar[:,0]**2 + Mbar[:,1]**2 + Mbar[:,2]**2

    for c in range(Nc):
        mask = (cell_index == c)
        if mask.any():
            for i in range(3):
                term_before_derivative[c,i] = np.mean(
                    (M[mask,0] - Mbar[mask,0]) *
                    Mbar[mask,i] *
                    Mbar_sq[mask]
                )

    derivative = np.zeros_like(term_before_derivative)
    for i in range(3):
        for c in range(1, Nc-1):
            derivative[c, i] = (term_before_derivative[c+1, i] - term_before_derivative[c-1, i]) / (2 * dx)
        derivative[0, i] = (term_before_derivative[1, i] - term_before_derivative[0, i]) / (dx)
        derivative[Nc-1, i] = (term_before_derivative[Nc-1, i] - term_before_derivative[Nc-2, i]) / (dx)

    gammabar = np.zeros_like(alpha)

    for c in range(Nc):
        for i in range(3):
            gammabar[c, i] = -m/(2 * k * Tbar) * alpha[c, i] + (1/10) * ((m/(k * Tbar)) ** 2) * derivative[c, i]
            
    return gammabar


def compute_N(M, Mbar, U, Nc, X, L, dx, m, k, Tbar, tau):

    N = np.zeros_like(M)
    
    cell_index = np.floor((X + L/2) / dx).astype(int)
    cell_index = np.clip(cell_index, 0, Nc-1)

    alpha = compute_alpha(M, Mbar, Nc, X, L, dx)
    cbar = compute_cbar(M, Mbar, U, Nc, X, L, dx, m, k, Tbar, tau)
    gammabar = compute_gammabar(alpha, M, Mbar, Nc, X, L, dx, m, k, Tbar, tau)
    Lambdabar = - ((m / (2 * k * Tbar)) ** 2) / tau

    
    for c in range(Nc):
        mask = (cell_index == c)
        if mask.any():
            Mbar_sq = np.sum(Mbar[mask, :]**2, axis=1)
            for i in range(3):
                
                N[mask, i] = alpha[c, i] + (1 / tau) * Mbar[mask, i]
                for j in range(3):
                    N[mask, i] += Mbar[mask, j] * cbar[c, i, j]
                N[mask, i] += gammabar[c, i] * (Mbar_sq - 3 * k * Tbar / m) 
                N[mask, i] += Lambdabar * Mbar[mask, i] * Mbar_sq                    

    return N