import numpy as np

def compute_alpha(M, Mbar, W, dx):

    diff = M[:, 0] - Mbar[:, 0]  # Np
    
    alpha_before_derivative = np.zeros((W.shape[0], 3))
    for i in range(3):
        alpha_before_derivative[:, i] = W @ (diff * Mbar[:, i])  # Nc
    
    # Derivative
    alpha = np.zeros_like(alpha_before_derivative)
    for i in range(3):
        alpha[1:-1, i] = (alpha_before_derivative[2:, i] - alpha_before_derivative[:-2, i]) / (2 * dx)
        alpha[0, i] = (alpha_before_derivative[1, i] - alpha_before_derivative[0, i]) / dx
        alpha[-1, i] = (alpha_before_derivative[-1, i] - alpha_before_derivative[-2, i]) / dx
    
    return alpha


def compute_cbar(M, Mbar, U, W, dx, m, k, Tbar, tau):

    Nc = W.shape[0]
    diff = M[:, 0] - Mbar[:, 0]  # Np

    term_before_derivative = np.zeros((Nc, 3, 3))
    for i in range(3):
        for j in range(3):
            term_before_derivative[:, i, j] = W @ (diff * Mbar[:, i] * Mbar[:, j])

    derivative = np.zeros_like(term_before_derivative)
    for i in range(3):
        for j in range(3):
            derivative[1:-1, i, j] = (term_before_derivative[2:, i, j] - term_before_derivative[:-2, i, j]) / (2 * dx)
            derivative[0, i, j] = (term_before_derivative[1, i, j] - term_before_derivative[0, i, j]) / dx
            derivative[-1, i, j] = (term_before_derivative[-1, i, j] - term_before_derivative[-2, i, j]) / dx

    # dUdx
    dUdx = np.zeros(Nc)
    dUdx[1:-1] = (U[2:, 0] - U[:-2, 0]) / (2 * dx)
    dUdx[0] = (U[1, 0] - U[0, 0]) / dx
    dUdx[-1] = (U[-1, 0] - U[-2, 0]) / dx
    
    Lambdabar = 0
    #Lambdabar = -((m / (2 * k * Tbar)) ** 2) / tau

    cbar = np.zeros((Nc, 3, 3))
    for i in range(3):
        for j in range(3):
            cbar[:, i, j] = (m / (2 * k * Tbar)) * derivative[:, i, j]
            if i == j:
                cbar[:, i, j] += (0.5 * dUdx) - (1 / tau) - 15 * m * Lambdabar / (k * Tbar)
    
    return cbar


def compute_gammabar(alpha, M, Mbar, W, dx, m, k, Tbar):
    Nc = W.shape[0]
    diff = M[:, 0] - Mbar[:, 0]  # Np
    Mbar_sq = np.sum(Mbar**2, axis=1)  # Np

    term_before_derivative = np.zeros((Nc, 3))
    for i in range(3):
        term_before_derivative[:, i] = W @ (diff * Mbar[:, i] * Mbar_sq)

    derivative = np.zeros_like(term_before_derivative)
    for i in range(3):
        derivative[1:-1, i] = (term_before_derivative[2:, i] - term_before_derivative[:-2, i]) / (2 * dx)
        derivative[0, i] = (term_before_derivative[1, i] - term_before_derivative[0, i]) / dx
        derivative[-1, i] = (term_before_derivative[-1, i] - term_before_derivative[-2, i]) / dx

    gammabar = np.zeros_like(alpha)
    for i in range(3):
        gammabar[:, i] = -m / (2 * k * Tbar) * alpha[:, i] + (1/10) * (m / (k * Tbar))**2 * derivative[:, i]

    return gammabar


def compute_N(M, Mbar, U, W, dx, m, k, Tbar, tau):
    Nc = W.shape[0]
    N = np.zeros_like(M)

    alpha = compute_alpha(M, Mbar, W, dx)
    cbar = compute_cbar(M, Mbar, U, W, dx, m, k, Tbar, tau)
    gammabar = compute_gammabar(alpha, M, Mbar, W, dx, m, k, Tbar)

    Lambdabar = 0
    #Lambdabar = -((m / (2 * k * Tbar)) ** 2) / tau

    Mbar_sq = np.sum(Mbar**2, axis=1)  # Np

    NiNi = compute_NiNi_analytical(alpha, cbar, gammabar, Lambdabar, tau, Tbar)

    W_col_sum = W.sum(axis=0)  # Np
    W_col_sum = np.where(W_col_sum == 0, 1.0, W_col_sum)  # add this


    alpha_at_particle = (W.T @ alpha) / W_col_sum[:, None]      # Np x 3
    cbar_at_particle = np.einsum('cp,cij->pij', W, cbar) / W_col_sum[:, None, None]  # Np x 3 x 3
    gammabar_at_particle = (W.T @ gammabar) / W_col_sum[:, None]  # Np x 3

    for i in range(3):
        N[:, i] = alpha_at_particle[:, i] + (1/tau) * Mbar[:, i]
        for j in range(3):
            N[:, i] += Mbar[:, j] * cbar_at_particle[:, i, j]
        N[:, i] += gammabar_at_particle[:, i] * (Mbar_sq - 3*k*Tbar/m)
        N[:, i] += Lambdabar * Mbar[:, i] * Mbar_sq
    return N, NiNi

def compute_NiNi_analytical(alpha, cbar, gammabar, Lambdabar, tau, Tbar):

    NiNi = np.zeros_like(alpha)
    
    for i in range(3):
        B = cbar[:, i, :].copy()
        B[:, i] += 1.0 / tau 

        NiNi[:, i] += alpha[:, i]**2                      
        NiNi[:, i] += Tbar * np.sum(B**2, axis=1)       
        NiNi[:, i] += 6 * Tbar**2 * gammabar[:, i]**2   
        NiNi[:, i] += 35 * Lambdabar**2 * Tbar**3      
        NiNi[:, i] += 10 * Lambdabar * Tbar**2 * B[:, i]

    return NiNi


def compute_NiNi_mc(alpha, cbar, gammabar, Lambdabar, tau, Tbar, n_samples=10000):

    Nc = alpha.shape[0]
    NiNi = np.zeros((Nc, 3))
    
    Y = np.random.normal(0, np.sqrt(Tbar), size=(n_samples, 3))
    Y_sq = np.sum(Y**2, axis=1)  
    
    for c in range(Nc):
        for i in range(3):
            Ni = alpha[c, i] + (1/tau) * Y[:, i]
            for j in range(3):
                Ni += cbar[c, i, j] * Y[:, j]
            Ni += gammabar[c, i] * (Y_sq - 3 * Tbar)
            Ni += Lambdabar * Y[:, i] * Y_sq
            
            NiNi[c, i] = np.mean(Ni**2)
    
    return NiNi