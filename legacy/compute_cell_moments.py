def compute_cell_moments_NW(M, W, m, k, Tbar, Mbar=None, Ubar=None):
    Nc = W.shape[0]
    if Mbar is not None:
        W_col_sum = W.sum(axis=0)  # Np
        Ubar_at_X = (W.T @ Ubar) / W_col_sum[:, None]  # Np x 3
        M_corrected = M - Mbar + Ubar_at_X
    else:
        M_corrected = M
    
    U = W @ M_corrected             # Nc x 3
    
    T = np.zeros(Nc)
    for c in range(Nc):
        residuals = M_corrected - U[c]
        T[c] = (m / (3*k)) * (W[c] * (residuals**2).sum(axis=1)).sum()
        if Mbar is not None and Tbar is not None:
            T[c] += Tbar
    
    return U, T

def compute_cell_moments(M, W, m, k, Tbar, Mbar=None, Ubar=None):
    Nc = W.shape[0]
    
    if Mbar is not None:
        W_col_sum = W.sum(axis=0)
        W_col_sum = np.where(W_col_sum == 0, 1.0, W_col_sum)  

        Ubar_at_X = (W.T @ Ubar) / W_col_sum[:, None]
        M_corrected = M - Mbar + Ubar_at_X
    else:
        M_corrected = M
    
    # U from variance-reduced estimator
    U = W @ M_corrected 
    
    # T from raw M — NOT M_corrected
    T = np.zeros(Nc)
    for c in range(Nc):
        U_raw = W[c] @ M          # raw mean
        residuals = M - U_raw     # raw fluctuations
        T[c] = (m/(3*k)) * (W[c] * (residuals**2).sum(axis=1)).sum()
    
    return U, T





if correlated_process:    
    N, NiNi = compute_N(M, Mbar, U[step-1], W, dx, m, k, Tbar, tau)

    if step in [1, 10, 50, 100, 200, 500, 750]:
        alpha = compute_alpha(M, Mbar, W, dx)
        cbar = compute_cbar(M, Mbar, U[step-1], W, dx, m, k, Tbar, tau)
        
        print(f"\n{'='*50}")
        print(f"Step {step}")
        print(f"{'='*50}")
        
        print(f"\n  N statistics:")
        for d, name in enumerate(['x','y','z']):
            print(f"    N[{name}]: mean={np.mean(N[:,d]):+.4f}  std={np.std(N[:,d]):.4f}  max={np.abs(N[:,d]).max():.4f}")
        
        print(f"\n  Particle statistics:")
        print(f"    std(Mbar[:,0]) = {np.std(Mbar[:,0]):.4f}")
        print(f"    std(M-Mbar)    = {np.std(M[:,0]-Mbar[:,0]):.4f}")
        print(f"    mean(M-Mbar)   = {np.mean(M[:,0]-Mbar[:,0]):.4f}")
        
        print(f"\n  Coefficient statistics:")
        print(f"    alpha: mean={np.mean(alpha):+.4f}  std={np.std(alpha):.4f}  max={np.abs(alpha).max():.4f}")
        print(f"    cbar:  mean={np.mean(cbar):+.4f}  std={np.std(cbar):.4f}  max={np.abs(cbar).max():.4f}")
        print(f"    cbar diagonal only:")
        for d in range(3):
            print(f"      cbar[{d},{d}]: mean={np.mean(cbar[:,d,d]):+.4f}  std={np.std(cbar[:,d,d]):.4f}")
        
        # Energy correction health check
        NiNi = compute_NiNi_analytical(alpha, cbar, 
                compute_gammabar(alpha, M, Mbar, W, dx, m, k, Tbar),
                -((m/(2*k*Tbar))**2)/tau, tau, Tbar)
        Atilde = (k*Tbar/m)*(1-np.exp(-2*dt/tau)) - dt**2 * NiNi
        print(f"\n  Energy correction A_tilde:")
        print(f"    min={Atilde.min():.6f}  mean={Atilde.mean():.6f}")
        pct_negative = 100 * np.mean(Atilde < 0)
        print(f"    % negative (clamped to 0): {pct_negative:.1f}%")
        print(f"{'='*50}\n")

        print(f"U[step,:,0] (wall-normal, should be ~0):")
        print(U[step-1,:,0])
        print(f"U[step,:,1] (streamwise, should be linear):")  
        print(U[step-1,:,1])



def compute_cell_moments(M, cell_index, Nc, Mbar=None, Ubar=None):
    U = np.zeros((Nc, 3))
    T = np.zeros(Nc)
    
    if Mbar is not None:
        if Ubar is None:
            raise ValueError("If Mbar is provided, Ubar must also be provided")
        M_corrected = M - Mbar + Ubar[cell_index]  # Np x 3
    else:
        M_corrected = M

    for c in range(Nc):
        mask = cell_index == c
        if mask.any():
            V_cell = M_corrected[mask]
            U[c] = V_cell.mean(axis=0)
            
            residuals = V_cell - U[c]
            T[c] = (m / (3*k)) * (residuals**2).sum(axis=1).mean()
            
            if Mbar is not None:
                T[c] += Tbar
    
    return U, T