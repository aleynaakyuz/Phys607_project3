def Gelman_Rubin(chains):
    J = len(chains)
    L = len(chains[0])
    chain_mean = np.mean(chains)
    grand_mean = np.mean(chain_mean)
    B = L/(J-1)* np.sum((chain_mean - grand_mean)**2)
    for chain in chains:
        s2j = np.var(chain)
    W = np.mean(s2j)
    R = ((L-1)/L * W + B/L)/ W
    return R
