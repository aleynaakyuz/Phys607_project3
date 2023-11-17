# This is a code for 2D mcmc

import numpy as np
import tqdm
from matplotlib import pyplot as plt

    
def mcmc(initial, post, prop, iterations):
    x = [initial]
    p = [post(x[-1])]
    for i in tqdm.tqdm(range(iterations)):
        x_test = prop(x[-1])
        p_test = post(x_test)

        acc = p_test / p[-1]
        u = np.random.uniform(0, 1)
        if u <= acc:
            x.append(x_test)
            p.append(p_test)
    return np.array(x), np.array(p)
    