# This is a code for 1D mcmc

import numpy as np
import tqdm


def mcmc(initial, data, post, prop, iterations):
    x = [initial]
    p = [post(data, x[-1])]
    for i in tqdm(range(iterations)):
        x_test = prop(x[-1])
        p_test = post(data, x_test)

        acc = p_test / p[-1]
        u = np.random.uniform(0, 1)
        if u <= acc:
            x.append(x_test)
            p.append(p_test)
    return x, p
    