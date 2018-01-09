def lorentz(sample_len=1000, sigma=10, rho=28, beta=8 / 3, step=0.01):
    """This function generates a Lorentz time series of length sample_len,
    with standard parameters sigma, rho and beta.
    """

    x = np.zeros([sample_len])
    y = np.zeros([sample_len])
    z = np.zeros([sample_len])

    # Initial conditions taken from 'Chaos and Time Series Analysis', J. Sprott
    x[0] = 0
    y[0] = -0.01
    z[0] = 9

    for t in range(sample_len - 1):
        x[t + 1] = x[t] + sigma * (y[t] - x[t]) * step
        y[t + 1] = y[t] + (x[t] * (rho - z[t]) - y[t]) * step
        z[t + 1] = z[t] + (x[t] * y[t] - beta * z[t]) * step

    x.shape += (1,)
    y.shape += (1,)
    z.shape += (1,)

    return np.concatenate((x, y, z), axis=1)
