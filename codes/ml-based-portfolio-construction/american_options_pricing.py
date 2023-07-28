import numpy as np


def american_option_binomial_tree(S, K, T, r, v, q, N, option_type):
    """
    Parameters:
        S: Underlying asset price
        K: Option strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        v: Volatility of the underlying asset
        q: Dividend yield
        N: Number of binomial steps
        option_type: 'call' or 'put'

    Returns the American option price
    """
    deltaT = T / N
    u = np.exp(v * np.sqrt(deltaT))
    d = 1 / u
    p = (np.exp((r - q) * deltaT) - d) / (u - d)

    # Initialize our f_{i,j} tree with zeros
    fs = np.zeros((N + 1, N + 1))

    # Compute the leaves, f_{N,j}
    for j in range(N + 1):
        fs[N][j] = max(
            (S * (u**j) * (d ** (N - j)) - K, 0)
            if option_type == "call"
            else (K - S * (u**j) * (d ** (N - j)), 0)
        )

    # Compute backward the rest of the tree
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            fs[i][j] = np.exp(-r * deltaT) * (
                p * fs[i + 1][j + 1] + (1 - p) * fs[i + 1][j]
            )
            fs[i][j] = max(
                fs[i][j],
                max(
                    (S * (u**j) * (d ** (i - j)) - K, 0)
                    if option_type == "call"
                    else (K - S * (u**j) * (d ** (i - j)), 0)
                ),
            )

    return fs[0][0]


# Test the function
S = 2500  # underlying asset price
K = 1750  # option strike price
T = 1  # time to maturity
r = 0.05  # risk-free rate
v = 0.3  # volatility of underlying asset
q = 0  # dividend yield
N = 100  # number of steps
option_type = "put"  # 'call' or 'put'

price = american_option_binomial_tree(S, K, T, r, v, q, N, option_type)
print("The price of the American {} option is {:.2f}".format(option_type, price))
