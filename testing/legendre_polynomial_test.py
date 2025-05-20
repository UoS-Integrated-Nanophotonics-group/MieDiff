import numpy as np
import matplotlib.pyplot as plt
import torch

z = torch.linspace(-1, 1, 128, dtype=torch.double)
# n = torch.randn(128, dtype=torch.double)

fig, ax = plt.subplots(1, 1)

ns = [1.0, 2.0, 3.0, 4.0]

for i in ns:
    n = torch.tensor(i)
    test = torch.special.legendre_polynomial_p(z, n)
    ax.plot(z.detach(), test.detach())
plt.show()


