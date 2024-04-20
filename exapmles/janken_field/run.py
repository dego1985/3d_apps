import time
import numpy as np
import matplotlib.pyplot as plt

def rr(phi: np.ndarray):
    res = phi.copy()
    res += np.roll(phi, 1, axis=0)
    res += np.roll(phi, -1, axis=0)
    res += np.roll(phi, 1, axis=1)
    res += np.roll(phi, -1, axis=1)
    return res

def update(phi: np.ndarray, N: int):
    x = phi.copy()
    ms = []
    for i in range(N):
       ms.append(x == i)

    for i in range(N):
        phi[rr(ms[(i+1)%N]) * ms[i]] = i+1

    M = 101
    phi += np.random.randint(0,M,phi.shape) == 0

    # phi = (phi-1) % N
    phi = phi % N
    return phi
    
def main() -> int:
    L = 400
    N = 5
    phi = np.random.randint(0,N,(L,L))

    # m0 = phi == 0
    # phi[rr(m0)] = 0

    im = plt.imshow(phi, cmap="jet")
    interval = 1
    for i in range(10000):
        if i % interval == 0:
            im.set_data(phi)
            plt.draw()
            plt.pause(1e-3)
        phi = update(phi, N)

if __name__ == "__main__":
    main()
