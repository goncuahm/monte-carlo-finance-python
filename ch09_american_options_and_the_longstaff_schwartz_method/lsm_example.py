import numpy as np

# The 8-path example of Longstaff and Schwartz (2001): K = 1.10, r = 6%, t = 0,1,2,3
S = np.array([[1.00, 1.09, 1.08, 1.34], [1.00, 1.16, 1.26, 1.54],
              [1.00, 1.22, 1.07, 1.03], [1.00, 0.93, 0.97, 0.92],
              [1.00, 1.11, 1.56, 1.52], [1.00, 0.76, 0.77, 0.90],
              [1.00, 0.92, 0.84, 1.01], [1.00, 0.88, 1.22, 1.34]])
K, disc = 1.10, np.exp(-0.06)
cash = np.maximum(K - S[:, 3], 0)                      # cash flows if held to t = 3
tau = np.full(8, 3)
for j in [2, 1]:
    h = np.maximum(K - S[:, j], 0)
    itm = h > 0
    Y = cash * disc**(tau - j)                          # value of continuing
    X = S[itm, j]
    b = np.polyfit(X, Y[itm], 2)[::-1]                  # Y = b0 + b1 X + b2 X^2
    cont = b[0] + b[1] * X + b[2] * X**2
    print(f"t = {j}: regression  E[Y|X] = {b[0]:.3f} + {b[1]:.3f} X + {b[2]:.3f} X^2")
    print("  path  S      exercise  continuation  decision")
    for i, e, c in zip(np.where(itm)[0], h[itm], cont):
        print(f"  {i+1:3d}  {S[i, j]:.2f}   {e:.4f}    {c:8.4f}      "
              f"{'exercise' if e > c else 'continue'}")
    ex = np.where(itm)[0][h[itm] > cont]
    cash[ex], tau[ex] = h[ex], j
print("\ncash flows:", " ".join(f"{c:.2f}@t{t}" for c, t in zip(cash, tau)))
print(f"option value = {np.mean(cash * disc**tau):.4f}   (European put: "
      f"{np.mean(np.maximum(K - S[:, 3], 0)) * disc**3:.4f})")
