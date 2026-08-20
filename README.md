[README.md](https://github.com/user-attachments/files/31249098/README.md)
# Numerical data and code for *Emptiness formation in the Lieb--Liniger gas*

This repository contains the numerical code and data associated with the numerical hydrodynamic calculations in the paper

**“Emptiness formation in the Lieb--Liniger gas: hydrodynamic instantons and a conjectured rate function.”**

## Contents

- 'hydrodynamic_results.csv' — numerical results from the hydrodynamic minimization.
- 'templateCode.py' — template code with adjustable parameters used to solve the Lieb--Liniger equation of state and minimize the Euclidean hydrodynamic action.

The calculation is performed in units with $R=1$ and $\rho_0=1$.

## Numerical data

The file `hydrodynamic_results.csv` contains one row for each interaction strength used in the numerical calculations.

The columns are:

- `gamma0` — dimensionless Lieb--Liniger coupling $\gamma_0=c/\rho_0$.
- `f_numerical` — dimensionless EFP rate function obtained from numerical minimization of the hydrodynamic action.
- `tau_c` — optimized critical imaginary time of the emptiness instanton.

These data are the numerical values used in the figures and comparisons reported in the paper.

## Running the numerical calculation

The coupling is selected by setting the parameter `c` near the beginning of the solver. Since the production calculations use \(\rho_0=1\), one has

$$
\gamma_0 = c.
$$

Running the script constructs the Lieb--Liniger equation of state and then minimizes the hydrodynamic action. The optimized EFP action and critical time can then be compared directly with the values in `hydrodynamic_results.csv`.
The production calculations use double precision and a $200\times200$ hydrodynamic grid. Further details of the parametrization, numerical domain, equation-of-state construction, and optimization procedure are given in the paper.

## Reproducing plots

The CSV file can be loaded directly with NumPy or pandas. For example,

```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("hydrodynamic_results.csv")

plt.loglog(data["gamma0"], data["f_numerical"], "o")
plt.xlabel(r"$\gamma_0$")
plt.ylabel(r"$f(\gamma_0)$")
plt.tight_layout()
plt.show()
```

The same data can be used to reproduce the numerical critical-time plot using the `tau_c` column.

## Citation

If you use these data or this code, please cite the associated paper.

## License

License TBD.
