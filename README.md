# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
## Date: 02-05-2026
## Name: Ramitha Chowdary S
## Reg no: 212224240130
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('/content/AirPassengers.csv')

# Declare required variables and set figure size
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]
X = data['#Passengers']

# Visualise the original data
plt.plot(X)
plt.title('Original Data')
plt.show()

# ACF and PACF of original data
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

# Fitting the ARMA(1,1) model and deriving parameters
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11   = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

# Fitting the ARMA(2,2) model and deriving parameters
arma22_model  = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22   = arma22_model.params['ar.L1']
phi2_arma22   = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```
### OUTPUT:
Original Data:
<img width="608" height="329" alt="image" src="https://github.com/user-attachments/assets/48ed6757-b376-4e7f-b974-e03702e9e53a" />

Original Data ACF & PACF:
<img width="738" height="365" alt="image" src="https://github.com/user-attachments/assets/b9832ea7-2316-4405-8de2-2d5f7cb8b146" />

SIMULATED ARMA(1,1) PROCESS:
<img width="615" height="326" alt="image" src="https://github.com/user-attachments/assets/72f563a3-5bf7-401a-a4f1-c8032c95056a" />

Autocorrelation — ARMA(1,1):
<img width="621" height="324" alt="image" src="https://github.com/user-attachments/assets/ae7f1695-6726-477d-b75f-e77de43faaa0" />

Partial Autocorrelation — ARMA(1,1):
<img width="620" height="327" alt="image" src="https://github.com/user-attachments/assets/86fc1b24-ab4f-42c0-991c-7fce8eb57c61" />

SIMULATED ARMA(2,2) PROCESS:
<img width="620" height="328" alt="image" src="https://github.com/user-attachments/assets/d81ac7e2-c609-46b6-99d3-2b9a8ab980b8" />

Autocorrelation — ARMA(2,2):
<img width="622" height="329" alt="image" src="https://github.com/user-attachments/assets/71b1d6b4-0744-4d76-a848-dde26d19ae5d" />

Partial Autocorrelation — ARMA(2,2):
<img width="624" height="329" alt="image" src="https://github.com/user-attachments/assets/c887916e-b245-4c1e-a637-6f97c2595a98" />

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
