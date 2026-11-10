# Transfer Matrix summary

## Theory: normal acoustic propagation / reflection in layered stack

- Feiler S. Gold L. Hartmann S. & Giffin G. A. (2025). Modeling acoustic attenuation sound velocity and wave propagation in lithium-ion batteries via a transfer matrix. *Batteries & Supercaps 8*(3) e202400478.
- Byrnes S. J. (2016). Multilayer optical calculations. arXiv preprint arXiv:1603.02720.
- [To read: TMM for estimating dispersion](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1054&context=herrick)

Using modified notation from the Feiler paper for clarity.

An acoustic waveform $\psi_i$ enters a stack of materials $M_n$. Here layers 0 and 4 are coupled to the transducer and we assume there is no reflective interface.

$$
\psi_i \rightarrow M_0 \|\ M_1 \|\ M_2 \|\ M_3 \|\ M_4
$$

### Propagation matrix

$P_{M_n}$ is the propagation matrix through material $M_n$:

$$
P_{M_n} =
\begin{bmatrix}
e^{(ik_{M_n} - \alpha_{M_n}) d_{M_n}} & 0 \\
0 & e^{-(ik_{M_n} - \alpha_{M_n}) d_{M_n}}
\end{bmatrix}
$$

- $d_{M_n}$ is the layer thickness
- $\alpha_{M_n}$ is the attenuation coefficient of the medium
- $c_{M_n}$ is the speed of sound through the medium
- $k_{M_n}$ is the wavenumber defined as:

$$
k_{M_n} = \frac{2\pi}{\lambda} = \frac{2\pi f}{c_{M_n}}
$$

### Interface matrix

$D_{M_nM_m}$ is the interface matrix while travelling from $M_n$ to $M_m$:

$$
D_{M_nM_m} = \frac{1}{t_{M_nM_m}}
\begin{bmatrix}
1 & r_{M_nM_m} \\
r_{M_nM_m} & 1
\end{bmatrix}
$$

- $r_{M_nM_m}$ is the reflection coefficient travelling from $M_n$ to $M_m$:

$$
r_{M_nM_m} = \frac{Z_{M_m} - Z_{M_n}}{Z_{M_n} + Z_{M_m}}
$$

- $t_{M_nM_m}$ is the transmission coefficient travelling from $M_n$ to $M_m$:

$$
t_{M_nM_m} = \frac{2 Z_{M_m}}{Z_{M_n} + Z_{M_m}}
$$

- $Z_{M_n}$ is the acoustic impedance: $Z_{M_n} = \rho_{M_n} c_{M_n}$

### Full stack

The full stack is constructed as:

$$
\begin{bmatrix} \psi_{ir} \\ 
\psi_{il} \end{bmatrix}
= P_{M_0} D_{M_0M_1} P_{M_1} D_{M_1M_2} P_{M_2} D_{M_2M_3} P_{M_3} D_{M_3M_4} P_{M_4}
\begin{bmatrix} \psi_{fr} \\ 
\psi_{fl} \end{bmatrix}
$$

**Boundary conditions:**

- $\psi_{ir}$ is our input wave as a numpy array.
- $\psi_{il}$ is unknown. This represents our reflected wave.
- $\psi_{fr}$ is unknown. This represents our transmitted wave.
- $\psi_{fl}$ is 0. This is under the assumption that the last layer extends infinitely.
    - This might not be a good assumption. We may need the acoustic properties of the clamp or transducers to use as the last layer if they contribute to the waveform significantly.
    - Alternatively we can have a material at the end of the stack that guides the pulse away from the sample like an angled block.

The final waveform is related to the initial through the transfer matrix:

$$
\psi_i = T \psi_f
$$

$$
\begin{bmatrix} \psi_{ir} \\ 
\psi_{il} \end{bmatrix} =
\begin{bmatrix} T_{11} & T_{12} \\ 
T_{21} & T_{22} \end{bmatrix}
\begin{bmatrix} \psi_{fr} \\ 
\psi_{fl} \end{bmatrix}
$$

$(\psi_{fr} \psi_{fl})$ are the final right- and left-moving waveforms and $(\psi_{ir} \psi_{il})$ are the initial ones.

$T$ is the transfer matrix constructed by multiplying the propagation and interface matrices together. With $\psi_{fl}=0$ we can solve the system:

- Transmission where the transmission coefficient is $T(k) = 1/T_{11}$:

$$
\psi_{fr} = \frac{1}{T_{11}} \psi_{ir}
$$

- Reflection where the reflection coefficient is $R(k) = T_{21}/T_{11}$:

$$
\psi_{il} = \frac{T_{21}}{T_{11}} \psi_{ir}
$$

### Multi-frequency input

If your input signal contains multiple frequencies (such as our Morlet) you need to calculate the transfer matrix at every frequency since the behavior depends on the wavenumber $k = 2\pi f / c$:

1. Take the FFT of the Morlet input.
2. Calculate the transfer matrix $T$ at every frequency. If this is too expensive we can threshold which bins to use.
3. Calculate $`(\psi_{fr} \psi_{il})`$ in the frequency domain and apply the IFFT to get back to the time domain.

When using the FFT there will be signal at both $\pm f$. Then $`T(-f) = T(f)^*`$ because $`P_{M_n}(-f) = P_{M_n}(f)^*`$ and $`D_{M_nM_m}`$ is not frequency dependent.

## Discretizing the time signal: padding and phase shift to center the signal

There are complications when discretizing the signal. Consider that the time axis is partially negative so we need to apply a phase shift.

- Based on [this theory](https://brianmcfee.net/dstbook-site/content/ch06-dft-properties/Shifting.html) and [this post on Stack Exchange](https://app.notion.com/p/chang-lab/3dc8603a8ec680529706c5e31bc19ae3?v=3dc8603a8ec680db9714000c9264bde8&p=3dc8603a8ec680d79318d0fb7bd7e53a&pm=s).
- And this book: Mallat S. (1999). *A wavelet tour of signal processing*. Used to prove the FFT formula for the Morlet formula (2.18).

Translation in the time domain is a phase shift in the Fourier domain (2.18):

$$
f(t - t_0) \rightarrow e^{-i t_0 \omega} \hat{f}(\omega)
$$

The discrete Fourier transform is:

$$
X[k] = \sum_{n=0}^{N-1} x[n] \ e^{-\frac{j 2\pi k n}{N}}
$$

- $X[k]$ is the DFT
- $k$ is the frequency bin index (not the physical wavenumber)
- $N$ is the number of samples (number of timesteps)

Separate the magnitude $A_k$ and angle $\phi_k$ for each bin $k$:

$$
X[k] = A_k \ e^{j \phi_k}
$$

Now suppose there is a time delay of $d$ points i.e. time is 0 at position $d$ in the time axis. Define a new function for the shifted waveform and write its DFT in terms of the old one.

- Assume the shift is circular so $y[n] = x[(n-d) \bmod N]$ shortened to $y[n] = x[n-d]$.
- $Y[k] = \mathrm{FFT}(y[n])$.
- Let $m = n - d$.

$$
Y[k] = \sum_{n=0}^{N-1} y[n] \ e^{-\frac{j 2\pi k n}{N}}
= \sum_{n=0}^{N-1} x[n-d] \ e^{-\frac{j 2\pi k n}{N}}
$$

$$
Y[k] = \sum_{m=-d}^{N-1-d} x[m] \ e^{-\frac{j 2\pi k (m+d)}{N}}
= \sum_{m=-d}^{N-1-d} x[m] \ e^{-\frac{j 2\pi k m}{N}} \ e^{-\frac{j 2\pi k d}{N}}
$$

$$
Y[k] = e^{-\frac{j 2\pi k d}{N}} \sum_{m=-d}^{N-1-d} x[m] \ e^{-\frac{j 2\pi k m}{N}}
$$

Finally for a circular shift:

$$
Y[k] = e^{-\frac{j 2\pi k d}{N}} X[k]
$$

However if the shift is not circular we need to pad zeros at the beginning or end perform the TMM calculations then crop off the "wrapped-around" portion of the signal. We will need to test several pad lengths to see how much is enough to prevent interference with the TMM signal.

## Comparing FFT amplitudes and understanding power scaling

- Look at Feiler SI A.6 for reconstruction of Morlets.
- Mallat S. (1999). *A wavelet tour of signal processing*. Formulas (2.17) (2.18).
- Yi H. & Shu H. (2012). The improvement of the Morlet wavelet for multi-period analysis of climate data. *Comptes Rendus Geoscience 344*(10) 483-497. https://doi.org/10.1016/j.crte.2012.09.007

Fourier transform:

$$
\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t) \ e^{-i\omega t} \ dt
$$

Real Morlet:

$$
f(t) = e^{-\frac{t^2}{2\sigma^2}} \cos(\omega_0 t)
$$

Multiplication in the time domain is convolution ($\ast$) in the Fourier domain (2.17):

$$
f_1(t) f_2(t) \rightarrow \frac{1}{2\pi} \left( \hat{f_1} \ast \hat{f_2} \right)(\omega)
$$

With Morlets $f_1(t) = e^{-\frac{t^2}{2\sigma^2}}$ and $f_2(t) = \cos(\omega_0 t)$.

### Gaussian

$$
\hat{f_1}(\omega)
= \int_{-\infty}^{\infty} e^{-\frac{t^2}{2\sigma^2}} e^{-i\omega t} \ dt
= \int_{-\infty}^{\infty} e^{-\frac{t^2}{2\sigma^2} - i\omega t} \ dt
$$

Expand and complete the square:

$$
a^2 + 2ab + b^2 = (a+b)^2 \;\rightarrow\; a^2 + 2ab = (a+b)^2 - b^2
$$

$$
a^2 = \left(\frac{t}{\sqrt{2}\sigma}\right)^2 \;\rightarrow\; a = \frac{t}{\sqrt{2}\sigma}
$$

$$
2ab = 2\\frac{t}{\sqrt{2}\sigma}\ b = i\omega t \;\rightarrow\; b = \frac{i\omega\sigma}{\sqrt{2}}
$$

$$
b^2 = -\frac{\omega^2\sigma^2}{2}
$$

$$
\hat{f_1}(\omega)
= \int_{-\infty}^{\infty} e^{-\left( \frac{t^2}{2\sigma^2} + i\omega t \right)} dt
= \int_{-\infty}^{\infty} e^{-\left( \left( \frac{t}{\sqrt{2}\sigma} + \frac{i\omega\sigma}{\sqrt{2}} \right)^2 + \frac{\omega^2\sigma^2}{2} \right)} dt
$$

Using the Gaussian integral $\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$ with

$$
x = \frac{t}{\sqrt{2}\sigma} + \frac{i\omega\sigma}{\sqrt{2}} \qquad dx = \frac{dt}{\sqrt{2}\sigma}
$$

$$
\hat{f_1}(\omega)
= e^{-\frac{\omega^2\sigma^2}{2}} \sqrt{2}\sigma \int_{-\infty}^{\infty} e^{-x^2} \ dx
= \sqrt{2\pi}\\sigma\ e^{-\frac{\omega^2\sigma^2}{2}}
$$

### Cosine

Using Euler's formula:

$$
f_2(t) = \cos(\omega_0 t) = \frac{e^{i\omega_0 t} + e^{-i\omega_0 t}}{2}
$$

$$
\hat{f_2}(\omega) = \int_{-\infty}^{\infty} \frac{e^{i\omega_0 t} + e^{-i\omega_0 t}}{2} \ e^{-i\omega t} \ dt
$$

$$
\hat{f_2}(\omega) = \int_{-\infty}^{\infty} \frac{e^{-i(\omega - \omega_0)t} + e^{-i(\omega + \omega_0)t}}{2} \ dt
$$

Using the definition of the Dirac delta $\int_{-\infty}^{\infty} e^{-ikt} \ dt = 2\pi\delta(k)$:

$$
\hat{f_2}(\omega) = \frac{2\pi\delta(\omega - \omega_0) + 2\pi\delta(\omega + \omega_0)}{2}
= \pi \left\[ \delta(\omega - \omega_0) + \delta(\omega + \omega_0) \right\]
$$

### Convolution

$$
\hat{f}(\omega)
= \frac{1}{2\pi} \left( \hat{f_1} \ast \hat{f_2} \right)(\omega)
= \frac{1}{2\pi} \left( \sqrt{2\pi}\\sigma\ e^{-\frac{\omega^2\sigma^2}{2}} \ast \pi \left\[ \delta(\omega - \omega_0) + \delta(\omega + \omega_0) \right\] \right)
$$

$$
\hat{f}(\omega) = \sqrt{\frac{\pi}{2}}\\sigma \left( e^{-\frac{(\omega - \omega_0)^2\sigma^2}{2}} + e^{-\frac{(\omega + \omega_0)^2\sigma^2}{2}} \right)
$$

- The FT of the real Morlet is two Gaussians located at $\pm\omega_0$ the transducer frequency.
- The peak height scales the original amplitude by $\sqrt{\pi/2}\\sigma$.
- The new standard deviation is $1/\sigma$.

## MISC: revisit the proofs

Attempt at doing the convolution directly (needs checking see notes):

$$
\hat{f}(\omega) = \frac{1}{2\pi} \left( \hat{f_1} \ast \hat{f_2} \right)(\omega)
= \frac{1}{2\pi} \int_{-\infty}^{\infty} f_1(\tau) f_2(t - \tau) \ d\tau
$$

$$
\hat{f}(\omega) = \frac{1}{2\pi} \int_{-\infty}^{\infty} e^{-\frac{\tau^2}{2\sigma^2}} \cos(\omega_0 (t - \tau)) \ d\tau
$$

Euler's formula:

$$
\cos(\omega_0 (t - \tau)) = \frac{e^{i\omega_0 (t - \tau)} + e^{-i\omega_0 (t - \tau)}}{2}
$$

Expand and factor constants:

$$
\hat{f}(\omega) = 
\frac{1}{4\pi} \int_{-\infty}^{\infty} e^{-\frac{\tau^2}{2\sigma^2}} e^{i\omega_0 (t - \tau)} \ d\tau + 
\frac{1}{4\pi} \int_{-\infty}^{\infty} e^{-\frac{\tau^2}{2\sigma^2}} e^{-i\omega_0 (t - \tau)} \ d\tau
$$

$$
\hat{f}(\omega) = 
\frac{e^{i\omega_0 t}}{4\pi} \int_{-\infty}^{\infty} e^{-\left( \frac{\tau^2}{2\sigma^2} + i\omega_0 \tau \right)} d\tau + 
\frac{e^{-i\omega_0 t}}{4\pi} \int_{-\infty}^{\infty} e^{-\left( \frac{\tau^2}{2\sigma^2} - i\omega_0 \tau \right)} d\tau
$$

Complete the square:

$$
a = \frac{\tau}{\sqrt{2}\sigma} \qquad 2ab = \pm i\omega_0 \tau \;\rightarrow\; b = \pm i\omega_0 \sigma / \sqrt{2} \qquad b^2 = -\frac{\omega_0^2\sigma^2}{2}
$$

(The original draft had $b = \pm i\omega_0\sqrt{2}\sigma$ and $b^2 = \mp 2\omega_0^2\sigma^2$ which does not follow from $2ab = \pm i\omega_0\tau$.)

Original draft continuation kept for reference:

$$
\hat{f}(\omega)
= \frac{e^{i\omega_0 t} e^{2\omega_0^2\sigma^2}}{4\pi} \int_{-\infty}^{\infty} e^{-\left( \frac{\tau}{\sqrt{2}\sigma} + i\omega_0\sqrt{2}\sigma \right)^2} d\tau + 
\frac{e^{-i\omega_0 t} e^{-2\omega_0^2\sigma^2}}{4\pi} \int_{-\infty}^{\infty} e^{-\left( \frac{\tau}{\sqrt{2}\sigma} - i\omega_0\sqrt{2}\sigma \right)^2} d\tau
$$

$$
\hat{f}(\omega)
= \frac{\sigma}{2\sqrt{2\pi}} \left( e^{i\omega_0 t + 2\omega_0^2\sigma^2} + e^{-\left( i\omega_0 t + 2\omega_0^2\sigma^2 \right)} \right)
= \frac{\sigma}{\sqrt{2\pi}} \cosh\!\left( i\omega_0 t + 2\omega_0^2\sigma^2 \right)
$$

## Misc

**Note: this is wrong since the reflected waves are already interacting in the first part. Needs testing.**

TODO (not in the paper verify): reflection off the back wall (between materials $M_2$ and $M_3$) can be written as:

$$
\begin{bmatrix} \psi_{ir} \\ 
\psi_{il} \end{bmatrix}
= P_{M_0} D_{M_0M_1} P_{M_1} D_{M_1M_2} P_{M_2} D_{M_2M_3} P_{M_3} D_{M_3M_4}
P_{M_3} D_{M_3M_2} P_{M_2} D_{M_2M_1} P_{M_1} D_{M_1M_0} P_{M_0}
\begin{bmatrix} \psi_{fr} \\ 
\psi_{fl} \end{bmatrix}
$$
