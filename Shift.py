import numpy as np
from scipy.optimize import minimize

def chi_check(X,z_f,z_var,pp_f,freqs):
    dt, a = X
    num = (z_f - (a * pp_f * np.exp(-1j*2*np.pi*freqs*dt)))[1:] # consider k>0
    chisq = np.sum(np.abs(num)**2) / (z_var * 2048/2)
    return chisq

def template_match(template, z, offpulse_bins = None):
    """
    Template matching code from Pulsar Timing and Relativistic Gravity, Taylor 1992, appendix A

    Returns delta_t, error in delta_t, scale, and baseline shift needed to match the profile with the provided template.

    Arguments
    ----------
    template : np_array
        Template profile

    z :
        Profile to match the template to

    offpulse_bins :
        indices of the off pulse in z. If not provided, assume bins below the median are off pulse.
    """
    nbins = template.size

    t = np.linspace(0,1,nbins,endpoint=False) # 2048 is the number of bins per pulse
    t += (t[1]-t[0])/2
    ppdt = t[1]-t[0]

    freqs = np.fft.rfftfreq(nbins,ppdt)

    template_f = np.fft.rfft(template)
    z_f = np.fft.rfft(z)

    # if offpulse_bins is provided, use that to compute variance, otherwise, use bins below the median.
    if offpulse_bins:
        z_var = np.sum(np.var(z[offpulse_bins])) # variance in the off pulse, for chisquare purposes
    else:
        z_var = np.sum(np.var(z[z<np.median(z)]))
    xguess = [0.,1.]
    dt, a = minimize(chi_check, x0=xguess, args=(z_f,z_var,template_f,freqs)).x
    # error term is eq. A10 from the paper, massaged a bit.
    dterr = np.sqrt( (z_var*nbins/2)/a / np.sum( ((2*np.pi*freqs)**2*(z_f*template_f.conj()*np.exp(1j*2*np.pi*freqs*dt)+z_f.conj()*template_f*np.exp(-1j*2*np.pi*freqs*dt)))[1:] ).real )
    b = (z_f[0] - dt * template_f[0]).real/nbins
    return dt, dterr, a, b

def shift(z, dt):
    """
    Sub-bin alignment

    returns z shifted by dt in units of phase (ie., dt=1 returns the same z).

    Arguments
    ---------
    z :
        Profile to shift

    dt :
        Phase to shift
    """

    nbins = z.size
    freqs = np.fft.rfftfreq(nbins,ppdt)
    return np.fft.irfft(np.exp(-1j*2*np.pi*np.fft.rfftfreq(nbins,1./nbins)*dt)*np.fft.rfft(z))
