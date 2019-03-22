import numpy as np
from scipy.optimize import minimize

def chi_check(dt,z_f,z_var,pp_f,freqs,ngates):
    a = np.sum( (z_f*pp_f.conj()*np.exp(1j*2*np.pi*freqs*dt) + z_f.conj()*pp_f*np.exp(-1j*2*np.pi*freqs*dt))[1:] ) / np.sum( 2 * np.abs(pp_f[1:])**2 )
    num = (z_f - (a * pp_f * np.exp(-1j*2*np.pi*freqs*dt)))[1:] # consider k>0
    chisq = np.sum(np.abs(num)**2) / (z_var * ngates/2)
    return chisq

def template_match(template, z, off_gates = None, smart_guess = True, xguess=0., method='Nelder-Mead'):
    """
    Template matching code from Pulsar Timing and Relativistic Gravity, Taylor 1992, appendix A

    Returns delta_t, error in delta_t, scale, and baseline shift needed to match the profile with the provided template.

    Fang Xi Lin, Rob Main, 2019

    Arguments
    ----------
    template : np_array
        Template profile

    z :
        Profile to match the template to

    off_gates :
        indices of the off pulse in z. If not provided, assume gatess below the median are off pulse. Used to estimate errors.

    smart_guess :
        By default, we cross correlate template profile and profile to be matched, and use the highest bin as the initial guess for dt.

    xguess :
        Only used if smart_guess is set to False. Initial guess `dt` for the minimization.

    method :
        Method to use in scipy.optimize.minimize in the chisquare minimization. Nelder-Mead typically performs well, given a reasonable initial guess.
        However, it does tend to converge to local minima if the inital guess is far from the true values.
    """
    ngates = template.size

    assert z.size == ngates, "Template profile and profile to fit are not of the same size. Perhaps rebin one of them?"
    ## TODO: make it work despite the different lengths

    t = np.linspace(0,1,ngates,endpoint=False)
    t += (t[1]-t[0])/2
    ppdt = t[1]-t[0]

    freqs = np.fft.rfftfreq(ngates,ppdt)

    template_f = np.fft.rfft(template)
    z_f = np.fft.rfft(z)

    # if off_gates is provided, use that to compute variance, otherwise, use gates below the median.
    if off_gates:
        z_var = np.sum(np.var(z[off_gates])) # variance in the off pulse, for chisquare purposes
    else:
        z_var = np.sum(np.var(z[z<np.median(z)]))

    if smart_guess == True:
    # cross-correlate profiles, take peak bin to get starting guess
        profcorr = np.fft.irfft( np.fft.rfft(z) * np.fft.rfft(template).conj() )
        x = np.fft.fftfreq(len(template))
        xguess = x[np.argmax(profcorr)]

    minchisq = minimize(chi_check, x0=xguess, args=(z_f,z_var,template_f,freqs,ngates),method=method)
    if minchisq.success != True:
        print('Chi square minimization failed to converge. !!BEWARE!!')
        
    dt = minchisq.x
    
    # error term is eq. A10 from the paper, massaged a bit.
    dterr = np.sqrt( (z_var*ngates/2)/a / np.sum( ((2*np.pi*freqs)**2*(z_f*template_f.conj()*np.exp(1j*2*np.pi*freqs*dt)+z_f.conj()*template_f*np.exp(-1j*2*np.pi*freqs*dt)))[1:] ).real )
    
    a = np.sum( (z_f*template_f.conj()*np.exp(1j*2*np.pi*freqs*dt) + z_f.conj()*template_f*np.exp(-1j*2*np.pi*freqs*dt))[1:] ) / np.sum( 2 * np.abs(template_f[1:])**2 )
    
    b = (z_f[0] - a * template_f[0]).real/ngates
    
    return dt, dterr, a, b

def shift(z, dt):
    """
    Sub-bin shifting

    returns z shifted by dt in units of cycle (ie., dt=1 returns the same z).

    Arguments
    ---------
    z :
        Profile to shift

    dt :
        Phase to shift
    """

    ngates = z.size
    freqs = np.fft.rfftfreq(ngates,1./ngates)
    return np.fft.irfft(np.exp(-1j*2*np.pi*freqs*dt)*np.fft.rfft(z))
