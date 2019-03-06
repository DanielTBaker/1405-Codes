import numpy as np



## 1-D Wiener Filter
def WF_1D(S, Smat, N1d, bdpnts=np.array([])):
	"""
	1-D Wiener Filter

	Returns the 1-D Wiener Filtered Version of an array as well as the Noise Variance on each point

	Arguments
	----------
	S : np_array
		Array to be filtered. If S is 2D filter is applied along axis 1
	Smat : np_array
		Covariance Matrix of the signal
	N1d : np_array
		Noise Variance at each point
	bdpnts : np_array
		Indices of points with effectively infinite noise
	"""

	N = np.copy(N1d)
	if bdpnts.shape[0] > 0:
		if S.dtype == 'complex128' or S.dtype == 'complex64':
			N[bdpnts] = N.max() * 1e10
			Nmat = np.diag(N)
			W = Smat @ np.linalg.inv(Smat + Nmat)
			errsq = Smat - (W @ Smat).conj().T - W @ Smat + W @ (Smat + Nmat) @ W.conj().T
			WS = (W @ S.T).T
		else:
			N[bdpnts] = N.max() * 1e10
			Nmat = np.diag(N)
			W = Smat @ np.linalg.inv(Smat + Nmat)
			errsq = Smat - (W @ Smat).T - W @ Smat + W @ (Smat + Nmat) @ W.T
			WS = (W @ S.T).T
	else:
		if S.dtype == 'complex128' or S.dtype == 'complex64':
			Nmat = np.diag(N)
			W = Smat @ np.linalg.inv(Smat + Nmat)
			errsq = Smat - (
			W @ Smat).conj().T - W @ Smat + W @ (Smat + Nmat) @ W.conj().T
			WS = (W @ S.T).T
		else:
			Nmat = np.diag(N)
			W = Smat @ np.linalg.inv(Smat + Nmat)
			errsq = Smat - (W @ Smat).T - W @ Smat + W @ (Smat + Nmat) @ W.T
			WS = (W @ S.T).T
	return (WS, np.diag(errsq))


## 2-D Wiener Filter
def WF_2D(S,bf0=1,bf1=1):
	"""
	2-D Wiener Filter

	Arguments
	---------
	S : np_array
		Array to be filtered
	bf0 : int
		Rebinning factor in fourier space for axis 0, or number of chunks along 0 axis in real space
	bf1 : int
		Rebinning factor in fourier space for axis 1, or number of chunks along 1 axis in real space

	Note: Rebinning code is still being developed. It produces a better result within each chunk, but is disonctinuous at the boundries

	"""

	n1 = S.shape[1]
	n0 = S.shape[0]

	S2=S[:n0-np.mod(n0,bf0),:n1-np.mod(n1,bf1)]

	n1 = S2.shape[1]
	n0 = S2.shape[0]

	f0=np.fft.fftfreq(n0)
	f1=np.fft.fftfreq(n1)

	CS = np.abs(np.fft.fft2(S2)/np.sqrt(n1*n0))**2
	CS_rebin = np.fft.ifftshift(np.mean(np.reshape(np.fft.fftshift(CS), (n0//bf0, bf0, n1//bf1, bf1)), axis=(3, 1)))

	NSE = CS[np.abs(f0)>9*f0.max()/10,:][:,np.abs(f1)>9*f1.max()/10].mean()

	CS_rebin_sig = CS_rebin - NSE
	CS_rebin_sig[CS_rebin_sig < NSE] = 0

	S3=S2*0

	for i in range(bf0):
		for j in range(bf1):
			S3[(n0//bf0)*i:(n0//bf0)*(i+1),(n1//bf1)*j:(n1//bf1)*(j+1)] = np.fft.ifft2(np.fft.fft2(S2[(n0//bf0)*i:(n0//bf0)*(i+1),(n1//bf1)*j:(n1//bf1)*(j+1)])*(CS_rebin_sig/CS_rebin))
	return(S3)

	
	
