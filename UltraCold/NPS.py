'Calculation of the Noise Power Spectrum (NPS)'
from numpy import abs, pi
from numpy.fft import fftshift, fft2


def _sd_from_od(OD, imgSysData=None):
    """
    Calculate atom surface density (sd) from optical density (od).
    """

    if imgSysData["ODtoAtom"] == 'beer':
        px = imgSysData["CCDPixelSize"] / imgSysData["magnification"]
        AtomNum = OD * 2 * pi / (3 * imgSysData["wavelen"]**2) * px**2
    else:
        AtomNum = OD * imgSysData["ODtoAtom"]
    return AtomNum


def from_od(ODs, OD_avg, norm=False, imgSysData=None):
    """
    Calculate the noise power spectrum (NPS), from a set of images and their
    average. Note that `ODs` is considered as an ensemble, which means each
    image of `ODs` should be taken under identical conditions, and `ODAvg`
    is considered as the ensemble average.

    ODs: list, each element of which is a numpy.ndarray, the matrix of each 
        image.
    ODAvg: numpy.ndarray, the ensemble average of `ODs`.
    norm: bool. If true, use the atom number to normalize the noise power 
        spectrum. If false, use OD to calculate. In the latter case, the 
        absolute value of the noise power spectrum is meaningless.

    noisePowSpecs: list, each element of which is the noise power spectrums of
        an image in `ODs`. 
    NPS: numpy.ndarray, the average of noisePowSpecs, which is taken as 
        ensemble average
    """

    NPSs = []
    for OD in ODs:
        noise = OD - OD_avg
        if norm:
            noise = _sd_from_od(noise, imgSysData)
        noiseFFT = fftshift(fft2(fftshift(noise)))
        NPSs_avg = abs(noiseFFT)**2
        NPSs.append(NPSs_avg)

    NPSs_avg = sum(NPSs) / len(NPSs)
    return NPSs_avg, NPSs
