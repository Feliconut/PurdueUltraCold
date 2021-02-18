from . import NPS

# %%
def from_od(OD_data, norm=False, imgSysData=None):
    """
    This is a combination of `ReadInImages` and `calcNPS__`. 
    Calculate the experimental imaging response function.

    ----------
    parameters

    same as `ReadInImages`

    -------
    returns

    M2k_Exp: numpy.ndarray, the calculated imaging response function 
        aalready subtracted by pure noise result).
    M2k_Exp_atom: numpy.ndarray, the imaging response function calculated 
        from images of atoms (suffered from shot noise).
    M2k_Exp_noise: numpy.ndarray, the imaging response function calculated 
        from images of pure noise (only shot noise affects)
    imgIndexMin, imgIndexMax, atomODs, noiseODs, atomODAvg, noiseODAvg: 
        same as `readInImages` 
    """
    ODs_atom, ODs_atom_avg, ODs_noise, ODs_noise_avg = OD_data

    # M2k_Exp is same as NPSs_avg.
    M2k_Exp_atom, _ = NPS.from_od(ODs_atom, ODs_atom_avg, norm, imgSysData)
    M2k_Exp_noise, _ = NPS.from_od(ODs_noise, ODs_noise_avg, norm, imgSysData)

    M2k_Exp_atom = M2k_Exp_atom / ODs_atom_avg.sum()
    M2k_Exp_noise = M2k_Exp_noise / ODs_atom_avg.sum()

    M2k_Exp_atom[M2k_Exp_atom.shape[0] // 2, M2k_Exp_atom.shape[1] // 2] = 0

    M2k_Exp = M2k_Exp_atom  #- M2k_Exp_noise

    return M2k_Exp, M2k_Exp_atom, M2k_Exp_noise
