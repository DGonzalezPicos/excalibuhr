import os
import sys
import copy
os.environ["OMP_NUM_THREADS"] = "1"
import json
import warnings
import numpy as np
import pickle
import urllib.request
from scipy.interpolate import interp1d, splrep, splev, RegularGridInterpolator#, CubicSpline
from scipy import signal
from scipy.ndimage import gaussian_filter
# from numpy.polynomial import polynomial as Poly
from numpy.polynomial import chebyshev as Chev
from astropy.io import fits
from PyAstronomy import pyasl
import pymultinest
import excalibuhr.utils as su 
from excalibuhr.data import SPEC2D
from excalibuhr.telluric import TelluricGrid
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from petitRADTRANS.retrieval import rebin_give_width as rgw
import petitRADTRANS.poor_mans_nonequ_chem as pm
# from petitRADTRANS.retrieval import cloud_cond as fc
# from typeguard import typechecked
from sksparse.cholmod import cholesky
# import matplotlib.pyplot as plt 
# signal.savgol_filter
import pylab as plt
cmap = plt.get_cmap("tab10")

def getMM(species):
    """
    Get the molecular mass of a given species.

    This function uses the molmass package to
    calculate the mass number for the standard
    isotope of an input species. If all_iso
    is part of the input, it will return the
    mean molar mass.

    Args:
        species : string
            The chemical formula of the compound. ie C2H2 or H2O
    Returns:
        The molar mass of the compound in atomic mass units.
    """
    from molmass import Formula

    e_molar_mass = 5.4857990888e-4  # (g.mol-1) e- molar mass (source: NIST CODATA)

    if species == 'e-':
        return e_molar_mass
    elif species == 'H-':
        return Formula('H').mass + e_molar_mass

    name = species.split("_")[0]
    name = name.split(',')[0]
    f = Formula(name)

    if "all_iso" in species:
        return f.mass
    
    return f.isotope.massnumber

def calc_elemental_ratio(a, b, abundances):
    tol = 1e-20
    from molmass import Formula
    abund_a, abund_b = [], []
    for key in abundances:
        species = key.split("_")[0]
        f = Formula(species)
        df = f.composition().dataframe()
        if a in df.index:
            abund_a.append(df.loc[a, 'Count']*abundances[key])
        if b in df.index:
            abund_b.append(df.loc[b, 'Count']*abundances[key])
    abund_a = np.sum(abund_a, axis=0)
    abund_b = np.sum(abund_b, axis=0)
    return abund_a/(abund_b+tol)


def calc_mass_to_mol_frac(abundances, MMW):
    mol_fraction = {}
    for key in abundances:
        if "36" in key.split("_"):
            mass = getMM('[13C]O')
        else:
            mass = getMM(key)
        mol_fraction[key] = abundances[key] / mass * MMW
    return mol_fraction

def calc_MMW(abundances):
    """
    calc_MMW
    Calculate the mean molecular weight in each layer.

    Args:
        abundances : dict
            dictionary of abundance arrays, each array must have the shape of the pressure array used in pRT,
            and contain the abundance at each layer in the atmosphere.
    """
    mmw = sys.float_info.min  # prevent division by 0

    for key in abundances.keys():
        # exo_k resolution
        spec = key.split("_")[0]
        mmw += abundances[key] / getMM(spec)

    return 1.0 / mmw

class Parameter:

    def __init__(self, name, value=None, prior=(0,1), is_free=True):
        self.name = name
        self.is_free = is_free
        self.value = value
        self.prior = prior

    def set_value(self, value):
        self.value = value

    def set_prior(self, prior):
        self.prior = prior


class Photometry:

    def __init__(self, filter_name, f_lambda, f_err=None):

        self.name = 'photometry'
        self.mode = 'c-k'
        self.filter_name = filter_name
        self.f_lambda = f_lambda
        self.f_err = f_err
        self.get_filter_transm()

    def get_filter_transm(self):
        
        url = "http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id="+ self.filter_name
        urllib.request.urlretrieve(url, "filter.dat")
        transmission = np.genfromtxt('filter.dat')
        transmission[:,0] *= 1e-1 # wavelength in nm
        if transmission.size == 0:
            raise ValueError("The filter data of {} could not be downloaded".format(self.filter_name))
        os.remove('filter.dat')

        self.wlen = transmission[:,0]
        self.flux = transmission[:,1]

    def make_wlen_bins(self):
        data_wlen_bins = np.zeros_like(self.wlen)
        data_wlen_bins[:-1] = np.diff(self.wlen)
        data_wlen_bins[-1] = data_wlen_bins[-2]
        self.wlen_bins = data_wlen_bins


class ObsData(SPEC2D):

    def __init__(self, wlen=None, flux=None, err=None, filename=None, 
                 name=None, R=None):
        
        super().__init__(wlen=wlen, flux=flux, err=err, filename=filename)
        
        self.name = name
        self.R = R

        if self.R > 1000.:
            self.mode = 'lbl'
        else:
            self.mode = 'c-k'

        if self.name.lower() == 'crires':
            self.detector_bin = 3
        else:
            self.detector_bin = 1
        



class Retrieval:

    def __init__(self, retrieval_name, out_dir):

        self.retrieval_name = retrieval_name
        self.out_dir = out_dir
        self.prefix = self.out_dir+'/'+self.retrieval_name+'_'
        if not os.path.exists(self.out_dir):
            os.path.mkdir(self.out_dir)
        self.obs = {} 
        self.params = {}
        self.model_tellu = lambda x: np.ones_like(x)

        attr_keys = {
                'key_radius': 'radius',
                'key_rv': 'vsys',
                'key_spin': 'vsini',
                'key_limb_dark_u': 'limb',
                'key_distance': 'distance',
                'key_airmass': 'airmass',
                'key_carbon_to_oxygen': 'C/O',
                'key_carbon_iso': '13/12C',
                'key_metalicity': '[Fe/H]',
                'key_gravity': 'logg',
                'key_tellu_temp': 'tellu_temp',
                'key_quench': 'log_P_quench',
                'key_molecule_tolerance': 'beta',
                'key_teff_tolerance': 'gamma',
                'key_t_bottom': "t_00",
            }
        for par in attr_keys.keys():
            setattr(self, par, attr_keys[par])

    
    def add_observation(self, obs_list):
        for obs in obs_list:
            obs.make_wlen_bins()
            if obs.name == 'photometry':
                if obs.name not in self.obs:
                    self.obs[obs.name] = [obs]
                else:
                    self.obs[obs.name].append(obs)
            else:
                self.obs[obs.name] = obs



    def add_parameter(self, name, value=None, prior=(0,1), is_free=True):
        self.params[name] = Parameter(name, value, prior, is_free)
        print(f"Add parameter - {name}, value: {value}, prior: {prior}, is_free: {is_free}")
        


    def add_pRT_objects(self,
                       rayleigh_species=['H2', 'He'], 
                       continuum_opacities=['H2-H2', 'H2-He'],
                       cloud_species=None,
                       lbl_opacity_sampling=5,
                       ):
        if cloud_species is not None:
            do_scat_emis = True
        else:
            do_scat_emis = False

        self.pRT_object = {}
        for instrument in self.obs.keys():
            self.pRT_object[instrument] = []
            if instrument == 'photometry':
                for data_object in self.obs[instrument]:
                    dw = (data_object.wlen[1]-data_object.wlen[0])*5.
                    wlen_cut = [(data_object.wlen[0]-dw)*1e-3, (data_object.wlen[-1]+dw)*1e-3]
                    # print(wlen_cut)
                    rt_object = Radtrans(
                            line_species=self.line_species_ck,
                            rayleigh_species=rayleigh_species,
                            continuum_opacities=continuum_opacities,
                            cloud_species=cloud_species,
                            mode=data_object.mode,
                            wlen_bords_micron=wlen_cut,
                            lbl_opacity_sampling=lbl_opacity_sampling,
                            do_scat_emis=do_scat_emis,
                        )
                    rt_object.setup_opa_structure(self.press)
                    self.pRT_object[instrument].append(rt_object)
            else:
                data_object = self.obs[instrument]
                for i in range(0, data_object.Nchip, data_object.detector_bin):
                    wave_tmp = data_object.wlen[i:i+data_object.detector_bin].flatten()
                    # set pRT wavelength range sparing 200 pixels 
                    # beyond the data wavelengths for each order
                    dw = (wave_tmp[1]-wave_tmp[0])*200.
                    wlen_cut = [(wave_tmp[0]-dw)*1e-3, (wave_tmp[-1]+dw)*1e-3]
                    rt_object = Radtrans(
                            line_species=self.line_species,
                            rayleigh_species=rayleigh_species,
                            continuum_opacities=continuum_opacities,
                            cloud_species=cloud_species,
                            mode=data_object.mode,
                            wlen_bords_micron=wlen_cut,
                            lbl_opacity_sampling=lbl_opacity_sampling,
                            do_scat_emis=do_scat_emis,
                        )
                    rt_object.setup_opa_structure(self.press)
                    self.pRT_object[instrument].append(rt_object)
        
    

    def add_free_PT_model(self):

        for i in range(self.N_t_knots):
            self.add_parameter(f't_{i:02}')

        self.PT_model = self.free_PT_model


    def free_PT_model(self):

        p_ret = np.copy(self.press)
        t_names = [x for x in self.params if x.split('_')[0]=='t']
        t_names.sort(reverse=True)
        knots_t = [self.params[x].value for x in t_names]
        knots_p = np.logspace(np.log10(self.press[0]),np.log10(self.press[-1]), len(knots_t))
        t_spline = splrep(np.log10(knots_p), knots_t, k=1)
        tret = splev(np.log10(p_ret), t_spline, der=0)
        t_smooth = gaussian_filter(tret, 1.5)
        
        return t_smooth
    

    def add_free_chem_model(self):

        for species in self.line_species:
            if '36' in species.split('_'):
                param_name = "logX_" + species.split('_')[0] + "_36"
            else:
                param_name = "logX_" + species.split('_')[0]
            self.add_parameter(param_name, value=-20, prior=(-12, -2))
        self.chem_model = self.free_chem_model
    
    def free_chem_model(self):

        abundances = {}
        for species in self.line_species:
            if '36' in species.split('_'):
                param_name = "logX_" + species.split('_')[0] + "_36"
            else:
                param_name = "logX_" + species.split('_')[0]
            abundances[species] = np.ones_like(self.press) * 1e1 ** self.params[param_name].value

        sum_masses = 0
        for species in abundances.keys():
            sum_masses += abundances[species][0]
        massH2He = 1. - sum_masses
        abundances['H2'] = 2.*0.84/(4*0.16+2*0.84) * massH2He * np.ones_like(self.press)
        abundances['He'] = 4.*0.16/(4*0.16+2*0.84) * massH2He * np.ones_like(self.press)

        MMW = calc_MMW(abundances)*np.ones_like(self.press)

        # set abundances for corr-k species
        for species in self.line_species_ck:
            if '36' in species.split('_'):
                param_name = "logX_" + species.split('_')[0] + "_36"
            else:
                param_name = "logX_" + species.split('_')[0]
            abundances[species] = np.ones_like(self.press) * 1e1 ** self.params[param_name].value

        if self.debug:
            c2o = calc_elemental_ratio('C', 'O', 
                        calc_mass_to_mol_frac(abundances, MMW))
            co_iso_name = []
            for key in abundances:
                if key.split('_')[0] == 'CO':
                    co_iso_name.append(key)
            co_iso_name.sort()
            c_iso = abundances[co_iso_name[1]]/abundances[co_iso_name[0]]
            print("C/O: ", np.mean(c2o))
            print("12CO/13CO: ", np.mean(c_iso))

        return abundances, MMW


    def add_equ_chem_model(self):

        self.add_parameter(self.key_carbon_iso, prior=(-12, 0))
        self.add_parameter(self.key_carbon_to_oxygen, prior=(0.1, 1.5))
        self.add_parameter(self.key_metalicity, prior=(-1.5, 1.5))
        self.add_parameter(self.key_quench, value=-10, 
                           prior=(np.log10(self.press[0]), np.log10(self.press[-1])))
        
        self.chem_model = self.equ_chem_model

    def equ_chem_model(self):

        COs = self.params[self.key_carbon_to_oxygen].value * np.ones_like(self.press)
        FeHs = self.params[self.key_metalicity].value * np.ones_like(self.press)
        abund_interp = pm.interpol_abundances(
                    COs, FeHs, 
                    self.temp, self.press,
                    Pquench_carbon = 1e1**self.params[self.key_quench].value,
                    )
        MMW = abund_interp['MMW']

        abundances = {}
        abundances['H2'] = abund_interp['H2']
        abundances['He'] = abund_interp['He']
        for species in self.line_species + self.line_species_ck:
            if '36' in species.split('_'):
                abundances[species] = abund_interp[species.split('_')[0]] \
                                       * 1e1**self.params[self.key_carbon_iso].value
            else:
                abundances[species] = abund_interp[species.split('_')[0]]

        return abundances, MMW
        
    
    def add_cloud_model(self):
        pass
    
        
    
    def forward_model_pRT(self, leave_out=None):

        # get temperarure profile
        self.temp = self.PT_model()

        # get chemical abundances and mean molecular weight
        self.abundances, self.MMW = self.chem_model()

        # leave out one species
        if leave_out is not None:
            for key in self.abundances:
                if leave_out in key:
                    self.abundances[key] = 1e-30 * np.ones_like(self.press)

        self.model_native = {}
        for instrument in self.obs.keys():
            model = []
            for rt_object in self.pRT_object[instrument]:
                rt_object.calc_flux(self.temp,
                        self.abundances,
                        1e1**self.params[self.key_gravity].value,
                        self.MMW,
                        # contribution=contribution,
                        )
                # convert flux f_nu to f_lambda in unit of W/cm**2/um
                f_lambda = rt_object.flux*rt_object.freq**2./nc.c * 1e-7
                wlen_nm = nc.c/rt_object.freq/1e-7
                if self.key_radius in self.params:
                    f_lambda *= (self.params[self.key_radius].value * nc.r_jup \
                                / self.params[self.key_distance].value / nc.pc)**2
                model.append([wlen_nm, f_lambda])
            self.model_native[instrument] = model

        # if self.debug:
        #     plt.plot(self.temp, self.press)
        #     plt.ylim([self.press[-1],self.press[0]])
        #     plt.yscale('log')
        #     plt.xlabel('T (K)')
        #     plt.ylabel('Pressure (bar)')
        #     plt.show()
        #     plt.clf()

    def add_telluric_model(self, 
                           tellu_species=['H2O', 'CH4', 'CO2'],
                           tellu_grid_path=None,
                           ):
        self.tellu_species = tellu_species
        tellu_grid_path = self.out_dir
        tellu_grid = TelluricGrid(tellu_grid_path,
                    #  wave_range=,
                     free_species=self.tellu_species)
        self.tellu_grid = tellu_grid.load_grid()
        self.humidity_range = tellu_grid.humidity_range
        self.ppmv_range = tellu_grid.ppmv_range
        self.temp_range = tellu_grid.temp_range
        self.fixed_species = [s for s in tellu_grid.all_species if s not in tellu_grid.free_species]
        
        self.add_parameter(self.key_airmass, prior=(1., 3.))
        self.add_parameter(self.key_tellu_temp, prior=(self.temp_range[0], self.temp_range[-1]))
        
        for species in tellu_species:
            param_name = "tellu_" + species.split('_')[0]
            if species == 'H2O':
                self.add_parameter(param_name, prior=(self.humidity_range[0], self.humidity_range[-1]))
            else:
                self.add_parameter(param_name, prior=(self.ppmv_range[0], self.ppmv_range[-1]))


    def forward_model_telluric(self):

        y = np.ones_like(self.tellu_grid['WAVE'])
        for species in self.tellu_species:
            param_name = "tellu_" + species.split('_')[0]
            if species == 'H2O':
                rel_range = self.humidity_range
            else:
                rel_range = self.ppmv_range
            y *= RegularGridInterpolator((self.temp_range, rel_range), 
                        self.tellu_grid[species], bounds_error=False, fill_value=None)(
                        [self.params[self.key_tellu_temp].value, self.params[param_name].value])[0]
            y[y<0.] = 0.
        for species in self.fixed_species:
            y *= self.tellu_grid[species]
        tellu_native = y**(self.params[self.key_airmass].value)
        self.model_tellu = interp1d(self.w_tellu_native, tellu_native, bounds_error=False, fill_value='extrapolate')
        
        # if self.debug:
        #     plt.plot(self.w_tellu_native, tellu_native)
        #     plt.show()

    def plot_model_debug(self, model):
        for instrument in self.obs.keys():
            for dt in model[instrument]:
                wave_tmp, flux_tmp = dt[0], dt[1]
                plt.plot(wave_tmp, flux_tmp, 'k')
        plt.show()


    def plot_rebin_model_debug(self, model):
        fig, axes = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (3,1)})
        
        for instrument in self.obs.keys():
            for i, flux_tmp in enumerate(model[instrument]):
                if instrument == 'photometry':
                    wave_tmp = self.obs[instrument][i].wlen.mean()
                    flux_obs = self.obs[instrument][i].f_lambda
                    err_obs = self.obs[instrument][i].f_err
                    # print(flux_obs, flux_tmp)
                else:
                    wave_tmp = self.obs[instrument].wlen[i]
                    flux_obs = self.obs[instrument].flux[i]
                    err_obs = self.obs[instrument].err[i]
                    axes[0].errorbar(wave_tmp, flux_obs, err_obs, color='r', alpha=0.8)
                    axes[0].plot(wave_tmp, flux_tmp, color='k', alpha=0.8, zorder=10)
                    axes[1].plot(wave_tmp, (flux_obs-flux_tmp), color='k', alpha=0.8, zorder=10)
        # axes[0].set_ylim(0.1, 1.2)
        # axes[1].set_ylim(-5,5)
        axes[0].set_ylabel(r'Flux')
        axes[1].set_ylabel(r'Residual')
        axes[1].set_xlabel('Wavelength (nm)')
        plt.show()
        plt.close(fig)
        # return fig, axes


    

    def apply_rot_broaden_rv_shift(self):
        self.model_spin = {}
        for instrument in self.obs.keys():
            model = self.model_native[instrument]
            model_tmp = []
            for dt in model:
                wave_tmp, flux_tmp = dt[0], dt[1]
                wave_shift = wave_tmp * (1. + self.params["vsys"].value*1e5 / nc.c) 
                wlen_up = np.linspace(wave_tmp[0], wave_tmp[-1], len(wave_tmp)*20)

                flux_take = interp1d(wave_shift, flux_tmp, bounds_error=False, fill_value='extrapolate')(wlen_up)
                flux_spin = pyasl.fastRotBroad(wlen_up, flux_take, self.params[self.key_limb_dark_u].value, self.params[self.key_spin].value)
                model_tmp.append([wlen_up, flux_spin])
            self.model_spin[instrument] = model_tmp
        
        

    def add_instrument_kernel(self, Lorentzian_kernel=False):
        if self.fit_instrument_kernel:
            for instrument in self.obs.keys():
                self.add_parameter(f'{instrument}_G', prior=(0.3e5, 2e5))
                if Lorentzian_kernel:
                    self.add_parameter(f'{instrument}_L', prior=(0.1, 3))


    def apply_instrument_broaden(self):
        self.model_convolved = {}
        for instrument in self.obs.keys():
            if instrument == 'photometry':
                # skip broadening for photometry objects 
                self.model_convolved[instrument] = self.model_spin[instrument]
            else:
                if f'{instrument}_G' in self.params:
                    inst_G = self.params[f'{instrument}_G'].value
                else:
                    inst_G = self.obs[instrument].R
                if f'{instrument}_L' in self.params:
                    inst_L = self.params[f'{instrument}_L'].value
                else:
                    inst_L = 0
                model_target = self.model_spin[instrument]
                model_tmp = []
                for dt in model_target:
                    wave_tmp, flux_tmp = dt[0], dt[1]
                    flux_full = flux_tmp * self.model_tellu(wave_tmp)
                    flux_conv = su.SpecConvolve_GL(wave_tmp, flux_full, inst_G, inst_L)
                    model_tmp.append([wave_tmp, flux_conv])
                self.model_convolved[instrument] = model_tmp


    def apply_rebin_to_obs_wlen(self):
        self.model_rebin = {}
        for instrument in self.obs.keys():
            model_tmp = []
            if instrument == 'photometry':
                for model_target, obs_target in zip(self.model_convolved[instrument], self.obs[instrument]):
                    flux_rebin = rgw.rebin_give_width(
                                    model_target[0], 
                                    model_target[1],
                                    obs_target.wlen, 
                                    obs_target.wlen_bins,
                                    )
                    integrand1 = obs_target.wlen*obs_target.flux*flux_rebin
                    integrand2 = obs_target.wlen*obs_target.flux
                    integral1 = np.trapz(integrand1, obs_target.wlen)
                    integral2 = np.trapz(integrand2, obs_target.wlen)
                    model_tmp.append(integral1/integral2)
                    
            else:
                model_target = self.model_convolved[instrument]
                obs_target = self.obs[instrument]
                for i in range(obs_target.wlen.shape[0]):
                    flux_rebin = rgw.rebin_give_width(
                            model_target[i//obs_target.detector_bin][0], 
                            model_target[i//obs_target.detector_bin][1],
                            obs_target.wlen[i], 
                            obs_target.wlen_bins[i],
                            )
                    model_tmp.append(flux_rebin)
            self.model_rebin[instrument] = np.array(model_tmp)

        # if self.debug:
            # self.plot_model_debug(self.model_native)
            # self.plot_model_debug(self.model_spin)
            # self.plot_model_debug(self.model_convolved)
            # self.plot_rebin_model_debug(self.model_rebin)

    def add_poly_model(self): #or spline?
        if self.fit_poly:
            for instrument in self.obs.keys():
                if instrument != 'photometry':
                    for i in range(self.obs[instrument].Nchip):
                        for o in range(1, self.fit_poly+1):
                            self.add_parameter(f'poly_{instrument}_{o}_{i:02}', prior=(-5e-2/o, 5e-2/o))
        

    def apply_poly_continuum(self):
        # self.model_cont = {}
        for instrument in self.obs.keys():
            if instrument != 'photometry':
                model_target = self.model_rebin[instrument]
                obs_target = self.obs[instrument]
                model_tmp = []
                for i, y_model in enumerate(model_target):
                    x = obs_target.wlen[i]
                    # y = obs_target.flux[i]

                    if self.fit_poly:
                        # correct for the slope or higher order poly of the continuum
                        poly = [1.] 
                        for o in range(1, self.fit_poly+1):
                            poly.append(self.params[f'poly_{instrument}_{o}_{i:02}'].value)
                        y_poly = Chev.chebval((x - np.mean(x))/(np.mean(x)-x[0]), poly)
                        y_model *= y_poly
                        # plt.plot(x, y_poly)

                    model_tmp.append(y_model)
                # plt.show()
                self.model_rebin[instrument] = np.array(model_tmp)

        # if self.debug:
        #     self.plot_rebin_model_debug(self.model_rebin)


    def add_GP(self, GP_chip_bin=None, prior_amp=(-6,-3), prior_tau=(-5,0)):
        if self.fit_GP:
            for instrument in self.obs.keys():
                if instrument != 'photometry':
                    if GP_chip_bin is None: #use one kernel for all orders
                        self.add_parameter(f"GP_{instrument}_amp", prior=prior_amp)
                        self.add_parameter(f"GP_{instrument}_tau", prior=prior_tau)
                    else: #different kernel for each chip
                        for i in range(0, self.obs[instrument].Nchip, GP_chip_bin):
                            self.add_parameter(f"GP_{instrument}_amp_{i:02}", prior=prior_amp)
                            self.add_parameter(f"GP_{instrument}_tau_{i:02}", prior=prior_tau)
        

    def calc_scaling(self, y_model, y_data, y_cov):
        if y_cov.ndim == 2:
            # sparse Cholesky decomposition
            cov_chol = cholesky(y_cov)
            # Scale the model flux to minimize the chi-squared
            lhs = np.dot(y_model, cov_chol.solve_A(y_model))
            rhs = np.dot(y_model, cov_chol.solve_A(y_data))
            f_det = rhs/lhs
            
        elif y_cov.ndim == 1:
            # Scale the model flux to minimize the chi-squared
            lhs = np.dot(y_model, 1/y_cov * y_model)
            rhs = np.dot(y_model, 1/y_cov * y_data)
            f_det = rhs/lhs
        return f_det
    
    def calc_err_infaltion(self, y_model, y_data, y_cov):
        if y_cov.ndim == 2:
            cov_chol = cholesky(y_cov)
            chi_squared = np.dot((y_data-y_model), cov_chol.solve_A(y_data-y_model))
        elif y_cov.ndim == 1:
            chi_squared = np.sum(((y_data-y_model)**2/y_cov))
        return np.sqrt(chi_squared/len(y_data))

   
    def calc_logL(self, y_model, y_data, y_cov):
        if y_cov.ndim == 2:
            cov_chol = cholesky(y_cov)
            # Compute the chi-squared error
            chi_squared = np.dot((y_data-y_model), cov_chol.solve_A(y_data-y_model))

            # Log of the determinant (avoids under/overflow issues)
            logdet_cov = cov_chol.logdet()

        elif y_cov.ndim == 1:
            chi_squared = np.sum(((y_data-y_model)**2/y_cov))

            # Log of the determinant
            logdet_cov = np.sum(np.log(y_cov))
        # print(chi_squared, logdet_cov)

        return -(len(y_data)*np.log(2*np.pi)+chi_squared+logdet_cov)/2., chi_squared



    def prior(self, cube, ndim, nparams):
        i = 0
        indices = []
        for key in self.params:
            if self.params[key].is_free:
                a, b = self.params[key].prior
                cube[i] = a+(b-a)*cube[i]
                if key == 't_00':
                    t_i = cube[i]
                elif key.split("_")[0] == 't':
                    indices.append(i)
                i += 1

        if self.PT_is_free:
            # enforce decreasing temperatures from bottom to top layers 
            for k in indices:
                t_i = t_i * cube[k] #(1.-0.5*cube[k])
                cube[k] = t_i


    def loglike(self, cube, ndim, nparams):
        log_likelihood, chi2_reduced, N = 0., 0., 0

        if self.leave_out is not None:
            i_p = 0 # parameter count
            for pp in self.params:
                if self.params[pp].is_free:
                    self.params[pp].set_value(cube[i_p])
                    i_p += 1

            self.forward_model_pRT(leave_out=self.leave_out)
            if self.fit_telluric:
                self.forward_model_telluric()
            self.apply_rot_broaden_rv_shift()
            self.apply_instrument_broaden()
            self.apply_rebin_to_obs_wlen()
            if self.fit_poly:
                self.apply_poly_continuum()
            model_reduced = copy.copy(self.model_rebin)
            
        # draw parameter values from cube
        i_p = 0 # parameter count
        for pp in self.params:
            if self.params[pp].is_free:
                self.params[pp].set_value(cube[i_p])
                i_p += 1
                if self.debug:
                    print(f"{i_p} \t {pp} \t {self.params[pp].value}")

        self.forward_model_pRT()
        if self.fit_telluric:
            self.forward_model_telluric()
        self.apply_rot_broaden_rv_shift()
        self.apply_instrument_broaden()
        self.apply_rebin_to_obs_wlen()
        if self.fit_poly:
            self.apply_poly_continuum()


        self.flux_scaling, self.err_infaltion, self.model_reduce = {}, {}, {}
        for instrument in self.obs.keys():
            if instrument == 'photometry':
                for model_target, obs_target in zip(self.model_rebin[instrument], self.obs[instrument]):
                    chi2 = ((obs_target.f_lambda - model_target)/obs_target.f_err)**2 
                    log_det = np.log(2*np.pi*obs_target.f_err**2)
                    log_likelihood += -(chi2 + log_det)/2.
                    chi2_reduced += chi2
                    N += 1
            else:
                model_target = self.model_rebin[instrument]
                if self.leave_out is not None:
                    model_single = model_target - model_reduced[instrument]
                obs_target = self.obs[instrument]._copy()
                if self.fit_GP:
                    amp = [1e1**self.params[key].value for key in self.params \
                                        if "amp" in key.split("_") and \
                                        instrument in key.split("_")]
                    tau = [1e1**self.params[key].value for key in self.params \
                                        if "tau" in key.split("_") and \
                                        instrument in key.split("_")]

                    obs_target.make_covariance(amp, tau)
                    cov = obs_target.cov
                else:
                    cov = obs_target.err**2

                f_dets = np.ones(obs_target.Nchip)
                if self.fit_scaling:
                    for i in range(obs_target.Nchip):
                        f_det = self.calc_scaling(model_target[i], obs_target.flux[i], cov[i])
                        f_dets[i] = f_det
                        model_target[i] *= f_det
                        if self.leave_out is not None:
                            model_single[i] *= f_det

                # add model uncertainties due to the inacurate molecular line list
                if self.leave_out is not None and self.key_molecule_tolerance in self.params:
                    cov += (self.params[self.key_molecule_tolerance].value * model_single)**2
                if self.leave_out is not None and self.key_teff_tolerance in self.params:
                    cov += 1e1**self.params[self.key_teff_tolerance].value * self.temp[np.searchsorted(self.press, 1.)] / 3000.


                betas = np.ones(obs_target.Nchip)
                if self.fit_err_inflation:
                    for i in range(obs_target.Nchip):
                        beta = self.calc_err_infaltion(model_target[i], obs_target.flux[i], cov[i])
                        cov[i] *= beta**2
                        betas[i] = beta
                
                # Add to the log-likelihood
                for i in range(obs_target.Nchip):
                    log_l, chi2 = self.calc_logL(model_target[i], obs_target.flux[i], cov[i])
                    log_likelihood += log_l
                    chi2_reduced += chi2
                    N += obs_target.flux.size
            
                self.model_rebin[instrument] = model_target
                self.model_reduce[instrument] = model_single
                self.flux_scaling[instrument] = f_dets
                self.err_infaltion[instrument] = betas
        
        if self.debug:
            print("Chi2_r: ", chi2_reduced/(N-self.n_params))
            # print(self.flux_scaling, self.err_infaltion)
            # self.plot_rebin_model_debug(self.model_rebin)
            # self.plot_rebin_model_debug(self.model_reduce)

        return log_likelihood


    def setup(self, 
              obs,
              line_species,
              param_prior,
              press=None,
              N_t_knots=None, 
              chemistry='free',
              line_species_ck=None,
              fit_instrument_kernel=True,
              Lorentzian_kernel=False,
              leave_out=None,
              fit_GP=False, 
              fit_poly=1, 
              fit_scaling=True, 
              fit_err_inflation=False,
              fit_telluric=False, 
              tellu_grid_path=None,
              ):

        if press is None:
            self.press = np.logspace(-5,1,50)
        else:
            self.press = press

        self.line_species = line_species
        if line_species_ck is not None:
            self.line_species_ck = line_species_ck
        else:
            self.line_species_ck = line_species
        self.fit_GP = fit_GP
        self.fit_poly = fit_poly
        self.fit_scaling = fit_scaling
        self.fit_err_inflation = fit_err_inflation
        self.fit_telluric = fit_telluric
        self.fit_instrument_kernel = fit_instrument_kernel
        self.leave_out = leave_out

        self.add_observation(obs)
        assert self.obs, "No input observations provided"

        print("Creating pRT objects for input data...")
        self.add_pRT_objects()

        if N_t_knots is not None:
            self.N_t_knots = N_t_knots
            self.PT_is_free = True
            self.add_free_PT_model()

        if chemistry == 'free':
            self.add_free_chem_model()
            self.chem_is_free = True
        elif chemistry == 'equ':
            self.add_equ_chem_model()
        
        if fit_telluric:
            self.add_telluric_model(tellu_grid_path=tellu_grid_path)
        
        self.add_instrument_kernel(Lorentzian_kernel)
        self.add_poly_model()
        self.add_GP()
        
        self.set_parameter_priors(param_prior)

        # count number of params
        parameters = []
        for x in self.params:
            if self.params[x].is_free:
                parameters.append(self.params[x].name)
        self.n_params = len(parameters)
        print(f"{self.n_params} free parameters in total.")
        json.dump(parameters, open(self.prefix + 'params.json', 'w'))


    def set_parameter_priors(self, param_prior):
        for key in param_prior:
            if key in self.params:
                if isinstance(param_prior[key], (float, int, type(None))):
                    self.params[key].set_value(param_prior[key])
                    self.params[key].is_free = False
                else:
                    self.params[key].set_prior(param_prior[key])
            else:
                if isinstance(param_prior[key], (float, int, type(None))):
                    self.add_parameter(key, value=param_prior[key], is_free=False)
                else:
                    self.add_parameter(key, prior=param_prior[key])


    def run(self, n_live_points=500, debug=False):

        self.debug = debug

        pymultinest.run(self.loglike,
            self.prior,
            self.n_params,
            outputfiles_basename=self.prefix,
            resume = False, 
            verbose = True, 
            const_efficiency_mode = True, 
            sampling_efficiency = 0.05,
            n_live_points = n_live_points)
        

    def print_params_info(self, values):
        name_params = [self.params[x].name for x in self.params if self.params[x].is_free]
        for name, value in zip(name_params, values):
            print(f"{name}: \t\t {value}")

    def best_fit_model(self, which='mean', leave_out=None):
        self.debug = True
        a = pymultinest.Analyzer(n_params=self.n_params, outputfiles_basename=self.prefix)
        s = a.get_stats()
        best_fit = s['modes'][0][which]
        # self.print_params_info(best_fit[:self.n_params])

        name_params = [self.params[x].name for x in self.params if self.params[x].is_free]
        if leave_out is not None:
            for i, key in enumerate(name_params):
                if key == f'logX_{leave_out}':
                    best_fit[i] = -20
                    break 

        fig, axes = self.loglike(best_fit[:self.n_params], self.n_params, self.n_params)

        return self.model_rebin, fig, axes