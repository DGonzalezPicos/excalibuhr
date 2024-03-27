import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
# from astropy import stats

from scipy import signal

import pathlib
import subprocess
import shutil

from excalibuhr.data import SPEC2D
from excalibuhr import utils

class Molecfit:
    
    pixel_scale = 0.056 # arcsec/pixel (updated 2024)
    flux_unit = 0 # phot / (s *  m^2 * mum * as^2) [no conversion]
    expert_mode = False
    
    # default config values
    ftol = 1e-2 # tolerance for fit, increase to 1e-10 for robust results, 1e-2 for testing
    # xtol = ftol # tolerance for fit
    
    wlc_n = 2 # degree of Chebyshev polynomial for wavelength calibration
    continuum_n = 3 # degree of polynomial for continuum fit
    fit_lorentz = 'TRUE' # fit Lorentzian profile for telluric lines
    
    file_label = 'STANDARD'
    scale_pwv = False
    sgwl = 15 # smoothing length for Savitzky-Golay filter (used for sigma-clipping and quality control)
    
    
    
    def __init__(self, workpath, night, clean_start=False):
        
        # Directory management
        self.path = pathlib.Path(workpath) / night
        
        
        
        self.molecfit_dir = self.path / 'out/molecfit'
        
        if clean_start:
            shutil.rmtree(self.molecfit_dir)
            
            
        molecfit_subdirs = ['input', 'output', 'config', 
                            'model', 'calctrans',
                            'correct']
        for subdir in molecfit_subdirs:
            (self.molecfit_dir / subdir).mkdir(parents=True, exist_ok=True)
            setattr(self, subdir+'_dir', self.molecfit_dir / subdir)
            
    def __repr__(self):
        out = f'******** Molecfit object ********'
        out += f'\nWorkpath: {self.path}'
        out += f'\nNight: {self.path.name}'
        # out += f'\nTarget: {self.file_label}'
        
        out += f'\nAttributes: {list(self.__dict__.keys())}'
        return out
            
    def prepare_data(self, spec, Nedge=30, normalize=True, clip=1.0,
                     telluric_standard=False, file_label=None, # deprecated
                     file_name=None, # New (2024-03-26)
                     airmass=None,
                     pwv=None,
                     mask_brgamma=True, plot=False):
        
        print(f'[prepare_data] Preparing input CRIRES data as multiextension FITS files...')
        self.Nedge = Nedge
        
        """DEPRECATED: Prepare input data for molecfit.
        self.file_label = 'STANDARD' if telluric_standard else 'SCIENCE'
        if file_label is not None:
            self.file_label = file_label
            
        savename = self.input_dir / f'{self.file_label}.fits'
        """
        # self.file
        # self.file_name = self.input_dir / file_name # New (2024-03-26)
        self.update_file_name(file_name)
        
        if airmass is not None:
            # update airmass in header
            spec.header['ESO TEL AIRM START'] = airmass
            spec.header['ESO TEL AIRM END'] = airmass
            print(f'[prepare_data] Airmass updated to {airmass}')
            
        if pwv is not None:
            # update pwv in header
            spec.header['ESO TEL AMBI IWV START'] = pwv
            spec.header['ESO TEL AMBI IWV END'] = pwv
            print(f'[prepare_data] PWV updated to {pwv}')
            
        primary_hdu = fits.PrimaryHDU(header=spec.header)
        n_chip, n_pix = spec.flux.shape
        
        # Copy header from the primary HDU
        hdu_out = fits.HDUList([primary_hdu])
        hdu_win = fits.HDUList([primary_hdu])
        hdu_wex = fits.HDUList([primary_hdu])
        hdu_atm = fits.HDUList([primary_hdu])
        hdu_con = fits.HDUList([primary_hdu])
        hdu_cor = fits.HDUList([primary_hdu])
        # Initialize empty lists
        wmin = []
        wmax = []
        map_chip = np.arange(1, n_chip+1) # starting at 1
        map_ext = np.arange(0, n_chip+1) # starting at 0
        wlc_fit = np.ones_like(map_chip)
        
        # Initialize lists for WAVE_EXCLUDE.fits
        wave_exclude_lower = []
        wave_exclude_upper = []
        map_chip_exclude = []
        # wlc_fit_exclude = [] # generate at the end with same shape as map_chip_exclude filled with ones
        
        
        self.continuum_constant = np.nanmedian(spec.flux[:,Nedge:-Nedge]) # to be used in config file
        if normalize:
            self.normalize_constant = np.copy(self.continuum_constant) # store for later use
            spec.flux /= self.continuum_constant
            spec.err /= self.continuum_constant
            # self.continuum_constant = np.nanmedian(spec.flux[:,Nedge:-Nedge]) # to be used in config file
            
        if plot:
            # plt.figure(7, figsize=(12,12))
            # fig, ax = plt.subplots(7,1, figsize=(14,12))
            fig, ax = utils.fig_order_subplots(n_chip//3, 3,
                                               xlabel='Wavelength [um]', 
                                               )
            colors = plt.cm.viridis(np.linspace(0,1,n_chip))
            # vspan with no edge
            axvspan_args = dict(facecolor='r', alpha=0.15, edgecolor='none', lw=0)
            # plt.title('Input CRIRES data')
            # plt.xlabel('Wavelength [nm]')
            # plt.ylabel('Flux')
            # plt.grid(True)
            ax[-1].set_xlabel('Wavelength [um]')
            
        for chip in range(n_chip):
            
            # wavelength to micron
            wave = spec.wlen[chip] * 1e-3
            flux = np.nan_to_num(spec.flux[chip], 0.0)
            err = spec.err[chip]
            
            # Get the wavelength range
            wmin.append(wave[Nedge])
            wmax.append(wave[-Nedge])
                       
            if mask_brgamma and abs(np.median(wave) - 2.166) < 0.5 * (wmax[-1]-wmin[-1]):
                # mask brackett gamma line
                wave_exclude_lower.append(2.162)
                wave_exclude_upper.append(2.168)
                map_chip_exclude.append(chip+1)
                if plot:
                    # plt.axvspan(wave_exclude_lower[-1], wave_exclude_upper[-1], color='red', alpha=0.2)
                    ax[chip//3].axvspan(wave_exclude_lower[-1], wave_exclude_upper[-1], **axvspan_args)
                    
            # mask telluric lines
            # mask emission lines
            deep_lines = flux < 0.5 * np.median(flux)
            mask = flux - np.median(flux[~deep_lines]) > clip * np.std(flux[~deep_lines])
            # grow mask by 3 pixels
            mask = np.convolve(mask, np.ones(3), mode='same').astype(bool)
            # include nans in mask
            mask = np.logical_or(mask, np.isnan(flux)) # mask nans
            
            print(f'[prepare_data] Chip {chip+1}: {np.sum(mask)} emission lines clipped')
            # find lower edges of masked regions
            change_sign = np.diff(mask.astype(int))
            idx_low = np.where(change_sign == 1)[0]
            idx_up = np.where(change_sign == -1)[0]
            # add to WAVE_EXCLUDE.fits
            for i in range(len(idx_low)):
                wave_exclude_lower.append(wave[idx_low[i]])
                wave_exclude_upper.append(wave[idx_up[i]])
                map_chip_exclude.append(chip+1)
                if plot:
                    ax[chip//3].axvspan(wave[idx_low[i]], wave[idx_up[i]], **axvspan_args)
            
            if plot:
                ax[chip//3].axvspan(wave[0], wave[Nedge], **axvspan_args)
                ax[chip//3].axvspan(wave[-Nedge], wave[-1], **axvspan_args)
                
                flux_plot = np.copy(flux) # plot ignoring edges of chip
                flux_plot[:Nedge] = np.nan
                flux_plot[-Nedge:] = np.nan
                ax[chip//3].plot(wave, flux_plot, lw=1.0, color=colors[chip])
                ax[chip//3].fill_between(wave, flux_plot-err, flux_plot+err, alpha=0.5, color=colors[chip])
                    
            # create FITS columns
            col1 = fits.Column(name='WAVE', format='D', array=wave)
            col2 = fits.Column(name='FLUX', format='D', array=flux)
            col3 = fits.Column(name='ERR', format='D', array=err)
            
            # Create FITS table with WAVE, SPEC, and ERR
            table_hdu = fits.BinTableHDU.from_columns([col1, col2, col3])
            # Append table HDU to output of HDU list
            hdu_out.append(table_hdu)
        
           
        # Save SCIENCE data 
        self.wfits(hdu_out, self.file_name, subdir='input')  
        
        # Create FITS file with WAVE_INCLUDE
        col_wmin = fits.Column(name="LOWER_LIMIT", format="D", array=wmin)
        col_wmax = fits.Column(name="UPPER_LIMIT", format="D", array=wmax)
        col_map = fits.Column(name="MAPPED_TO_CHIP", format="I", array=map_chip)
        col_wlc = fits.Column(name="WLC_FIT_FLAG", format="I", array=wlc_fit)
        col_cont = fits.Column(name="CONT_FIT_FLAG", format="I", array=wlc_fit)
        columns = [col_wmin, col_wmax, col_map, col_wlc, col_cont]
        table_hdu = fits.BinTableHDU.from_columns(columns)
        hdu_win.append(table_hdu)
        self.wfits(hdu_win, 'WAVE_INCLUDE.fits', subdir='input')
        
        # Create FITS with WAVE_EXCLUDE
        wlc_fit_exclude = np.ones_like(map_chip_exclude)
        col_wmin = fits.Column(name="LOWER_LIMIT", format="D", array=wave_exclude_lower)
        col_wmax = fits.Column(name="UPPER_LIMIT", format="D", array=wave_exclude_upper)
        col_map = fits.Column(name="MAPPED_TO_CHIP", format="I", array=map_chip_exclude)
        col_wlc = fits.Column(name="WLC_FIT_FLAG", format="I", array=wlc_fit_exclude * 0.)
        col_cont = fits.Column(name="CONT_FIT_FLAG", format="I", array=wlc_fit_exclude * 0.)
        columns = [col_wmin, col_wmax, col_map, col_wlc, col_cont]
        table_hdu = fits.BinTableHDU.from_columns(columns)
        hdu_wex.append(table_hdu)
        self.wfits(hdu_wex, 'WAVE_EXCLUDE.fits', subdir='input')
        
         # Create FITS file with MAPPING_ATMOSPHERIC
        name = "ATM_PARAMETERS_EXT"
        col_atm = fits.Column(name=name, format="K", array=map_ext)
        table_hdu = fits.BinTableHDU.from_columns([col_atm])
        hdu_atm.append(table_hdu)
        self.wfits(hdu_atm, 'MAPPING_ATMOSPHERIC.fits', subdir='input')
        
        # Create FITS file with MAPPING_CONVOLVE
        name = "LBLRTM_RESULTS_EXT"
        col_conv = fits.Column(name=name, format="K", array=map_ext)
        table_hdu = fits.BinTableHDU.from_columns([col_conv])
        hdu_con.append(table_hdu)
        self.wfits(hdu_con, 'MAPPING_CONVOLVE.fits', subdir='input')
        
        # Create FITS file with MAPPING_CORRECT
        name = "TELLURIC_CORR_EXT"
        col_corr = fits.Column(name=name, format="K", array=map_ext)
        table_hdu = fits.BinTableHDU.from_columns([col_corr])
        hdu_cor.append(table_hdu)
        self.wfits(hdu_cor, f'MAPPING_CORRECT.fits', subdir='input')
        
        if plot: # savefig
            # plt.xlim(np.min(spec.wlen) * 1e-3, np.max(spec.wlen) * 1e-3)
            # plt.ylim(0, 1.01 * np.max(spec.flux))
            # set xlim for all axes
            wmin = np.min(spec.wlen, axis=-1)[np.arange(0, len(spec.wlen), 3)]
            wmax = np.max(spec.wlen, axis=-1)[np.arange(2, len(spec.wlen)+1, 3)]
            xpad = 0.0005
            assert np.shape(wmin) == np.shape(wmax), f'wmin {np.shape(wmin)} and wmax {np.shape(wmax)} must have same shape'
            for i, axi in enumerate(ax):
                axi.set_xlim(wmin[i] * 1e-3 - xpad, wmax[i] * 1e-3 + xpad)
                # axi.set_ylim(0, 1.01 * np.max(s2d.flux[np.arange(i, len(s2d.wlen), 3)]))
                axi.set_ylabel('Flux')
                axi.grid(True, alpha=0.2)
            
            out_fig = self.input_dir / file_name.replace('.fits', '.pdf')
            fig.savefig(out_fig, dpi=200)
            print(f'[prepare_data] Plot saved: {out_fig}')
            plt.close()
        return self
    
    def wfits(self, hdu, file, subdir='input'):
        """Write FITS file."""
        # assert isinstance(hdu, fits.HDUList), 'hdu must be a fits.HDUList'
        # assert isinstance(file, str), 'file must be a string'
        assert getattr(self, subdir+'_dir').exists(), f'{subdir} directory does not exist'
        outfile = getattr(self, subdir+'_dir') / file
        hdu.writeto(outfile, overwrite=True)
        print(f'- FITS file saved: {outfile}')
    
    def replace_config(self, recipe='molecfit_model'):
        
        # create dictionary with tuple (before, after)
        molecfit_model_dict = {
            'USE_INPUT_KERNEL': ('TRUE', 'FALSE'),
            'LIST_MOLEC' : ('NULL', 'H2O,CO2,CO,CH4,N2O'),
            'FIT_MOLEC'  : ('NULL', '1,1,1,1,1'),
            'REL_COL'    : ('NULL', '1.0,1.0,1.0,1.0,1.0'),
            'MAP_REGIONS_TO_CHIP': ('1', 'NULL'),
            'COLUMN_LAMBDA': ('lambda', 'WAVE'),
            'COLUMN_FLUX': ('flux', 'FLUX'),
            'COLUMN_DFLUX': ('NULL', 'ERR'),
            'PIX_SCALE_VALUE': ('0.086', f'{self.pixel_scale}'),
            'CONTINUUM_CONST': ('1.0', f'{self.continuum_constant}'),
            'FTOL' : ('1e-10', f'{self.ftol}'),
            'XTOL' : ('1e-10', f'{self.ftol}'), # same as ftol
            'CHIP_EXTENSIONS' : ('FALSE', 'TRUE'),
            'FIT_WLC' : ('0', 'NULL'),
            'WLC_N'   : ('1', f'{self.wlc_n}'),
            'WLC_CONST' : ('-0.05', '0.00'),
            'FIT_RES_LORENTZ' : ('TRUE', self.fit_lorentz),
            'VARKERNEL' : ('FALSE', 'TRUE'),
            'CONTINUUM_N' : ('0', f"{self.continuum_n}")
        }            
        
        molecfit_calctrans_dict = {
            'USE_INPUT_KERNEL': ('TRUE', 'FALSE'),
            'CHIP_EXTENSIONS' : ('FALSE', 'TRUE'),
            'SCALE_PWV' : ('none', 'auto' if self.scale_pwv else 'none'),
            'SGWL' : ('15', f'{self.sgwl}'),
        }
        
        molecfit_correct_dict = {
            'CHIP_EXTENSIONS' : ('FALSE', 'TRUE'),
            'WLC_REF' : ('DATA', 'MODEL'), # use wavesol from molecfit model
        }
        if recipe == 'molecfit_model':
            replace_dict = molecfit_model_dict
        elif recipe == 'molecfit_calctrans':
            replace_dict = molecfit_calctrans_dict
        elif recipe == 'molecfit_correct':
            replace_dict = molecfit_correct_dict
        return replace_dict
    def create_config(self, recipe='molecfit_model', stdout=None, **kwargs):
        
        config_file = self.config_dir / f"{recipe}.rc"

        esorex = ["esorex", f"--create-config={config_file}", recipe]
        print(f"[create_config] Creating file: config/{recipe}.rc", end="", flush=True)
        subprocess.run(esorex, cwd=self.config_dir, stdout=stdout, check=True)
        
        with open(config_file, "r", encoding="utf-8") as open_config:
            config_text = open_config.read()
            
            # config_text = text_replace(config_text, 'USE_INPUT_KERNEL', 'TRUE', 'FALSE')
            replace_dict = self.replace_config(recipe=recipe)
            
            for label, (before, after) in replace_dict.items():
                config_text = config_text.replace(f'{label}={before}', f'{label}={after}')
            
            if self.expert_mode and recipe == 'molecfit_model':
                config_text = config_text.replace('EXPERT_MODE=FALSE', 'EXPERT_MODE=TRUE')
            # save results
            with open(config_file, "w", encoding="utf-8") as open_config:
                open_config.write(config_text)
                print(f'[create_config] {config_file} updated!')
        return self
        
    
    def molecfit_model(self, verbose=True, expert_mode=False):
                
        self.expert_mode = expert_mode
        # Create SOF file
        print("Creating SOF file:")

        sof_file = pathlib.Path(self.model_dir / "files.sof")
        # sof_tags = ["SCIENCE", "WAVE_INCLUDE", "WAVE_EXCLUDE"]
        sof_tags = ["WAVE_INCLUDE", "WAVE_EXCLUDE"]            
            
        with open(sof_file, "w", encoding="utf-8") as sof_open:
            [self.write_sof(sof_open, tag) for tag in sof_tags]
            # add fits extension if not present
           
            if not str(self.file_name).endswith('.fits'):
                self.file_name = self.file_name + '.fits'
            sof_open.write(f"{self.input_dir / f'{self.file_name}'} SCIENCE\n")
                
            expert_file = self.model_dir / f"BEST_FIT_PARAMETERS.fits"
            if self.expert_mode and expert_file.exists():
                print(f'[molecfit_model] Expert mode: reading initial parameters from {expert_file}')
                sof_open.write(f"{expert_file} INIT_FIT_PARAMETERS\n")
                
            
        # Create EsoRex configuration file if not found
        self.create_config("molecfit_model", verbose)
        
        # Read molecules from config file
        config_file = self.config_dir / "molecfit_model.rc"

        with open(config_file, "r", encoding="utf-8") as open_config:
            config_text = open_config.readlines()

        print("\nSelected molecules:")
        for line_item in config_text:
            if line_item[:10] == "LIST_MOLEC":
                for mol_item in line_item[11:].split(","):
                    print(f"   - {mol_item}")
                    
        # Run EsoRex
        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={self.model_dir}",
            "molecfit_model",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=self.model_dir, stdout=stdout, check=True)            
        return self 
    
    def write_sof(self, sof_open, label, subdir='input'):
        file = getattr(self, f'{subdir}_dir') / f'{label}.fits'
        sof_open.write(f"{file} {label}\n")
        return sof_open
    
    def update_file_name(self, file_name=None):
        
        # assert file_name is not None, 'file_name must be provided'
        if file_name is None:
            return self
        # convert to str if pathlib.Path
        if isinstance(file_name, pathlib.Path):
            file_name = file_name.name
        self.file_name = str(self.input_dir / file_name)
        self.file_name_stem = (self.input_dir / file_name).stem

        self.file_name_out = str(self.output_dir / file_name)
        return self
        
    def molecfit_calctrans(self, scale_pwv=False, 
                           file_name=None,
                           sgwl=10, 
                           verbose=True):
        
        print(f'[molecfit_calctrans] Creating SOF file:')
        sof_file = self.calctrans_dir / 'files.sof'
        sof_tags = ['MAPPING_ATMOSPHERIC', 'MAPPING_CONVOLVE', 
                    'ATM_PARAMETERS', 'MODEL_MOLECULES', 'BEST_FIT_PARAMETERS']
        
        self.scale_pwv = scale_pwv
        self.sgwl = int(sgwl) # smoothing length for Savitzky-Golay filter (used for sigma-clipping and quality control)
        # Update attribute
        self.update_file_name(file_name)
        # if file_name is not None:
        #     self.file_name = file_name
        # self.file_label = 'SCIENCE' if self.scale_pwv else self.file_label
        
        # if file_label is not None:
        #     self.file_label = file_label
        #     self.scale_pwv = False if "STANDARD" in file_label else True
        #     print(f'Setting scale_pwv to {self.scale_pwv} for {file_label}')
    
        with open(sof_file, 'w', encoding='utf-8') as sof_open:
            for tag in sof_tags:
                subdir = 'input' if tag in ['MAPPING_ATMOSPHERIC', 'MAPPING_CONVOLVE'] else 'model'
                sof_open = self.write_sof(sof_open, tag, subdir=subdir)
            sof_open.write(f"{self.file_name} SCIENCE\n")
                
        self.create_config('molecfit_calctrans', verbose, scale_pwv=scale_pwv, sgwl=sgwl)
        config_file = self.config_dir / 'molecfit_calctrans.rc'
        
        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={self.output_dir}",
            "molecfit_calctrans",
            sof_file,
                ]
        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=self.calctrans_dir, stdout=stdout, check=True)
        # rename files to include self.file_label
        rename_files = ['TELLURIC_DATA', 'CALCTRANS_ATM_PARAMETERS', 'CALCTRANS_CHIPS_COMBINED', 'LBLRTM_RESULTS']
        for file in rename_files:
            old_file = self.output_dir / f'{file}.fits'
            if (old_file).exists():
                # new_file = self.output_dir / f'{file}_{self.file_label}.fits'
                new_file = self.output_dir / f'{file}_{self.file_name_stem}.fits'
                old_file.rename(new_file)
                print(f'[molecfit_calctrans] Renaming {file}.fits to {file}_{self.file_name_stem}.fits')

        if not verbose:
            print(" [DONE]") 
        return self
    
    
    # @deprecated
    # def molecfit_correct(self, telluric_standard=False, verbose=True):
        
    #     sof_file = self.correct_dir / 'files.sof'
    #     # self.file_label = 'SCIENCE' if not telluric_standard else 'STANDARD'
        
        
        
    #     with open(sof_file, 'w', encoding='utf-8') as sof_open:
    #         # sof_open = self.write_sof(sof_open, 'SCIENCE', subdir='input')
    #         sof_open.write(f"{self.input_dir / f'{self.file_name_stem}.fits'} SCIENCE\n")
    #         # sof_open.write(f"{self.input_dir / f'MAPPING_CORRECT_{self.file_label}.fits'} MAPPING_CORRECT\n")
    #         sof_open = self.write_sof(sof_open, 'MAPPING_CORRECT', subdir='input')
    #         # sof_open = self.write_sof(sof_open, f'TELLURIC_CORR_{self.file_label}', subdir='output')
    #         sof_open.write(f"{self.output_dir / f'TELLURIC_CORR_{self.file_name_stem}.fits'} TELLURIC_CORR\n")
    #         sof_open.write(f"{self.output_dir / f'TELLURIC_DATA_{self.file_name_stem}.fits'} TELLURIC_DATA\n")

            
    #     self.create_config('molecfit_correct', verbose)
    #     print()
    #     config_file = self.config_dir / 'molecfit_correct.rc'
    #     esorex = [
    #         "esorex",
    #         f"--recipe-config={config_file}",
    #         f"--output-dir={self.output_dir}",
    #         "molecfit_correct",
    #         sof_file,
    #     ]

    #     if verbose:
    #         stdout = None
    #     else:
    #         stdout = subprocess.DEVNULL
    #         print("Running EsoRex...", end="", flush=True)

    #     subprocess.run(esorex, cwd=self.correct_dir, stdout=stdout, check=True)
        
    #     rename_files = ['TELLURIC_CORR']
    #     for f in rename_files:
    #         print(f'[molecfit_correct] Renaming {f}.fits to {f}_{self.file_name_stem}.fits')
    #         old_file = self.output_dir / f'{f}.fits'
    #         new_file = self.output_dir / f'{f}_{self.file_name_stem}.fits'
    #         old_file.rename(new_file)

    #     if not verbose:
    #         print(" [DONE]")
            
    #     return None
    
    def read_fit_results(self):
        file = self.model_dir / f"BEST_FIT_PARAMETERS.fits"
        with fits.open(file) as hdul:
            # print(hdul.info())
            data = hdul[1].data
            self.fit_results =  {k: (data['value'][i],data['uncertainty'][i]) for i,k in enumerate(data['parameter'])}
            
        
        self.fwhm_gauss = self.fit_results['gaussfwhm'][0] # FWHM of the Gaussian kernel
        self.fwhm_lorentz = self.fit_results['lorentzfwhm'][0] # FWHM of the Lorentzian kernel
        c = 299792.458 # speed of light in km/s
        if self.fwhm_gauss > self.fwhm_lorentz:
            self.fwhm = self.fwhm_gauss
            self.kernel = 'gauss'   
        
            # convert fwhm to sigma
            sigma = self.fwhm / (2 * np.sqrt(2 * np.log(2)))
            sigma_err = self.fit_results['gaussfwhm'][1] / (2 * np.sqrt(2 * np.log(2)))
            # convert to spectral resolution 
            # self.resolution = 299792.458 / sigma
            resolution = c / sigma
            resolution_err = (resolution**2/c)  * sigma_err
            self.fit_results['resolution'] = (resolution, resolution_err)

        elif self.fwhm_lorentz > self.fwhm_gauss:
            # self.fwhm = 2. * self.fwhm_lorentz
            sigma = self.fwhm_lorentz
            sigma_err = self.fit_results['lorentzfwhm'][1]
            self.kernel = 'lorentz'
            resolution = c / self.fwhm_lorentz
            resolution_err = (resolution**2/c) * sigma_err
            self.fit_results['resolution'] = (resolution, resolution_err)


            
        return self.fit_results
    
    def load_output_data(self, file_name=None):    
        # self.file_label = file_label
        self.update_file_name(file_name)
        file_tell = self.output_dir / f'TELLURIC_DATA_{self.file_name_stem}.fits'        
        with fits.open(file_tell) as hdul:
            # print(hdul.info())
            data = hdul[1].data
            
        self.add_continuum(data)
        
        with fits.open(file_tell) as hdul:
            # print(hdul.info())
            data = hdul[1].data


        self.reshape_data(data)
        self.fix_br_gamma_continuum()
           
        return data
    
    def load_science_header(self):
        # file_science = self.input_dir / f'{self.file_label}.fits'
        file_science = self.file_name
        with fits.open(file_science) as hdul:
            print(hdul.info())
            # data = hdul[1].data
            header = hdul[0].header
            # print(header)
        return header
            
    def load_wave_exclude(self):
        file_exclude = self.input_dir / 'WAVE_EXCLUDE.fits'

        # def load_wave_exclude(file):
        
        with fits.open(file_exclude) as hdul:
            # print(hdul.info())  
            wave_exclude = hdul[1].data
            self.lower = wave_exclude['LOWER_LIMIT']
            self.upper = wave_exclude['UPPER_LIMIT']
            
        # create mask from wave_exclude
        self.mask_exclude = np.zeros_like(self.mwave, dtype=bool)
        for i in range(len(self.lower)):
            self.mask_exclude = np.logical_or(self.mask_exclude, 
                                              np.logical_and(self.mwave > self.lower[i], 
                                                             self.mwave < self.upper[i]))
        # add edges of each chip to mask
        self.mask_exclude[:, :self.Nedge] = True
        self.mask_exclude[:, -self.Nedge:] = True
        
        return self      
    
    def nan_mask(self, tell_threshold=0.1, tell_grow=11):
        # assert hasattr(self, 'mask_exclude'), 'Run load_wave_exclude() first'
        # assert hasattr(self, 'mtrans'), 'Run load_output_data() first'
        if not hasattr(self, 'mtrans'):
            self.load_output_data()
            
        if not hasattr(self, 'mask_exclude'):
            self.load_wave_exclude()
            
        for i in range(self.mwave.shape[0]):
            # mask telluric lines
            deep_lines = self.mtrans[i] < tell_threshold
            if tell_grow > 1:
                deep_lines = np.convolve(deep_lines, np.ones(tell_grow), mode='same').astype(bool)
            self.mask_exclude[i] = np.logical_or(self.mask_exclude[i], deep_lines)
        return self
    
    def residuals_outliers(self, sigma=3, grow=5):
        '''Find outliers in the residuals of the standard star after dividing by the 
        telluric model'''
        assert hasattr(self, 'mtrans'), 'Run load_output_data() first'
        assert hasattr(self, 'mask_exclude'), 'Run load_wave_exclude() first'
        # TODO: finish this function
        self.outliers = np.zeros_like(self.mwave, dtype=bool)
        for i in range(self.mwave.shape[0]):
            # res = self.flux[i] / (self.mtrans[i] * self.mcontinuum[i])
            res = self.flux[i] / self.mcont[i] / self.mtrans[i]
            # find outliers
            nans = self.mask_exclude[i]
            median, std = np.nanmedian(res[~nans]), np.nanstd(res[~nans])
            mask_outliers = np.abs(res - median) > sigma * std
            if grow > 1:
                mask_outliers = np.convolve(mask_outliers, np.ones(grow), mode='same').astype(bool)
            self.outliers[i] = mask_outliers
            
        return self
    
    @staticmethod
    def get_continuum(x, parameters, chip=1, fit_order=1):
        fit_order = int(fit_order)
        x_cen = x - np.mean(x) # center around 0, see documentation section 8.5.1
        coefs = [parameters[f'Range {chip}, chip {chip}, coef {i}'][0] for i in range(fit_order+1)][::-1]
        # coefficients must be from HIGHER to lower degree
        return np.polyval(coefs, x_cen)
    
    def read_config_file(self):
        config_file = self.config_dir / 'molecfit_model.rc'
        # load file
        read = np.loadtxt(config_file, dtype=str, delimiter='=')
        # make a dictionary
        self.config = dict(zip(read[:,0], read[:,1]))
        return self
            
    
    def add_continuum(self, data, recompute=False, ignore_br_gamma=False):
        file_tell = self.output_dir / f'TELLURIC_DATA_{self.file_name_stem}.fits'
        print(f'[add_continuum] Reading {file_tell}')
        parameters_dict = self.read_fit_results()
        # data = self.load_data(reshape=False)
        with fits.open(file_tell) as hdul:
            hdul.info()
            primary = hdul[0].header
            data = hdul[1].data
            columns = hdul[1].columns
            # print(columns.name)
        if 'mcontinuum' in data.columns.names and not recompute:
            return self
        if 'mcontinuum' in data.columns.names and recompute:
            # remove column with mcontinuum
            data = data.columns.del_col('mcontinuum')
            columns = data.columns
        
        continuum = np.zeros_like(data['mlambda'])
        self.read_config_file()
        chips = np.unique(data['chip'])
        # br_gamma_chip = int(np.argmin(np.abs(data['mlambda'] - 2.166)))
        # assert br_gamma_chip > 0, 'Br-gamma not found in data'
        for i, chip in enumerate(chips):
            mask = data['chip'] == chip
            wave = data['mlambda'][mask]
            
            
            continuum[mask] = self.get_continuum(wave, 
                                        parameters=parameters_dict, 
                                        chip=chip, 
                                        fit_order=self.config['CONTINUUM_N'])
            # if ignore_br_gamma and chip==br_gamma_chip:
            #     continuum[mask] = np.ones_like(continuum[mask])
        # add continuum to data as new column
        continuum_col = fits.ColDefs([fits.Column(name='mcontinuum', format='D', array=continuum)])
        # create new hdul
        hdul = fits.BinTableHDU.from_columns(data.columns + continuum_col, header=primary)   
        self.wfits(hdul, f'TELLURIC_DATA_{self.file_name_stem}.fits', subdir='output')
        print(f'[add_continuum] Saved {self.output_dir}/TELLURIC_DATA_{self.file_name_stem}.fits')
        return self
    
       
    def plot_output(self, 
                    # telluric_standard=False, 
                    # file_label=None, 
                    savefig=True):
        
        # self.file_label = 'STANDARD' if telluric_standard else 'SCIENCE'
        # if file_label is not None:
        #     self.file_label = file_label
        
        # assert hasattr(self, file_name), 'Run load_output_data() first'

        ### Load Wave Exclude ###
        if not hasattr(self, 'mask_exclude'):
            self.load_wave_exclude()

        ### Load Fit Results ###
        parameters_dict = self.read_fit_results()

        ### Load config file ###
        # read the wavelength and continuum fit order from the config file
        self.read_config_file()
        assert hasattr(self, 'mwave'), 'Run load_output_data() first'
        # if not hasattr(self, 'mwave'):
        #     self.load_data(reshape=True)

        # fig, ax = plt.subplots(7,1, figsize=(14,12))
        # n_orders = self.mwave.shape[0] // 3
        n_orders = self.mwave.shape[0]
        fig, ax = plt.subplots(
            figsize=(14,2.5*n_orders*2), nrows=n_orders*3, 
            gridspec_kw={'hspace':0, 'height_ratios':[1,1/3,1/5]*n_orders, 
                        'left':0.1, 'right':0.95, 
                        'top':(1-0.02*7/(n_orders*3)), 
                        'bottom':0.035*7/(n_orders*3), 
                        }
            )
        # colors = plt.cm.viridis(np.linspace(0,1,n_orders))
        lw = 1.2
        molecfit_color = 'orange'
        ax_spec = [ax[i*3] for i in range(n_orders)]
        ax_res = [ax[i*3+1] for i in range(n_orders)]
        # remove empty
        [ax[i*3+2].remove() for i in range(n_orders)]
        
        axvspan_args = dict(facecolor='r', alpha=0.15, edgecolor='none', lw=0)

        
        # TODO: manage continuum properly for standard and science
        """
        if 'STANDARD' in self.file_label:
            median = 1.0
            median_continuum = 1.0
        else:
            # median = np.median(data['cflux']) # corrected flux
            # median_continuum = np.median(data['mcontinuum'])
            median = np.median(self.cflux_savgol)
            median_continuum = np.median(self.mcont)
        """
            
        # median = np.median(self.cflux_savgol)
        # median = np.median(self.flux)
        # median_continuum = np.median(self.mcont)
        for i, chip in enumerate(range(self.mwave.shape[0])):            
            labels = [self.file_name_stem, 'Molecfit', 'Residuals'] if i == 0 else ['', '','']
            
            ax_spec_i = ax_spec[i]
            ax_res_i = ax_res[i]
            
            median = np.median(self.cflux_savgol[i])
            median_continuum = np.median(self.mcont[i])
            ax_spec_i.plot(self.mwave[i], self.flux[i] / median, lw=lw, alpha=.7, 
                           label=labels[0], color='k')
            
            ax_spec_i.plot(self.mwave[i], self.mcont[i] * self.mtrans[i] / median_continuum,
                           c='orange', lw=lw, alpha=.9, label=labels[1]) # WARNING: Multiply by continuum????
            ax_spec_i.plot(self.mwave[i], self.mcont[i] / median_continuum, 
                           c='lime', lw=2., alpha=0.6)
            res = (self.flux[i] / median) / (self.mtrans[i] * self.mcont[i] / median_continuum)
            res_err = self.err[i] / median / self.mcont[i] / self.mtrans[i]
            res[self.mask_exclude[i]] = np.nan
            ax_res_i.plot(self.mwave[i], res, lw=lw, alpha=.7,
                          label=labels[2], color='k')
            # plot edge region
            ax_spec_i.axvspan(self.mwave[i][0], self.mwave[i][self.Nedge], **axvspan_args)
            ax_spec_i.axvspan(self.mwave[i][-self.Nedge], self.mwave[i][-1], **axvspan_args)

            xlim = (np.min(self.mwave[i]), np.max(self.mwave[i]))
            xpad = 0.005 * (xlim[1] - xlim[0])
            ax_spec_i.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
            ax_res_i.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
            if hasattr(self, 'outliers'):
                f_outliers = np.where(self.outliers[i], res, np.nan)
                ax_res_i.plot(self.mwave[i], f_outliers, 'ro', ms=1, alpha=0.5)
            
        
        for lo, up in zip(self.lower, self.upper):
            # find chip 
            chip = np.argmin(np.abs(np.median(self.wave, axis=1) - (lo+up)/2.))
            ax_spec_i.axvspan(lo, up, **axvspan_args)
        

        # ax[0].legend()
        ax_res[0].set(ylabel='Residuals')
        ax_spec[0].legend()
        # ax[0].set_ylim(bottom=0.0)
        # ax[1].set_ylim(bottom=0.0)
        # wmin, wmax = np.min(data['mlambda']), np.max(data['mlambda'])
        
        # title = f'ftol = {self.ftol:.2e}'
        # ax[0].set(xlim=(wmin, wmax), ylabel='Flux [ADU]')
            # title=f'Order {orders_str}')
        # ax[1].set(xlabel='Wavelength [nm]', ylabel='Residuals')
        
        ax_res[-1].set_xlabel('Wavelength [um]')
        # find global minimum and maximum from set of axes in ax_res
        ymin = np.min([ax.get_ylim()[0] for ax in ax_res])
        ymax = np.max([ax.get_ylim()[1] for ax in ax_res])
        for axi in ax_res:
            axi.axhline(1, c=molecfit_color, ls='-', lw=lw, alpha=0.8, zorder=0)
            axi.set_ylim(ymin, ymax)
            
        if savefig:
            # outfig = self.output_dir / f'{self.file_label}_ftol{self.ftol:.2e}.png'
            out_fig = self.output_dir / f'{self.file_name_stem}_logftol{np.log10(self.ftol):.0f}.pdf'
            fig.savefig(out_fig, dpi=200)
            print(f'  - Saved {out_fig}')
        # plt.show()
        # plt.close()

        return None
    
    def reshape_data(self, data, fix_br_gamma=True, debug=False):
        
        # columns: chip, lambda, flux, weight, mlambda, mtrans, mweight, cflux, qual, mcontinuum
        # reshape data to have one column per chip
        chips = np.unique(data['chip'])
        self.n_chips = len(chips)
        npix_per_chip = len(data['mlambda']) // len(chips)
        print(f'[reshape_data] {len(data["mlambda"])} pixels, {len(chips)} chips, {npix_per_chip} pixels per chip')
        
        self.wave, self.flux, self.err, self.mwave, self.mtrans, self.mcont, self.cflux_savgol = (np.zeros((len(chips), npix_per_chip))
                                                                                        for i in range(7))
        
        
        for chip in chips:
            mask = data['chip'] == chip
            self.wave[chip-1] = data['lambda'][mask]
            self.flux[chip-1] = data['flux'][mask]
            self.err[chip-1] = 1. / data['weight'][mask]
            self.mwave[chip-1] = data['mlambda'][mask]
            self.mtrans[chip-1] = data['mtrans'][mask]
            self.mcont[chip-1] = data['mcontinuum'][mask]
            self.cflux_savgol[chip-1] = data['cflux_savgol'][mask]
        
        if self.fix_br_gamma_continuum:
            self.fix_br_gamma_continuum()
        print(f'[reshape_data] shape in: {data["mlambda"].shape}, shape out: {self.mwave.shape}')
        return self
            
    
    @staticmethod
    def find_outliers(y, y_ref, sigma=5.0, maxiters=5, grow_n=5):
        ''' Identify outliers on `y` by comparing it to `y_ref`'''
    
        outliers = np.isnan(y)
        y_corr = np.where(outliers, y_ref, y)
        
        for i in range(maxiters):
            # sigma clipping
            diff = np.abs(y_corr - y_ref)
            outliers_i = np.abs(diff-np.nanmean(diff)) > sigma * np.nanstd(diff)
            if np.sum(outliers_i) == 0:
                print(f'No outliers found after {i} iterations')
                break
            outliers_i = np.convolve(outliers_i, np.ones(grow_n), mode='same') > 0
            # replace outliers with reference values (e.g. savgol)
            y_corr[outliers_i] = y_ref[outliers_i]
            outliers |= outliers_i
        
        return y_corr, outliers
        
    # @deprecated
    # def divide_telluric(self, file_label='SCIENCE', 
    #                     deep_tellurics=0.05,
    #                     grow_tellurics=60,
    #                     sigma=5.0, maxiters=5,
    #                     debug=False,
    #                     plot=False):
        
        
    #     self.file_label = file_label
    #     self.reshape_data()
        
    #     self.flux_corr, self.err_corr = (np.zeros_like(self.flux) for _ in range(2))
        
    #     if debug:
    #         fig, ax = plt.subplots(7,1, figsize=(14, 12))
    #         ax_flat = ax.flatten()
            
    #     for chip in range(self.n_chips):
    #         print(f'[divide_telluric] Chip {chip+1}')
    #         zeros = (self.flux[chip] <= 0) | (self.mtrans[chip] <= 0)
    #         edge = np.where(zeros==False)[0][0]

    #         # Fitted continuum (transmission function)
    #         cont = self.mcont[chip]
    #         cont[zeros] = np.nan
    #         cont /= np.nanmedian(self.mcont[chip])
            
    #         # if chip == 10:
    #         #     # np.argmin(abs(np.median(molecfit.wave,axis=1)-2.166)) --> 10
    #         #     # ignore Br-gamma line
    #         #     print(f'[divide_telluric] Ignoring Br-gamma line (2.166 um)')
    #         #     print(f'[divide_telluric] median wavelength {np.median(self.mwave[chip])} um')
    #         #     cont = np.ones_like(cont)

    #         # Observed data
    #         f = self.flux[chip]
    #         err = self.err[chip]
    #         f[zeros] = np.nan
            
    #         # f_median = np.nanmedian(f)
    #         # f /= f_median
    #         # err /= f_median
            
    #         # Fitted telluric model
    #         trans = self.mtrans[chip]# / np.nanmedian(self.mtrans[chip])
    #         deep_tellurics_mask = trans < deep_tellurics * np.nanmedian(self.mtrans[chip])
    #         if grow_tellurics > 0:
    #             deep_tellurics_mask  = np.convolve(deep_tellurics_mask , np.ones(grow_tellurics), mode='same') > 0

    #         # Smoothed corrected flux --> savgol(data / trans)
    #         f_corr_savgol = self.cflux_savgol[chip]
    #         f_corr_savgol[zeros] = np.nan
    #         # f_corr_savgol /= np.nanmedian(f_corr_savgol)
    #         f_corr_savgol /= cont
        
    #         # Apply telluric correction
    #         residuals = f / (trans * cont)
    #         self.err_corr[chip] = err / (trans * cont)

    #         # Refine telluric correction --> replace outliers
    #         residuals[deep_tellurics_mask ] = np.nan
    #         f_corr_savgol[deep_tellurics_mask ] = np.nan
    #         self.flux_corr[chip], outliers = self.find_outliers(residuals, f_corr_savgol, sigma=sigma, maxiters=maxiters)
    #         # set the uncertainty of the (replaced) outliers to the 90th percentile of the non-outliers
    #         self.err_corr[chip, outliers] = min(3. * np.mean(self.err_corr[chip, ~outliers]), np.max(self.err_corr[chip, ~outliers]))
    
    #         if debug:
    #             labels = ['Residuals', 'Savgol', 'Outliers'] if chip == 0 else ['','','']
    #             ax[chip//3].plot(self.mwave[chip], self.flux_corr[chip], color='navy',
    #                                label=labels[0])
    #             ax[chip//3].fill_between(self.mwave[chip], self.flux_corr[chip] - self.err_corr[chip],
    #                                          self.flux_corr[chip] + self.err_corr[chip], alpha=0.3, color='navy')
    #             ax[chip//3].plot(self.mwave[chip], cont * np.nanmedian(self.flux_corr[chip]), color='magenta', label='Continuum')
                
    #             ylim = list(ax[chip//3].get_ylim())
    #             ylim[0] = max(abs(ylim[0]), 0.0)
    #             ylim[1] = min(ylim[1], 2.0)
    #             ax[chip//3].plot(self.mwave[chip], residuals, label=labels[-1], color='r', zorder=0)
    #             ax[chip//3].plot(self.mwave[chip], f_corr_savgol, alpha=0.7,
    #                                label=labels[1], color='lime', lw=2.)
    #             ax[chip//3].axhline(1.0, ls='-', c='k', alpha=0.8)
    #             # ax[chip//3].set(ylim=ylim, xlim=(self.mwave[chip][0], self.mwave[chip][-1]))
                
    #     if debug:
    #         ax_flat[0].legend(ncol=4, loc=(0.00, 1.10), fontsize=18, frameon=False)
    #         wmin = np.min(self.mwave, axis=-1)[np.arange(0, len(self.mwave), 3)]
    #         wmax = np.max(self.mwave, axis=-1)[np.arange(2, len(self.mwave)+1, 3)]
    #         xpad = 0.0005
    #         assert np.shape(wmin) == np.shape(wmax), f'wmin {np.shape(wmin)} and wmax {np.shape(wmax)} must have same shape'
    #         for i, axi in enumerate(ax):
    #             axi.set_xlim(wmin[i] - xpad, wmax[i] + xpad)
    #         # [axi.set(xlabel='Wavelength [um]') for axi in ax[-1, :].flatten()]
    #         # [axi.set(ylabel='Flux') for axi in ax[:, 0].flatten()]
    #         # [axi.set(title='Detector {}'.format(i+1)) for i, axi in enumerate(ax[0, :].flatten())]
    #         fig.savefig(self.output_dir / f'{file_label}_divide_telluric.png', dpi=200, bbox_inches='tight')
    #         print(f' - Saved {self.output_dir}/{file_label}_divide_telluric.png')
    #         if plot:
    #             plt.show()
    #         else:
    #             plt.close()
    #     return self
            
        
    
    def save_corrected_data(self):
        
        assert hasattr(self, 'flux_corr'), 'Run `divide_telluric` first'
        # fits header
        header = self.load_science_header()
        empty_hdu = fits.PrimaryHDU(header=header)
        # reshape data to CRIRES format
        n_orders, n_detectors = (7,3)
        
        
        def reshape(attr):
            assert len(attr.shape) > 1, 'Data must be 2D'
            attr_rs = attr[::-1,:].reshape(n_orders, n_detectors, -1) # order number in decreasing wave
            attr_rs = attr_rs[:, ::-1, :] # detector number in increasing wave
            return attr_rs
        attrs = ['wave', 'flux', 'err', 'flux_corr', 'err_corr', 'mwave', 'mtrans', 'mcont']
        hdus= []
        for attr in attrs:
            
            hdus.append(fits.ImageHDU(data=reshape(getattr(self, attr)), name=attr))
        
        hdul = fits.HDUList([empty_hdu] + hdus)
        self.wfits(hdul, f'TELLURIC_CORRECTED_{self.file_name_stem}.fits', subdir='output')
        print(f'[save_corrected_data] Saved TELLURIC_CORRECTED_{self.file_name_stem}.fits')
        return None
    
    def save_flattened_data(self):
        
        
        assert hasattr(self, 'mwave'), 'Run `reshape_data` first'
        # sort by wavelength (take Molecfit fitted wavelength solution)
        wave_flat = self.mwave.flatten() * 1e3 # [nm]
        sort = np.argsort(wave_flat)
        
        header = 'Wlen(nm) Flux Flux_err Transm Cont Nans'
        # Nans contains the edges of each chip, excluded wavelength regions from fit
        ## and outliers for standard-star residuals
        nans = np.zeros_like(self.mwave)
        if hasattr(self, 'mask_exclude'):
            nans = np.logical_or(nans, self.mask_exclude)
        if hasattr(self, 'outliers'):
            nans = np.logical_or(nans, self.outliers)
            
        data = np.array([wave_flat[sort], 
                         self.flux.flatten()[sort] * self.continuum_constant,
                         self.err.flatten()[sort]  * self.continuum_constant,
                         self.mtrans.flatten()[sort],
                         self.mcont.flatten()[sort] * self.continuum_constant,
                         nans.flatten()[sort].astype(int)]).T
        
        out_file = self.output_dir / f'{self.file_name_stem}.dat'
        np.savetxt(out_file, data, header=header)
        print(f' - Saved {out_file}')
        
        # # Save telluric model
        # # transmission = self.mtrans * self.mcont
        # assert hasattr(self, 'br_gamma_fix'), 'Run `fix_br_gamma_continuum` first'
        # telluric = np.array([wave_flat[sort],
        #                      self.mtrans.flatten()[sort],
        #                      self.mcont.flatten()[sort] * self.continuum_constant]).T
        # np.savetxt(self.output_dir / f'{name}_molecfit_transm.dat', telluric, header='Wlen(nm) Transmission Continuum')
        # print(f' - Saved {self.output_dir}/{name}_molecfit_transm.dat')
        
    
    def load_corrected_data(self): # not useful...
        
        file = self.output_dir / f'TELLURIC_CORRECTED_{self.file_name_stem}.fits'
        with fits.open(file) as hdul:
            hdul.info()
            data = hdul[1].data
            header = hdul[0].header
            # print(header)
        return data
    
    def fix_br_gamma_continuum(self, debug=False):
        ''' Fix Br-gamma continuum by fitting a polynomial to the continuum '''
        # if not hasattr(self, 'wave'):
        #     self.reshape_data()
        
        assert hasattr(self, 'mcont'), 'Run `add_continuum` first'
            
        br_gamma_wave = 2.166 # um
        br_gamma_chip = np.argmin(abs(np.median(self.wave,axis=1)-br_gamma_wave))
        print(f'[fix_br_gamma_continuum] Chip {br_gamma_chip+1}')
        # self.mcont[br_gamma_chip] = np.ones_like(self.mcont[br_gamma_chip])
        # take median from same chips of different orders
        self.mcont[br_gamma_chip,] = self.mcont[br_gamma_chip-3,]
        self.mcont[br_gamma_chip,] += self.mcont[br_gamma_chip+3,]
        self.mcont[br_gamma_chip,] /= 2.0
        
        self.br_gamma_fix = True
        return self
            
    
    
if __name__ == '__main__':
        from excalibuhr.data import SPEC2D
        import excalibuhr.utils as su
        workpath = '/home/dario/phd/crires'
        night = '2023-03-03'
        target = 'TWA28' # irrelevant for molecfit
        molecfit = Molecfit(workpath, night, clean_start=False)
        # TODO: create subdirectory for each target? 
        
        
        cal_path = pathlib.Path(workpath) / night / 'out/obs_calibrated'

        filename = cal_path/'TWA28_PRIMARY_CRIRES_SPEC2D.fits'
        filename_std = cal_path / 'iSco_PRIMARY_CRIRES_SPEC2D.fits'

        s2d = SPEC2D(filename=str(filename))
        s2d_std = SPEC2D(filename=str(filename_std))
        s2d_std.remove_blackbody(teff=17000, debug=False) # B3V star
        
        # Important: run `prepare_data` for STANDARD after SCIENCE (for correct WAVE_INCLUDE/EXCLUDE files for `molecfit_model`)
        molecfit.prepare_data(s2d, Nedge=30, clip=2.5, plot=True, telluric_standard=False)
        molecfit.prepare_data(s2d_std, Nedge=30, clip=1.0, plot=True, telluric_standard=True)
        
        
        ## update config values ##
        molecfit.ftol = 1e-10
        molecfit.wlc_n = 3
        molecfit.continuum_n = 5
        
        # run molecfit
        fit_model = True
        if fit_model:
            molecfit.molecfit_model(verbose=True, expert_mode=False)
        
        # scale_pwv = False for telluric standard, True for science target
        for scale_pwv in [False, True]:
            molecfit.molecfit_calctrans(verbose=True, scale_pwv=scale_pwv)
            molecfit.plot_output(telluric_standard=np.logical_not(scale_pwv))
            
            # FIXME: issue with molecfit_correct recipe file...
            # molecfit.molecfit_correct(telluric_standard=np.logical_not(scale_pwv), verbose=True)