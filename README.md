# ratri

**ratri** is a Python-based observation simulator and planning tool for High-Resolution Cross-Correlation Spectroscopy (HRCCS) of exoplanet atmospheres. It simulates time-series stellar spectra including telluric absorption, sky emission, instrument throughput, and optional planet model injection, for a range of ground-based high-resolution spectrographs.

Supported instruments: CRIRES+ (up to K band), CARMENES, GIANO, SPIRou, ANDES on ELT (on a single 100,000 grid for different modes), ANDES (with CARMENES CCD configuration for YJH band), TMT (with GIANO CCD configuration), NLOT (with GIANO CCD configuration).

---

## Requirements

- Python 3.9 or later
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) (recommended)

---

## Installation

### 1. Install Miniconda (if you don't have conda)

Download and install Miniconda for your platform from:
https://docs.conda.io/en/latest/miniconda.html

Follow the installer instructions. Once installed, open a new terminal and verify:

```bash
conda --version
```

### 2. Create a conda environment

```bash
conda create -n ratri python=3.10
conda activate ratri
```

You will need to activate this environment every time you use ratri:

```bash
conda activate ratri
```

### 3. Clone the repository

```bash
git clone https://github.com/dashspandan/Ratri.git
cd Ratri
```

### 4. Install dependencies
(Simplest)
 
```bash
pip install -r requirements.txt
```

Or install the package directly (editable mode recommended for development):

```bash
pip install -e .
```

### 5. Verify the installation

```bash
python -c "import ratri; print('ratri installed successfully')"
```

---

## Required data files

ratri requires the following external data files which are not included in the repository due to their size. Place them in the directories shown.

### Atmospheric transmission skytables (ESO SkyCalc)

Reference: Noll et al. 2012, A&A 543, A92; Jones et al. 2013, A&A 560, A91

Skytable files are downloaded automatically using the provided script, which
queries the ESO SkyCalc CLI tool. Run once after installation:

```bash
cd atm_transmission/
python download_skytables.py
```

This downloads skytable FITS files for all PWV values used by ratri's quality
tiers and saves them in `./atm_transmission/` with the naming convention:
```
skytable_a_1.0_p_{PWV}.fits
```

Internet access is required. The `skycalc_cli` package is broken on Python 3.11+, which is why we have mandated working in a conda environment with python 3.10. Already-downloaded files are skipped automatically, so re-running is safe.


### Instrument grid files

Download the files from [here](https://drive.google.com/file/d/1nvXSorHwQLGJOmeOqToXXGcI4yFy-eYc/view?usp=drive_link). If they are not available for some reason, please contact me. The crires+ modes files available in this folder are the only ones available to simulate using ratri for now. Other modes will be added in the future.

#### CRIRES+
Place instrument grid `.npy` files in `./instrument_grids_crires/`:
```
crires+{mode}mode.npy    # e.g. crires+y1028mode.npy
```

#### CARMENES, GIANO, SPIRou
Place wavelength and throughput `.npy` files in `./instrument_grids_spirougianocarmenes/`:
```
wlen_template_carmenes.npy    eta_carmenes.npy
wlen_template_giano.npy       eta_giano.npy
wlen_template_spirou.npy      eta_spirou.npy
```

#### ANDES
Place grid files in `./instrument_grids_andes/`:
```
parameters_for_ratri_band_{MODE}_ANDES_new.npy
parameters_for_ratri_band_{MODE}_ANDES_carmenesinterpol_new.npy
```
where `{MODE}` is one of `YJ`, `H`, `K`, `UBV`, `RIZ` (full ANDES) or `YJH` (CARMENES CCD).

### Stellar spectra (PHOENIX/ACES) (Others like ATLAS9 will be added in future)

Reference: Husser et al. 2013, A&A 553, A6

Stellar spectra are downloaded automatically on first use and cached in `./spectra_cache/`. No manual setup required - internet access is needed on first run only.

### Planet model files

Planet model spectra should be tab-separated text files (`.dat` or `.txt`) with the following column format (e.g. for tranmission spectroscopy):

```
# Wavelength (micron)    tdepth (Rp/Rs)^2    tdepth_no_mol    tdepth_onlymol
  0.30000   0.01177   0.01177   0.01177
  ...
```

Wavelength units are auto-detected (cm, micron, nm, Angstrom). For multi-molecule runs, provide one file per molecule in a dedicated folder, e.g. WASP-77ab_CO.dat.

---

## Quick start

### Step 0 - Quick-look planning (optional but recommended before Step 1, and only if you already have a model exoplanet spectrum in the format specified)

For a fast SNR and detectability estimate before committing to a full simulation, edit `ratri_1D.py` to set your target, instrument, molecule folder, and RV parameters, then run:

```bash
conda activate ratri
python ratri_1D.py
```

This produces:
- An SNR per pixel plot across all instrument orders
- A per-molecule detectability figure (CCF + detection significance vs RV)
- A combined figure overlaying all molecules
- A summary bar chart of detection significance per molecule

All figures are saved to `./ratri_preview_output/`.


### Step 1 - Generate a runner script

```bash
conda activate ratri
python generate_runner.py
```

This launches an interactive command-line wizard that:
- Queries the NASA Exoplanet Archive for orbital and stellar parameters
- Lets you select literature references per parameter
- Prompts for instrument, observing mode, exposure time, and date range
- Generates a ready-to-run simulation script named `{planet}_{ob}.py`

### Step 2 - Plan your observation

```bash
python {planet}_{ob}.py --plan
```

Identifies the best observable nights within your date range and saves them to a JSON file in the spectra cache directory.

### Step 3 - Run the simulation

```bash
python {planet}_{ob}.py --simulate
```

Runs the full HRCCS observation simulation and saves flux cubes, SNR cubes, and diagnostic plots to the output directory.

Running without arguments does both steps sequentially:

```bash
python {planet}_{ob}.py
```

---

## File overview

| File | Description |
|------|-------------|
| `ratri.py` | Main simulation engine and preview functions |
| `stellar.py` | Stellar parameter lookup and spectrum downloader |
| `generate_runner.py` | Interactive wizard to generate observation runner scripts |
| `ratri_1D.py` | Quick-look SNR and detectability planning tool |
| `requirements.txt` | Python dependencies |
| `setup.py` | Package installation script |

---

## PWV quality tiers

PWV tiers assume a lognormal distribution centered around the median PWV values. The tiers are as follows:

| Tier | Median PWV |
|------|------------|
| `excellent` | 0.3 mm |
| `very_good` | 1.25 mm |
| `good` | 3.0 mm |
| `avg` | 6.25 mm |
| `bad` | 15.0 mm |

As you run the generator script, you'll see that each site already has a default setting based on references in literature. Input a different tier if you want a better or worse night.

---

## Directory structure

```
ratri/
├── ratri.py
├── stellar.py
├── generate_runner.py
├── ratri_1D.py
├── requirements.txt
├── setup.py
├── README.md
├── atm_transmission/          # ESO SkyCalc skytable FITS files (user-provided)
│   └── skytable_a_1.0_p_*.fits
├── instrumentgrids_crires/    # CRIRES+ instrument grids (user-provided)
│   └── crires+*mode.npy
├── instrument_grids_spirougianocarmenes/   # CARMENES/GIANO/SPIRou grids (user-provided)
│   └── wlen_template_*.npy, eta_*.npy
├── instrument_grids_andes/    # ANDES grids (user-provided)
│   └── parameters_for_ratri_band_*.npy
├── spectra_cache/             # Auto-created on first run
│   ├── stellar_params_cache.json
│   └── *.fits (downloaded stellar spectra)
└── planet_models/             # User-provided planet model files
    └── {target}/
        └── {target}_{molecule}.dat
```

---

## Key references

- Husser et al. 2013, A&A 553, A6 - PHOENIX/ACES stellar spectra
- Noll et al. 2012, A&A 543, A92 - ESO SkyCalc sky model
- Jones et al. 2013, A&A 560, A91 - ESO SkyCalc sky model
- Morris et al. 2018, AJ 155, 128 - astroplan observation planning
- Winn 2010, arXiv:1001.2010 - transit/occultation geometry and duration formulae
- Kreidberg 2015, PASP 127, 1161 - batman transit model (for implementation later)
- Creevey et al. 2023, A&A 674, A26 - Gaia DR3 stellar parameters (optional querying if astroquery and TAP fail)

---

## Known limitations and planned features

The following features are planned for future releases:

- Rossiter-McLaughlin (RM) effect integration
- Phase-dependent (GCM) planet spectrum injection
- batman transit light curve weighting for transmission mode
- Your custom requirement (please feel free to reach out!)

---

## How to reference

Please cite:

 - Dash, Spandan, et al. "Detectability of oxygen fugacity regimes in the magma ocean world 55 Cancri e at high spectral resolution." Monthly Notices of the Royal Astronomical Society 538.4 (2025): 3042-3066.
 - Dash, Spandan, Dwaipayan Dubey, and Liton Majumdar. "Probing the Atmospheres of Young Long-Period Sub-Neptune Progenitors with ELT/ANDES." arXiv preprint arXiv:2602.22830 (2026). (In review)
 - (Optional) Dash, Spandan. Ground-based high-resolution cross-correlation spectroscopy of sub-Jovian exoplanet atmospheres. Diss. University of Warwick, 2025.

## License

Please see the `MIT LICENSE` tab [here](https://github.com/dashspandan/Ratri?tab=MIT-1-ov-file) for details.
