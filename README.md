# ratri

**ratri** is a Python-based observation simulator and planning tool for High-Resolution Cross-Correlation Spectroscopy (HRCCS) of exoplanet atmospheres. It simulates time-series stellar spectra including telluric absorption, sky emission, instrument throughput, and optional planet model injection, for a range of ground-based high-resolution spectrographs.

Supported instruments: CRIRES+ (up to K band), CARMENES, GIANO, SPIRou, ANDES, ANDES (CARMENES CCD), TMT+GIANO CCD, NLOT+GIANO CCD.

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
git clone https://github.com/dashspandan/Ratri
cd Ratri
```

### 4. Install dependencies

Simplest

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

ratri requires the following external data files which are not included in the repository due to their size. Place them in the directories shown if they are not already there.

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

Internet access and the `requests` package (installed via `requirements.txt`)
are required. The script calls the ESO SkyCalc REST API directly, avoiding
the `skycalc_cli` package which is broken on Python 3.11+. Already-downloaded files are skipped automatically, so re-running
is safe. The download covers:
```
0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0
```

### Instrument grid files

Download the files from [here](https://drive.google.com/file/d/1nvXSorHwQLGJOmeOqToXXGcI4yFy-eYc/view?usp=drive_link). If they are not available for some reason, please contact me. The crires+ files available in this folder are the only ones available to simulate using ratri for now. Other modes will be added in the future.

#### CRIRES+
Place instrument grid `.npy` files in `./instrumentgrids_crires/`:
```
crires+{mode}mode.npy    # e.g. crires+l3340mode.npy
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

### Stellar spectra (PHOENIX/ACES)

Reference: Husser et al. 2013, A&A 553, A6

Stellar spectra are downloaded automatically on first use and cached in `./spectra_cache/`. No manual setup required — internet access is needed on first run only.

### Planet model files

Planet model spectra should be tab-separated text files (`.dat` or `.txt`) with the following column format:

```
# Wavelength (micron)    tdepth (Rp/Rs)^2    tdepth_no_mol    tdepth_onlymol
  0.30000   0.01177   0.01177   0.01177
  ...
```

Wavelength units are auto-detected (cm, micron, nm, Angstrom). For multi-molecule runs, provide one file per molecule in a dedicated folder.

---

## Quick start

### Step 1 — Generate a runner script

```bash
conda activate ratri
python generate_runner.py
```

This launches an interactive wizard that:
- Queries the NASA Exoplanet Archive for orbital and stellar parameters
- Lets you select literature references per parameter
- Prompts for instrument, observing mode, exposure time, and date range
- Generates a ready-to-run simulation script named `{planet}_{ob}.py`

### Step 2 — Plan your observation

```bash
python {planet}_{ob}.py --plan
```

Identifies the best observable nights within your date range and saves them to a JSON file in the spectra cache directory.

### Step 3 — Run the simulation

```bash
python {planet}_{ob}.py --simulate
```

Runs the full HRCCS observation simulation and saves flux cubes, SNR cubes, and diagnostic plots to the output directory. We have kept an example WASP-76b_tm.py file for you to examine and play with.

Running without arguments does both steps sequentially:

```bash
python {planet}_{ob}.py
```

### Step 4 — Quick-look planning (optional, and only if you already have a model exoplanet spectrum in the format specified)

For a fast SNR and detectability estimate before committing to a full simulation, edit `ratri_1D.py` to set your target, instrument, molecule folder, and RV parameters, then run:

```bash
python ratri_1D.py
```

This produces:
- An SNR per pixel plot across all instrument orders
- A per-molecule detectability figure (CCF + detection significance vs RV)
- A combined figure overlaying all molecules
- A summary bar chart of detection significance per molecule

All figures are saved to `./ratri_preview_output/`.

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

## Instruments and modes

| Key | Telescope | Wavelength | Resolution |
|-----|-----------|------------|------------|
| `crires+` | VLT (8.2 m) | 1–5 μm (per mode) | 100,000 |
| `carmenes` | Calar Alto (3.5 m) | 0.52–1.71 μm | 80,400 |
| `giano` | TNG (3.58 m) | 0.95–2.45 μm | 50,000 |
| `spirou` | CFHT (3.6 m) | 0.95–2.35 μm | 70,000 |
| `andes` | ELT (38.5 m) | multiple bands | 100,000 |
| `andes_ccd_carmenes` | ELT (38.5 m) | YJH | 100,000 |
| `tmt_ccd_giano` | TMT (30 m) | 0.95–2.45 μm | 100,000 |
| `nlot_ccd_giano` | NLOT (13.7 m) | 0.95–2.45 μm | 60,000 (TBC) |

ANDES modes (pass as a list): `YJ`, `H`, `K`, `UBV`, `RIZ`.
ANDES CARMENES CCD mode: `YJH` only.

---

## PWV quality tiers

| Tier | Median PWV | Typical sites |
|------|------------|---------------|
| `excellent` | 0.3 mm | Cerro Armazones (best conditions) |
| `very_good` | 1.25 mm | Mauna Kea, Hanle (NLOT) |
| `good` | 3.0 mm | Cerro Paranal, La Palma |
| `avg` | 6.25 mm | Calar Alto |
| `bad` | 15.0 mm | Poor conditions |

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

- Husser et al. 2013, A&A 553, A6 — PHOENIX/ACES stellar spectra
- Noll et al. 2012, A&A 543, A92 — ESO SkyCalc sky model
- Jones et al. 2013, A&A 560, A91 — ESO SkyCalc sky model
- Winn 2010, arXiv:1001.2010 — transit/occultation geometry and duration formulae
- Kreidberg 2015, PASP 127, 1161 — batman transit model
- Creevey et al. 2023, A&A 674, A26 — Gaia DR3 stellar parameters

---

## Known limitations and planned features

The following features are planned for future releases:

- Rossiter-McLaughlin (RM) effect integration
- Phase-dependent (GCM) planet spectrum injection
- batman transit light curve weighting for transmission mode
- NLOT instrument parameters are preliminary pending first-light instrument selection
- SPIRou throughput grid beyond 2.35 μm (pending)

---

## License

MIT License. See `LICENSE` for details.
