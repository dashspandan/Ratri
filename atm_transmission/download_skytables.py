#!/usr/bin/env python3
"""
download_skytables.py — Download ESO SkyCalc telluric skytables for ratri.

Queries the ESO SkyCalc CLI tool to generate skytable FITS files for a range
of precipitable water vapour (PWV) values at airmass 1.0. These files are
used by ratri for telluric transmission and sky emission modelling.

The airmass correction per exposure is applied internally by ratri via
Beer-Lambert law (tr = tr_airmass1 ^ airmass), so only airmass=1.0 files
are needed here.

Requirements:
    pip install skycalc-cli

Usage:
    cd atm_transmission/
    python download_skytables.py

Output files (saved in the current directory):
    skytable_a_1.0_p_{PWV}.fits   for each PWV value in mm

Reference:
    Noll et al. 2012, A&A 543, A92
    Jones et al. 2013, A&A 560, A91
    https://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC

Note on observatory:
    The default config uses Paranal (Cerro Armazones is essentially the same
    site and not separately available in SkyCalc). For sky emission the
    differences between sites are modest at NIR wavelengths. The telluric
    transmission (used for the main ratri simulation) is site-independent
    at fixed PWV and airmass.
"""

import json
import os

AIRMASS = 1.0

# PWV grid in mm — covers all ratri _PWV_TIERS including interpolation points
PWV_LIST = [
	0.05, 0.1, 0.25, 0.5,
	1.0, 1.5,
	2.5, 3.5,
	5.0, 7.5,
	10.0, 20.0,
]

DEFAULTS_FILE = 'skycalc_defaults.json'

print(f'Downloading {len(PWV_LIST)} skytable files at airmass={AIRMASS}...')
print(f'PWV values (mm): {PWV_LIST}\n')

for pwv in PWV_LIST:
	outfile = f'skytable_a_{AIRMASS}_p_{pwv}.fits'

	if os.path.exists(outfile):
		print(f'  Already exists, skipping: {outfile}')
		continue

	with open(DEFAULTS_FILE, 'r') as f:
		data = json.load(f)

	data['airmass']    = AIRMASS
	data['pwv']        = pwv
	data['wgrid_mode'] = 'fixed_spectral_resolution'
	data['wmin']       = 300.0
	data['wmax']       = 2600.0
	data['wres']       = 200000

	with open('data.json', 'w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent='\t')

	print(f'  Downloading: {outfile}')
	ret = os.system(f'skycalc_cli -i data.json -o {outfile}')
	os.remove('data.json')

	if ret != 0:
		print(f'  ERROR: skycalc_cli failed for PWV={pwv} mm')
	else:
		print(f'  Saved: {outfile}')

print('\nDone. All skytable files are ready for ratri.')
