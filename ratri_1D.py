#!/usr/bin/env python3
"""
ratri_1D.py — Quick-look SNR and detectability planning tool.

Produces diagnostic figures for each molecule found in a folder, plus
combined figures across all molecules:

  Figure 1        — SNR per pixel (one figure, instrument/star dependent)
  Figure 2..n+1   — Detectability per molecule (CCF, log-L, sigma vs RV)
  Figure n+2      — Combined sigma curves for all molecules (one damp value)
  Figure n+3      — Summary bar chart: peak detection sigma per molecule

Edit the parameters below and run:
    python ratri_1D.py

Planet model file format (txt/dat, one file per molecule):
    Column 0 : wavelength (unit auto-detected: cm/micron/nm/Angstrom)
    Column 1 : total transit depth (Rp/Rs)^2  [transmission]
               or total Fp/Fs ratio            [emission]
    Column 2 : depth/flux without target molecule
    Column 3 : depth/flux with target molecule only  (used for CCF)

The n+1 th case uses col 1 as the template (all molecules combined).
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from ratri import (
	ratri_preview_setup,
	ratri_preview_snr,
	ratri_preview_ccf,
	load_planet_model,
)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

PLANET_NAME = 'WASP-76 b'
STAR_NAME   = 'WASP-76'

# ---------------------------------------------------------------------------
# Observation mode
# ---------------------------------------------------------------------------

OB = 'tm'   # 'tm' (transit) or 'em' (emission)

# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

INST  = 'nlot_ccd_giano'
MODES = []
T_EXP = 300   # seconds

# ---------------------------------------------------------------------------
# Molecule folder
# All .dat files in this folder will be processed automatically.
# ---------------------------------------------------------------------------

MOLECULE_FOLDER = './planet_models/WASP-76b' ##as an example
FILE_PATTERN    = '*.dat'

# ---------------------------------------------------------------------------
# Atmospheric conditions
# ---------------------------------------------------------------------------

QUAL    = None   # None = site default
AIRMASS = 1.2

# ---------------------------------------------------------------------------
# RV grid
# ---------------------------------------------------------------------------

RV_CENTRE = -20.0
RV_LEFT   = 60.0
RV_RIGHT  = 60.0
RV_BUFFER = 20.0   # km/s wing region for noise floor estimation
RV_INJECT = -20.0    # km/s — RV at which planet signal is injected
                   # set to vsys + Kp*sin(2*pi*phase) for a realistic test

# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------

DAMP_VALUES      = [1, 3, 5, 10]   # None = auto; or e.g. [1, 2, 5, 10, 20]
N_DAMP_AUTO      = 6
SUMMARY_DAMP_IDX = -1    # index into damp_values; -1 = last (highest)

# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

SN_MASK_THRESHOLD = 30

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

ATM_DIR           = './atm_transmission'
SPECTRA_CACHE_DIR = './spectra_cache'
STELLAR_LIBRARY   = None
SAVEDIR           = './ratri_preview_output'

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
	os.makedirs(SAVEDIR, exist_ok=True)

	# Setup — done once for all molecules
	ctx = ratri_preview_setup(
		planet_name       = PLANET_NAME,
		inst              = INST,
		modes             = MODES,
		ob                = OB,
		t_exp             = T_EXP,
		qual              = QUAL,
		airmass           = AIRMASS,
		sn_mask_threshold = SN_MASK_THRESHOLD,
		atm_dir           = ATM_DIR,
		spectra_cache_dir = SPECTRA_CACHE_DIR,
		star_name         = STAR_NAME,
		library           = STELLAR_LIBRARY,
		savedir           = SAVEDIR,
	)

	# SNR figure — once
	ratri_preview_snr(ctx)

	# Discover molecule files
	mol_files = sorted(glob.glob(os.path.join(MOLECULE_FOLDER, FILE_PATTERN)))
	if not mol_files:
		raise FileNotFoundError(
			f'No files matching {FILE_PATTERN} found in {MOLECULE_FOLDER}'
		)
	print(f'\nFound {len(mol_files)} molecule file(s):')
	for f in mol_files:
		print(f'  {os.path.basename(f)}')

	# Load total spectrum from first file (tr_d col 1, same in all files)
	first_model  = load_planet_model(mol_files[0])
	if 'planet_wav' not in first_model:
		raise ValueError(
			'Cannot auto-extract wavelength. Ensure files are .txt or .dat '
			'with wavelength in column 0.'
		)
	wav_p      = first_model['planet_wav'] * 1e4   # micron -> Angstrom
	tr_d_total = first_model['data'][:, 1]          # col 1 = total

	# Per-molecule CCF loop
	all_results   = []
	mol_labels    = []
	damp_vals_ref = None

	for mol_path in mol_files:
		mol_label = os.path.splitext(os.path.basename(mol_path))[0]
		# Try to extract short molecule name from filename
		parts     = mol_label.split('_')
		mol_short = parts[-1] if len(parts) > 1 else mol_label

		model    = load_planet_model(mol_path)
		data_arr = model['data']
		if data_arr.shape[1] >= 4:
			tr_d_only = data_arr[:, 3]
		else:
			print(f'  Warning: {mol_label} has <4 columns — using col 1 as template')
			tr_d_only = data_arr[:, 1]

		result = ratri_preview_ccf(
			ctx,
			wav_p       = wav_p,
			tr_d        = tr_d_total,
			tr_d_only   = tr_d_only,
			mol_label   = mol_short,
			rv_centre   = RV_CENTRE,
			rv_left     = RV_LEFT,
			rv_right    = RV_RIGHT,
			rv_inject   = RV_INJECT,
			damp_values = DAMP_VALUES,
			n_damp_auto = N_DAMP_AUTO,
			rv_buffer   = RV_BUFFER,
		)

		all_results.append(result)
		mol_labels.append(mol_short)
		if damp_vals_ref is None:
			damp_vals_ref = result['damp_values']

	# All-molecules case — use tr_d as both injection and template
	print('\n--- All molecules (total spectrum as template) ---')
	result_all = ratri_preview_ccf(
		ctx,
		wav_p       = wav_p,
		tr_d        = tr_d_total,
		tr_d_only   = tr_d_total,
		mol_label   = 'All',
		rv_centre   = RV_CENTRE,
		rv_left     = RV_LEFT,
		rv_right    = RV_RIGHT,
		rv_inject   = RV_INJECT,
		damp_values = damp_vals_ref,
		rv_buffer   = RV_BUFFER,
	)
	all_results.append(result_all)
	mol_labels.append('All')

	# Combined sigma figure — all molecules at summary damp index
	rv_grid   = all_results[0]['rv_grid']
	n_mols    = len(all_results)
	di        = SUMMARY_DAMP_IDX
	damp_used = damp_vals_ref[di]
	colors    = plt.cm.tab10(np.linspace(0, 1, n_mols))
	safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', PLANET_NAME).strip('_')

	fig_comb, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
	for mi, (res, label) in enumerate(zip(all_results, mol_labels)):
		lw = 1.8 if label == 'All' else 1.2
		axes[0].plot(rv_grid, res['ccf'][di],   color=colors[mi], label=label, lw=lw)
		axes[1].plot(rv_grid, res['logl'][di],  color=colors[mi], lw=lw)
		axes[2].plot(rv_grid, res['sigma'][di], color=colors[mi], lw=lw)

	for ax in axes:
		ax.axvline(RV_CENTRE, color='k', ls=':', lw=0.8, alpha=0.5)
	axes[2].axhline(3, color='red',     ls='--', lw=0.8, alpha=0.7, label=r'3$\sigma$')
	axes[2].axhline(5, color='darkred', ls='--', lw=0.8, alpha=0.7, label=r'5$\sigma$')
	axes[0].set_ylabel('CCF')
	axes[1].set_ylabel('log L')
	axes[2].set_ylabel(r'$\sigma$ of exclusion')
	axes[2].set_xlabel(r'RV shift (km s$^{-1}$)')
	axes[0].legend(fontsize=8, ncol=min(n_mols, 5))
	axes[2].legend(fontsize=8)
	fig_comb.suptitle(
		f'{PLANET_NAME} — All molecules — {INST.upper()} — {OB.upper()} — '
		f't_exp={T_EXP}s — damp={damp_used} (N={damp_used**2})',
		fontsize=10,
	)
	fig_comb.tight_layout()
	comb_path = os.path.join(
		SAVEDIR,
		f'{safe_name}_{INST}_{OB}_{T_EXP}s_combined_damp{damp_used}.png'
	)
	fig_comb.savefig(comb_path, dpi=200, bbox_inches='tight')
	print(f'\nCombined figure saved to {comb_path}')
	plt.show()
	plt.close(fig_comb)

	# Summary bar chart — detection sigma per molecule at summary damp
	det_sigmas = [res['det_sigma'][di] for res in all_results]

	fig_sum, ax_sum = plt.subplots(figsize=(max(6, n_mols * 0.9 + 2), 5))
	x    = np.arange(n_mols)
	bars = ax_sum.bar(x, det_sigmas, color=colors, edgecolor='k', linewidth=0.5)
	ax_sum.axhline(3, color='red',     ls='--', lw=1.0, alpha=0.8, label=r'3$\sigma$')
	ax_sum.axhline(5, color='darkred', ls='--', lw=1.0, alpha=0.8, label=r'5$\sigma$')
	ax_sum.set_xticks(x)
	ax_sum.set_xticklabels(mol_labels, rotation=30, ha='right', fontsize=10)
	ax_sum.set_ylabel(r'Detection significance ($\sigma$)')
	ax_sum.set_title(
		f'{PLANET_NAME} — {INST.upper()} — {OB.upper()} — '
		f't_exp={T_EXP}s — damp={damp_used} (N={damp_used**2})',
		fontsize=10,
	)
	ax_sum.legend(fontsize=9)
	ax_sum.set_ylim(bottom=0)
	for bar, val in zip(bars, det_sigmas):
		ax_sum.text(
			bar.get_x() + bar.get_width() / 2,
			bar.get_height() + 0.05,
			f'{val:.1f}',
			ha='center', va='bottom', fontsize=8,
		)
	fig_sum.tight_layout()
	sum_path = os.path.join(
		SAVEDIR,
		f'{safe_name}_{INST}_{OB}_{T_EXP}s_summary_damp{damp_used}.png'
	)
	fig_sum.savefig(sum_path, dpi=200, bbox_inches='tight')
	print(f'Summary figure saved to {sum_path}')
	plt.show()
	plt.close(fig_sum)

	print('\nAll done.')
	print(f'All figures saved to: {SAVEDIR}')
