#!/usr/bin/env python3
"""
generate_runner_v2.py — Interactive wizard to generate a ratri runner script.

Version 2: reference-aware queries. Retrieves all available references from
the NASA Exoplanet Archive ps/stellarhosts tables, shows parameter coverage
per reference, and lets the user select which reference to use for orbital
and stellar parameters. Supplementary references are tracked per parameter.

Usage:
    python generate_runner_v2.py
"""

import os
import re
import sys
import json
import math
import time
import warnings
import urllib.request
import urllib.parse
import numpy as np


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _ask(prompt, default=None, cast=str, validate=None):
	"""Prompt the user for input with an optional default and type cast."""
	if default is not None:
		full_prompt = f'  {prompt} [{default}]: '
	else:
		full_prompt = f'  {prompt}: '
	while True:
		raw = input(full_prompt).strip()
		if raw == '' and default is not None:
			return default
		if raw == '' and default is None:
			print('    This field is required.')
			continue
		try:
			val = cast(raw)
		except (ValueError, TypeError):
			print(f'    Invalid input — expected {cast.__name__}.')
			continue
		if validate is not None:
			msg = validate(val)
			if msg:
				print(f'    {msg}')
				continue
		return val


def _ask_choice(prompt, choices, default=None):
	"""Prompt the user to choose from a list of options."""
	choices_str = '/'.join(choices)
	if default:
		full_prompt = f'  {prompt} [{choices_str}, default={default}]: '
	else:
		full_prompt = f'  {prompt} [{choices_str}]: '
	while True:
		raw = input(full_prompt).strip().lower()
		if raw == '' and default:
			return default
		if raw in choices:
			return raw
		print(f'    Please choose one of: {choices_str}')


def _ask_yn(prompt, default='y'):
	return _ask_choice(prompt, ['y', 'n'], default=default) == 'y'


def _safe_planet_name(name):
	"""Convert planet name to a safe filename stem."""
	return re.sub(r'[^a-zA-Z0-9_]', '_', name).strip('_')


# ---------------------------------------------------------------------------
# Parameter persistence
# ---------------------------------------------------------------------------

def _params_file(planet_name, ob, spectra_cache):
	os.makedirs(spectra_cache, exist_ok=True)
	return os.path.join(
		spectra_cache,
		f'{_safe_planet_name(planet_name)}_{ob}_params.json'
	)


def _save_params(params, planet_name, ob, spectra_cache):
	path = _params_file(planet_name, ob, spectra_cache)
	with open(path, 'w') as f:
		json.dump(params, f, indent=2)
	print(f'  Parameters saved to {path}')


def _load_params(planet_name, ob, spectra_cache):
	path = _params_file(planet_name, ob, spectra_cache)
	if os.path.exists(path):
		with open(path) as f:
			return json.load(f)
	return None


# ---------------------------------------------------------------------------
# Archive query helpers
# ---------------------------------------------------------------------------

def _safe_float(v):
	"""
	Extract a float from an archive value, handling astropy MaskedQuantity
	(which carries physical units) and masked/NaN values.
	Returns None if the value is missing, masked, NaN, or infinite.
	"""
	try:
		if v is None:
			return None
		# MaskedQuantity carries units — extract numeric value via .value
		if hasattr(v, 'value'):
			v = float(v.value)
		else:
			v = float(v)
		import math
		return None if (math.isnan(v) or math.isinf(v)) else v
	except Exception:
		return None


def _tap_post(adql, timeout=30):
	"""
	Send an ADQL query to the NASA Exoplanet Archive TAP sync endpoint
	via HTTP POST and return the parsed JSON response as a list of dicts.

	Uses POST (not GET) as required by the IVOA TAP standard.
	Ref: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html

	Returns list of row dicts, or None on failure.
	"""
	url  = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync'
	body = urllib.parse.urlencode({'query': adql, 'format': 'json'}).encode()
	req  = urllib.request.Request(
		url, data=body, headers={'User-Agent': 'Mozilla/5.0'}
	)
	with urllib.request.urlopen(req, timeout=timeout) as r:
		return json.loads(r.read().decode())


def _tap_post_with_retries(adql, n_retries=3, base_timeout=20):
	"""
	_tap_post with retry loop and increasing timeout.
	Returns list of row dicts, or None if all attempts fail.
	"""
	for attempt in range(1, n_retries + 1):
		try:
			timeout = base_timeout * attempt
			print(f'  urllib TAP attempt {attempt}/{n_retries} (timeout={timeout}s)...',
			      end=' ', flush=True)
			result = _tap_post(adql, timeout=timeout)
			print('ok.')
			return result
		except Exception as e:
			print(f'failed: {e}')
			if attempt < n_retries:
				time.sleep(2 * attempt)
	return None


def _astroquery(table, where, select_cols):
	"""
	Query NASA Exoplanet Archive via astroquery.
	Returns astropy Table or None.
	Ref: https://astroquery.readthedocs.io/en/latest/ipac/nexsci/nasa_exoplanet_archive.html
	"""
	try:
		from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
	except ImportError:
		try:
			from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
		except ImportError:
			return None
	return NasaExoplanetArchive.query_criteria(
		table  = table,
		where  = where,
		select = select_cols,
	)


# ---------------------------------------------------------------------------
# Orbital parameter columns and display names
# ---------------------------------------------------------------------------

# Columns queried from ps table for orbital parameters
ORBITAL_COLS = [
	'pl_refname',
	'pl_orbper',
	'pl_tranmid',
	'pl_trandur',
	'pl_orbeccen',
	'pl_orblper',
	'pl_orbincl',
	'pl_orbsmax',
	'pl_radj',
	'st_radv',
]

# Key columns that must be present for a reference to be shown
ORBITAL_KEY_COLS = ['pl_orbper', 'pl_tranmid', 'pl_trandur', 'pl_orbincl']

ORBITAL_DISPLAY = {
	'pl_orbper'   : ('period',      'Period (days)'),
	'pl_tranmid'  : ('tranmid_bjd', 'T0 (BJD_TDB)'),
	'pl_trandur'  : ('trandur_h',   'Duration (h)'),
	'pl_orbeccen' : ('ecc',         'Eccentricity'),
	'pl_orblper'  : ('omega_deg',   'Omega (deg)'),
	'pl_orbincl'  : ('incl_deg',    'Inclination (deg)'),
	'pl_orbsmax'  : ('a_au',        'a (AU)'),
	'pl_radj'     : ('rp_rj',       'Rp (Rjup)'),
	'st_radv'     : ('vsys',        'vsys (km/s)'),
}

# Stellar parameter columns queried from stellarhosts
STELLAR_COLS = [
	'st_refname',
	'st_teff',
	'st_logg',
	'st_met',
	'st_rad',
	'st_mass',
	'sy_dist',
]

STELLAR_KEY_COLS = ['st_teff', 'st_rad']

STELLAR_DISPLAY = {
	'st_teff' : ('teff',     'T_eff (K)'),
	'st_logg' : ('logg',     'log g (cgs)'),
	'st_met'  : ('feh',      '[Fe/H] (dex)'),
	'st_rad'  : ('radius',   'R_star (R_sun)'),
	'st_mass' : ('mass',     'M_star (M_sun)'),
	'sy_dist' : ('distance', 'Distance (pc)'),
}


# ---------------------------------------------------------------------------
# Reference query functions
# ---------------------------------------------------------------------------

def _extract_row(row, col_map, is_astropy=False):
	"""
	Extract parameter values from a row (astropy Table row or dict).
	col_map: dict of archive_col -> (param_key, display_name)
	Returns dict of param_key -> value (float or None).
	"""
	result = {}
	for col, (key, _) in col_map.items():
		if col == 'pl_refname' or col == 'st_refname':
			continue
		try:
			v = row[col]
			result[key] = _safe_float(v)
		except (KeyError, IndexError):
			result[key] = None

	# Unit-aware conversions
	# pl_trandur comes in hours from astroquery (MaskedQuantity with unit h)
	# — _safe_float strips units via .value, giving the numeric value in hours
	# No * 24 needed.

	return result


def _get_refname(row, is_astropy=False):
	"""
	Extract reference name string from a row, stripping HTML anchor tags.
	The archive returns pl_refname/st_refname as HTML like:
	  <a refstr=... href=...>Addison et al. 2019</a>
	We extract just the inner text.
	"""
	try:
		v = row['pl_refname']
	except (KeyError, TypeError):
		try:
			v = row['st_refname']
		except (KeyError, TypeError):
			return 'Unknown'
	if hasattr(v, 'unmasked'):
		v = str(v.unmasked)
	v = str(v).strip()
	# Strip HTML anchor tags — extract text between > and <
	match = re.search(r'>([^<]+)<', v)
	if match:
		return match.group(1).strip()
	# Fallback: strip any remaining HTML tags
	v = re.sub(r'<[^>]+>', '', v).strip()
	return v if v else 'Unknown' 


def _query_references(planet_name, host_name, param_type):
	"""
	Query all available references for a planet from the NASA archive.

	param_type: 'orbital' or 'stellar'

	Returns list of dicts, each with:
	  'refname'  : str
	  'values'   : dict of param_key -> float|None
	  'coverage' : int (number of non-None key params)

	Only references with all key columns non-null are included.
	"""
	if param_type == 'orbital':
		cols      = ORBITAL_COLS
		key_cols  = ORBITAL_KEY_COLS
		col_map   = ORBITAL_DISPLAY
		table     = 'ps'
		where_col = 'pl_name'
		where_val = planet_name
		ref_col   = 'pl_refname'
	else:
		cols      = STELLAR_COLS
		key_cols  = STELLAR_KEY_COLS
		col_map   = STELLAR_DISPLAY
		table     = 'stellarhosts'
		where_col = 'hostname'
		where_val = host_name
		ref_col   = 'st_refname'

	select_str = ','.join(cols)
	# Require all key columns to be non-null
	key_conditions = ' AND '.join(f'{c} IS NOT NULL' for c in key_cols)
	where_str = f"{where_col}='{where_val}' AND {key_conditions}"

	rows_raw = None

	# Try astroquery first
	# Suppress harmless 'Unrecognized unit' warnings from astroquery
	warnings.filterwarnings('ignore', message='Unrecognized unit')
	try:
		print(f'  Trying astroquery ({table})...', end=' ', flush=True)
		tbl = _astroquery(table, where_str, select_str)
		if tbl is not None and len(tbl) > 0:
			print(f'ok. ({len(tbl)} references found)')
			rows_raw = [dict(zip(cols, [tbl[c][i] for c in cols]))
			            for i in range(len(tbl))]
			is_astropy = True
		else:
			print('no results.')
	except Exception as e:
		print(f'failed: {e}')

	# urllib POST fallback
	if rows_raw is None:
		adql = f"SELECT {select_str} FROM {table} WHERE {where_str}"
		data = _tap_post_with_retries(adql)
		if data:
			rows_raw = data
			is_astropy = False

	if not rows_raw:
		return []

	# Parse rows
	refs = []
	seen = set()
	for row in rows_raw:
		refname = _get_refname(row)
		if refname in seen:
			continue
		seen.add(refname)
		values   = _extract_row(row, col_map, is_astropy)
		key_param_keys = [col_map[c][0] for c in key_cols if c in col_map]
		coverage = sum(1 for pk in key_param_keys if values.get(pk) is not None)
		refs.append({
			'refname'  : refname,
			'values'   : values,
			'coverage' : coverage,
		})

	# Sort by coverage descending
	refs.sort(key=lambda r: r['coverage'], reverse=True)
	return refs


def _show_reference_table(refs, col_map, param_type):
	"""
	Print a table of available references with parameter coverage,
	using dynamic column widths based on content.
	"""
	show_cols = [(col, key, disp) for col, (key, disp) in col_map.items()]

	# Compute column widths dynamically
	w_ref = max(len(r['refname']) for r in refs)
	w_ref = max(w_ref, len('pscomppars (archive composite, mixed refs)'), 30)

	# For each value column, find the widest formatted value or header
	col_widths = []
	for _, key, disp in show_cols:
		w = len(disp)
		for ref in refs:
			val = ref['values'].get(key)
			if val is not None:
				w = max(w, len(f'{val:.6g}'))
		col_widths.append(max(w, 4))

	# Header
	header = f'  {"#":<4} {"Reference":<{w_ref}}'
	for (_, _, disp), cw in zip(show_cols, col_widths):
		header += f'  {disp:<{cw}}'
	print(header)
	print('  ' + '-' * (len(header) - 2))

	for i, ref in enumerate(refs):
		line = f'  {i:<4} {ref["refname"]:<{w_ref}}'
		for (_, key, _), cw in zip(show_cols, col_widths):
			val = ref['values'].get(key)
			marker = f'{val:.6g}' if val is not None else '—'
			line += f'  {marker:<{cw}}'
		print(line)

	print(f'  {len(refs):<4} {"pscomppars (archive composite, mixed refs)":<{w_ref}}')
	print(f'  {len(refs)+1:<4} {"Manual input":<{w_ref}}')


def _select_reference(refs, col_map, param_type, label):
	"""
	Interactive reference selection. Returns:
	  primary_ref   : dict with 'refname' and 'values', or None for manual
	  supplementary : list of dicts {'col', 'param_key', 'refname', 'value'}
	"""
	n = len(refs)
	print()
	print(f'  Select primary reference for {label} parameters:')
	print(f'  (Enter index 0-{n-1} for a specific reference,')
	print(f'   {n} for pscomppars composite, {n+1} for manual input)')

	while True:
		raw = input(f'  Choice [0-{n+1}]: ').strip()
		try:
			idx = int(raw)
		except ValueError:
			print('    Please enter a number.')
			continue
		if 0 <= idx < n:
			primary = refs[idx]
			break
		elif idx == n:
			primary = None  # use pscomppars
			break
		elif idx == n + 1:
			primary = 'manual'
			break
		print(f'    Please enter a number between 0 and {n+1}.')

	if primary == 'manual':
		return None, []
	if primary is None:
		print('  Using pscomppars composite (no single reference — values will be noted as mixed).')
		return None, []

	print(f'  Primary reference: {primary["refname"]}')

	# Check for missing key params in primary reference
	supplementary = []
	missing_params = []
	for col, (key, disp) in col_map.items():
		if primary['values'].get(key) is None:
			missing_params.append((col, key, disp))

	if missing_params:
		print()
		print(f'  The following parameters are missing from this reference:')
		for col, key, disp in missing_params:
			print(f'    {disp}')
		print()
		print('  For each missing parameter, choose a supplementary reference or manual input:')

		for col, key, disp in missing_params:
			# Find refs that have this parameter
			alt_refs = [r for r in refs
			            if r['refname'] != primary['refname']
			            and r['values'].get(key) is not None]

			if not alt_refs:
				print(f'    {disp}: no alternative references found — will prompt manually.')
				continue

			print(f'    {disp} — available from:')
			for i, r in enumerate(alt_refs):
				print(f'      {i}  {r["refname"]}  ({r["values"][key]:.6g})')
			print(f'      {len(alt_refs)}  Manual input')

			while True:
				raw = input(f'      Choice [0-{len(alt_refs)}]: ').strip()
				try:
					sidx = int(raw)
				except ValueError:
					print('        Please enter a number.')
					continue
				if 0 <= sidx < len(alt_refs):
					supplementary.append({
						'col'      : col,
						'param_key': key,
						'display'  : disp,
						'refname'  : alt_refs[sidx]['refname'],
						'value'    : alt_refs[sidx]['values'][key],
					})
					break
				elif sidx == len(alt_refs):
					break  # manual — will be prompted later
				print(f'        Please enter 0-{len(alt_refs)}.')

	return primary, supplementary


# ---------------------------------------------------------------------------
# Kp calculation
# ---------------------------------------------------------------------------

def _compute_kp(period_days, a_au, incl_deg, ecc):
	"""Compute planet RV semi-amplitude Kp (km/s) from orbital elements."""
	try:
		a_km  = a_au * 1.496e8
		P_s   = period_days * 86400.0
		i_rad = incl_deg * math.pi / 180.0
		e     = ecc if ecc else 0.0
		return (2 * math.pi * a_km * math.sin(i_rad)) / (P_s * math.sqrt(1 - e**2))
	except Exception:
		return None


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------

def main():
	print()
	print('=' * 60)
	print('  ratri runner generator v2')
	print('=' * 60)
	print()

	# --- Target ---
	print('[ Target ]')
	planet_name = _ask('Planet name (e.g. "55 Cnc e", "HD 189733 b")')
	host_name   = _ask(
		'Host star name (e.g. "55 Cnc")',
		default=planet_name.rsplit(' ', 1)[0],
	)

	# --- Cache directory ---
	print()
	print('[ Cache directory ]')
	spectra_cache = _ask(
		'Spectra cache directory (all cached files go here)',
		default='./spectra_cache',
	)

	# --- Observation mode ---
	print()
	print('[ Observation mode ]')
	print('  Phase convention used by ratri:')
	print('    Phase 0      = transit (planet in front of star)')
	print('    Phase ±0.5   = occultation (planet behind star)')
	print('    0 → +0.5     = post-transit, planet moving away (dayside approaching)')
	print('    +0.5 → ±1/0  = post-occultation, planet moving back')
	print('    Negative phases mirror positive phases on the other side.')
	print()
	print('  tm = transit,  em = emission/occultation (dayside)')
	ob = _ask_choice('Observation mode', ['tm', 'em'])

	# --- Load saved params ---
	saved     = _load_params(planet_name, ob, spectra_cache)
	use_saved = False
	if saved:
		print()
		print(f'  Found saved parameters in {_params_file(planet_name, ob, spectra_cache)}')
		use_saved = _ask_yn('  Load saved parameters (press n to re-enter)?', default='y')

	def _saved_val(key, fallback=None):
		if use_saved and saved and key in saved:
			return saved[key]
		return fallback

	# ---------------------------------------------------------------------------
	# Orbital parameters — reference-aware query from ps table
	# ---------------------------------------------------------------------------
	print()
	print(f'[ Querying NASA Exoplanet Archive — orbital parameters for "{planet_name}" ]')
	orb_refs = _query_references(planet_name, host_name, 'orbital')

	orbital_ref    = None
	orbital_suppl  = []
	orb_values     = {}

	if orb_refs:
		_show_reference_table(orb_refs, ORBITAL_DISPLAY, 'orbital')
		orbital_ref, orbital_suppl = _select_reference(
			orb_refs, ORBITAL_DISPLAY, 'orbital', 'orbital'
		)
		if orbital_ref and orbital_ref != 'manual':
			orb_values = dict(orbital_ref['values'])
			# Merge supplementary values
			for s in orbital_suppl:
				orb_values[s['param_key']] = s['value']
	else:
		print('  No references found — please enter all parameters manually.')

	# Override with saved values if loading
	if use_saved:
		for key in ['period', 'tranmid_bjd', 'trandur_h', 'ecc',
		            'omega_deg', 'incl_deg', 'a_au', 'vsys']:
			if saved.get(key) is not None:
				orb_values[key] = saved[key]

	# --- Prompt for each orbital parameter ---
	print()
	print('[ Orbital parameters ]')
	print('  Press Enter to accept retrieved value, or type to override.')
	print()

	period = _ask('Orbital period (days)',
	              default=orb_values.get('period'), cast=float)

	tranmid = _ask(
		'Mid-transit time T0 (BJD_TDB)',
		default=orb_values.get('tranmid_bjd'), cast=float,
		validate=lambda v: 'Expected BJD > 2400000' if v < 2400000 else None,
	)

	trandur_h = _ask('Transit duration (hours)',
	                 default=orb_values.get('trandur_h'), cast=float)

	ecc = _ask('Eccentricity', default=orb_values.get('ecc') or 0.0, cast=float,
	           validate=lambda v: 'Must be 0 <= e < 1' if not (0 <= v < 1) else None)

	omega_deg = _ask('Argument of periastron omega (degrees)',
	                 default=orb_values.get('omega_deg') or 90.0, cast=float)

	incl_deg = _ask('Orbital inclination (degrees)',
	                default=orb_values.get('incl_deg') or 90.0, cast=float)

	a_au = _ask('Semi-major axis (AU)',
	            default=orb_values.get('a_au'), cast=float)

	vsys = _ask('Systemic velocity vsys (km/s)',
	            default=orb_values.get('vsys') or 0.0, cast=float)

	# Kp always computed from orbital elements
	kp_computed = _compute_kp(period, a_au, incl_deg, ecc)
	if kp_computed:
		print(f'  Kp computed from orbital elements: {kp_computed:.2f} km/s')
	kp = _ask('Planet RV semi-amplitude Kp (km/s)',
	          default=round(kp_computed, 2) if kp_computed else None, cast=float)

	# ---------------------------------------------------------------------------
	# Stellar parameters — reference-aware query from stellarhosts
	# ---------------------------------------------------------------------------
	print()
	print(f'[ Querying NASA Exoplanet Archive — stellar parameters for "{host_name}" ]')
	st_refs = _query_references(planet_name, host_name, 'stellar')

	stellar_ref   = None
	stellar_suppl = []

	if st_refs:
		_show_reference_table(st_refs, STELLAR_DISPLAY, 'stellar')
		stellar_ref, stellar_suppl = _select_reference(
			st_refs, STELLAR_DISPLAY, 'stellar', 'stellar'
		)
	else:
		print('  No stellar references found — stellar.py will query the archive at runtime.')

	# --- Instrument ---
	print()
	print('[ Instrument ]')
	inst_choices = ['crires+', 'carmenes', 'giano', 'spirou', 'andes',
	                'andes_ccd_carmenes', 'tmt_ccd_giano', 'nlot_ccd_giano']
	inst = _ask_choice('Instrument', inst_choices, default=_saved_val('inst'))

	modes_str = '[]'
	if inst == 'crires+':
		print('  Enter CRIRES+ wavelength modes as comma-separated list')
		print('  (e.g. "l3340,l3380" or leave blank for none)')
		raw_modes = input('  Modes: ').strip()
		if raw_modes:
			modes_list = [m.strip() for m in raw_modes.split(',') if m.strip()]
			modes_str  = repr(modes_list)

	if inst == 'andes':
		print('  Supported ANDES modes: YJ, H, K, UBV, RIZ')
		print('  Enter as comma-separated list (e.g. "YJ,K" or "H,K")')
		raw_modes = input('  Modes: ').strip()
		if raw_modes:
			modes_list = [m.strip() for m in raw_modes.split(',') if m.strip()]
			invalid = [m for m in modes_list if m not in ('YJ','H','K','UBV','RIZ')]
			if invalid:
				print(f'    Warning: unrecognised modes {invalid} — double-check before running')
			modes_str = repr(modes_list)
		else:
			print('    No modes selected — defaulting to YJ,H,K')
			modes_str = repr(['YJ','H','K'])

	if inst == 'andes_ccd_carmenes':
		print('  andes_ccd_carmenes supports one mode only: YJH')
		modes_str = repr(['YJH'])

	t_exp = _ask('Exposure time (seconds)', default=_saved_val('t_exp', 300), cast=int)

	# --- Search parameters ---
	print()
	print('[ Search parameters ]')

	start_str = _ask(
		'Search start date (YYYY-MM-DD)',
		default=_saved_val('start_str'), cast=str,
		validate=lambda v: None if re.match(r'\d{4}-\d{2}-\d{2}', v)
		                       else 'Format must be YYYY-MM-DD',
	)

	from datetime import datetime, timedelta
	start_dt    = datetime.strptime(start_str, '%Y-%m-%d')
	end_default = (start_dt + timedelta(days=730)).strftime('%Y-%m-%d')
	end_str = _ask(
		'Search end date (YYYY-MM-DD)',
		default=_saved_val('end_str', end_default), cast=str,
		validate=lambda v: None if re.match(r'\d{4}-\d{2}-\d{2}', v)
		                       else 'Format must be YYYY-MM-DD',
	)

	airmass_min = _ask('Minimum airmass', default=_saved_val('airmass_min', 1.0), cast=float)
	airmass_max = _ask('Maximum airmass', default=_saved_val('airmass_max', 2.0), cast=float)
	min_dur     = _ask('Minimum observable window (hours)', default=_saved_val('min_dur', 1.5), cast=float)

	# --- Phase window ---
	print()
	print('[ Phase window ]')
	print('  Phase 0 = transit, Phase ±0.5 = occultation')
	print('  Positive phases: post-transit dayside')
	print('  Negative phases: post-occultation')
	print()

	if ob == 'em':
		occ_half      = (trandur_h / 2.0) / (period * 24.0)
		suggested_min = round(0.5 - 3 * occ_half, 3)
		suggested_max = round(-0.5 + 3 * occ_half, 3)
		print(f'  Suggested dayside window (3x occultation half-duration around ±0.5):')
		print(f'    PHASE_MIN = +{suggested_min},  PHASE_MAX = {suggested_max}')
	else:
		tr_half       = (trandur_h / 2.0) / (period * 24.0)
		suggested_min = round(-3 * tr_half, 3)
		suggested_max = round(3 * tr_half, 3)
		print(f'  Suggested transit window (3x transit half-duration around 0):')
		print(f'    PHASE_MIN = {suggested_min},  PHASE_MAX = +{suggested_max}')

	phase_min = _ask('PHASE_MIN', default=_saved_val('phase_min', suggested_min), cast=float)
	phase_max = _ask('PHASE_MAX', default=_saved_val('phase_max', suggested_max), cast=float)

	# --- File paths ---
	print()
	print('[ File paths ]')
	atm_dir      = _ask('ATM transmission directory',
	                    default=_saved_val('atm_dir', './atm_transmission'))
	planet_model = _ask('Planet model path (or "none")',
	                    default=_saved_val('planet_model', 'none'))
	planet_model_str = 'None' if planet_model.lower() == 'none' else repr(planet_model)

	safe_name    = _safe_planet_name(planet_name)
	savedir_base = _ask('Base output directory',
	                    default=_saved_val('savedir_base', f'./output/{safe_name}_{ob}/'))

	# --- Stellar library ---
	print()
	print('[ Stellar atmosphere library ]')
	print('  auto     : auto-select based on T_eff (recommended)')
	print('  phoenix  : PHOENIX/ACES (Husser et al. 2013) — T_eff 2300-7000 K')
	print('  btsettl  : BT-Settl (Allard et al.) — T_eff < 3500 K')
	print('  atlas9   : ATLAS9/Kurucz (Castelli & Kurucz 2004) — T_eff > 7000 K')
	print('  marcs    : MARCS (Gustafsson et al. 2008) — cool giants')
	lib_choice  = _ask_choice(
		'Stellar library',
		['auto', 'phoenix', 'btsettl', 'atlas9', 'marcs'],
		default=_saved_val('library', 'auto'),
	)
	library_val = 'None' if lib_choice == 'auto' else repr(lib_choice)

	# --- Stellar parameter overrides ---
	# Collect values from selected reference + supplementary first.
	# Then prompt for any parameters still missing.
	stellar_overrides = {}
	stellar_params_all = [
		('teff',     'T_eff',    'K'),
		('logg',     'log g',    'cgs'),
		('feh',      '[Fe/H]',   'dex'),
		('radius',   'R_star',   'R_sun'),
		('mass',     'M_star',   'M_sun'),
		('distance', 'Distance', 'pc'),
	]

	# Build dict of values found from reference selection
	stellar_from_refs = {}
	if stellar_ref and stellar_ref not in ('manual', None):
		for col, (key, disp) in STELLAR_DISPLAY.items():
			v = stellar_ref['values'].get(key)
			if v is not None:
				stellar_from_refs[key] = v
		for s in stellar_suppl:
			if s['value'] is not None:
				stellar_from_refs[s['param_key']] = s['value']

	# Show what was found from references
	print()
	print('[ Stellar parameters ]')
	if stellar_from_refs:
		print('  Values from selected reference(s):')
		for key, label, unit in stellar_params_all:
			v = stellar_from_refs.get(key)
			if v is not None:
				print(f'    {label} ({unit}): {v}')
			else:
				print(f'    {label} ({unit}): not found in reference')

	# Prompt to override any value (including ones found in references)
	print()
	print('  You can override any stellar value here.')
	print('  Press Enter to skip (keep reference value or leave unset).')
	for key, label, unit in stellar_params_all:
		ref_val = stellar_from_refs.get(key)
		raw = input(f'    {label} ({unit}) [{ref_val if ref_val is not None else "not set"}]: ').strip()
		if raw:
			try:
				stellar_overrides[key] = float(raw)
			except ValueError:
				print(f'      Invalid — keeping {ref_val}')

	# Write stellar values to cache:
	# reference values as base, manual overrides on top
	stellar_cache_updates = dict(stellar_from_refs)
	stellar_cache_updates.update(stellar_overrides)
	# Add reference provenance so it appears in the cache
	if stellar_cache_updates:
		if stellar_ref and stellar_ref not in ('manual', None):
			ref_parts = [stellar_ref['refname']]
			for s in stellar_suppl:
				ref_parts.append(f'{s["display"]} from {s["refname"]}')
			stellar_cache_updates['source'] = '; '.join(ref_parts)
		elif stellar_overrides:
			stellar_cache_updates['source'] = 'Manual input via generate_runner_v2'

	if stellar_cache_updates:
		try:
			sys.path.insert(0, '.')
			from stellar import update_cache
			update_cache(planet_name, stellar_cache_updates, spectra_cache)
			print('  Stellar parameters written to cache.')
		except Exception as e:
			print(f'  Could not write to cache ({e}) — enter overrides manually later.')

	# --- Output filename ---
	out_filename = _ask('Output script filename', default=f'{safe_name}_{ob}.py')

	# ---------------------------------------------------------------------------
	# Build reference strings for saving and embedding in generated script
	# ---------------------------------------------------------------------------

	def _ref_str(ref):
		"""Format reference name for saving/display."""
		if ref is None or ref == 'manual':
			return 'pscomppars (archive composite, mixed references)'
		return ref['refname']

	orbital_ref_str = _ref_str(orbital_ref)
	stellar_ref_str = _ref_str(stellar_ref)

	orbital_suppl_list = [
		{'param': s['display'], 'col': s['col'],
		 'refname': s['refname'], 'value': s['value']}
		for s in orbital_suppl
	]
	stellar_suppl_list = [
		{'param': s['display'], 'col': s['col'],
		 'refname': s['refname'], 'value': s['value']}
		for s in stellar_suppl
	]

	# Build reference comment block for generated script
	ref_comment_lines = [
		f'# Orbital parameters: {orbital_ref_str}',
	]
	for s in orbital_suppl_list:
		ref_comment_lines.append(f'#   {s["param"]} supplemented from: {s["refname"]}')
	if orbital_ref is None:
		ref_comment_lines.append(
			'#   WARNING: pscomppars mixes references — cite with caution'
		)
	ref_comment_lines.append(f'# Stellar parameters: {stellar_ref_str}')
	for s in stellar_suppl_list:
		ref_comment_lines.append(f'#   {s["param"]} supplemented from: {s["refname"]}')
	if stellar_ref is None:
		ref_comment_lines.append(
			'#   WARNING: pscomppars mixes references — cite with caution'
		)
	ref_comment_block = '\n'.join(ref_comment_lines)

	# ---------------------------------------------------------------------------
	# Save all parameters to cache
	# ---------------------------------------------------------------------------
	params_to_save = dict(
		planet_name        = planet_name,
		star_name          = host_name,
		period             = period,
		tranmid_bjd        = tranmid,
		trandur_h          = trandur_h,
		ecc                = ecc,
		omega_deg          = omega_deg,
		incl_deg           = incl_deg,
		a_au               = a_au,
		vsys               = vsys,
		kp                 = kp,
		inst               = inst,
		modes_str          = modes_str,
		t_exp              = t_exp,
		start_str          = start_str,
		end_str            = end_str,
		airmass_min        = airmass_min,
		airmass_max        = airmass_max,
		min_dur            = min_dur,
		phase_min          = phase_min,
		phase_max          = phase_max,
		atm_dir            = atm_dir,
		spectra_cache      = spectra_cache,
		planet_model       = planet_model,
		savedir_base       = savedir_base,
		library            = lib_choice,
		orbital_reference  = orbital_ref_str,
		orbital_supplementary = orbital_suppl_list,
		stellar_reference  = stellar_ref_str,
		stellar_supplementary = stellar_suppl_list,
	)
	_save_params(params_to_save, planet_name, ob, spectra_cache)

	# ---------------------------------------------------------------------------
	# Generate the runner script
	# ---------------------------------------------------------------------------
	print()
	print(f'Writing {out_filename}...')

	omega_rad = omega_deg * math.pi / 180.0

	script = f'''#!/usr/bin/env python3
"""
{out_filename} — ratri runner for {planet_name} ({ob.upper()})

Generated by generate_runner_v2.py.

Usage:
    python {out_filename}              # plan nights then simulate
    python {out_filename} --plan       # search and save night list only
    python {out_filename} --simulate   # load saved night list and simulate
"""

import argparse
import json
import os
import warnings
import numpy as np
from astropy.time import Time
import astropy.units as u

# Suppress ERFA/IERS warnings for dates beyond the IERS data validity range
try:
	import erfa
	warnings.filterwarnings('ignore', category=erfa.ErfaWarning)
except ImportError:
	pass
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=AstropyWarning)
import logging
logging.getLogger('astropy').setLevel(logging.ERROR)

from ratri import ratri_auto, _select_night, _OBSERVATORIES, _PWV_TIERS


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

ATM_DIR       = {repr(atm_dir)}
SPECTRA_CACHE = {repr(spectra_cache)}
PLANET_MODEL  = {planet_model_str}
SAVEDIR_BASE  = {repr(savedir_base)}
NIGHTS_FILE   = {repr(os.path.join(spectra_cache, safe_name + '_' + ob + '_nights.json'))}


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

INST  = {repr(inst)}
MODES = {modes_str}
T_EXP = {t_exp}


# ---------------------------------------------------------------------------
# Observation mode
# ---------------------------------------------------------------------------

# Phase convention:
#   Phase 0    = transit centre (planet in front of star)
#   Phase ±0.5 = occultation centre (planet behind star)
#   Positive phases: post-transit, planet moving away from observer
#   Negative phases: post-occultation, planet returning toward observer
OB = {repr(ob)}


# ---------------------------------------------------------------------------
# Search parameters
# ---------------------------------------------------------------------------

T_SEARCH_START = Time('{start_str} 00:00:00', scale='utc')
T_SEARCH_END   = Time('{end_str} 00:00:00', scale='utc')

# Phase window
# {"Dayside (emission): wrap-around window centred on occultation at phase +-0.5" if ob == "em" else "Transit: symmetric window centred on phase 0"}
PHASE_MIN    = {phase_min}
PHASE_MAX    = {phase_max}

MIN_DURATION = {min_dur}    # minimum observable window in hours
AIRMASS_MIN  = {airmass_min}
AIRMASS_MAX  = {airmass_max}


# ---------------------------------------------------------------------------
# Orbital and system parameters
{ref_comment_block}
# ---------------------------------------------------------------------------

MEAN_TRANSIT_BJD = {tranmid}
T0               = Time(MEAN_TRANSIT_BJD, format='jd', scale='tdb')
PERIOD           = {period} * u.day
TR_DURATION_H    = {trandur_h}    # hours

E          = {ecc}
OMEGA      = {omega_rad:.6f}    # radians  ({omega_deg} deg)
VSYS       = {vsys}             # km/s
KP         = {kp}               # km/s

print(f'Kp = {{KP:.2f}} km/s')


# ---------------------------------------------------------------------------
# Shared keyword arguments for ratri_auto
# ---------------------------------------------------------------------------

def _common_kwargs(savedir, qual):
	return dict(
		planet_name       = {repr(planet_name)},
		star_name         = {repr(host_name)},
		inst              = INST,
		modes             = MODES,
		ob                = OB,
		t_exp             = T_EXP,
		T0                = T0,
		period            = PERIOD,
		mean_transit_BJD  = MEAN_TRANSIT_BJD,
		tr_duration_h     = TR_DURATION_H,
		e                 = E,
		omega             = OMEGA,
		vsys              = VSYS,
		Kp                = KP,
		t_search_start    = T_SEARCH_START,
		t_search_end      = T_SEARCH_END,
		phase_min         = PHASE_MIN,
		phase_max         = PHASE_MAX,
		airmass_min       = AIRMASS_MIN,
		airmass_max       = AIRMASS_MAX,
		min_duration      = MIN_DURATION,
		planet_model_path = PLANET_MODEL,
		atm_dir           = ATM_DIR,
		spectra_cache_dir = SPECTRA_CACHE,
		library           = {library_val},
		noise_seed        = 3,
		savedir           = savedir,
		qual              = qual,
	)


def _print_summary(results, savedir):
	chosen = results['chosen_night']
	orbit  = results['orbit']
	flux   = results['flux_cube']
	print(f'\\nSimulation complete.')
	print(f'  Night         : {{chosen["night"]}}')
	print(f'  n_spectra     : {{flux.shape[1]}}')
	print(f'  n_orders      : {{flux.shape[0]}}')
	print(f'  Phase range   : [{{orbit["ph"].min():+.4f}}, {{orbit["ph"].max():+.4f}}]')
	print(f'  Airmass range : [{{orbit["airmass"].min():.2f}}, {{orbit["airmass"].max():.2f}}]')
	print(f'  Stellar T_eff : {{results["stellar_params"]["teff"]:.0f}} K')
	print(f'  Library used  : {{results["spectrum_paths"]["library"]}}')
	print(f'  Saved to      : {{savedir}}')


# ---------------------------------------------------------------------------
# Plan step: search for observable nights and save to JSON
# ---------------------------------------------------------------------------

def plan():
	nights = _select_night(
		target_name  = {repr(planet_name)},
		inst         = INST,
		t_start      = T_SEARCH_START,
		t_end        = T_SEARCH_END,
		T0           = T0,
		period       = PERIOD,
		phase_min    = PHASE_MIN,
		phase_max    = PHASE_MAX,
		airmass_min  = AIRMASS_MIN,
		airmass_max  = AIRMASS_MAX,
		min_duration = MIN_DURATION,
	)

	if not nights:
		print('\\nNo qualifying nights found. Try adjusting the search parameters.')
		return None

	nights_json = []
	for i, n in enumerate(nights):
		nights_json.append(dict(
			rank            = i,
			night           = n['night'],
			evening_date    = n['evening_date'],
			total_hours     = n['total_hours'],
			total_hours_all = n['total_hours_all'],
			phase_str       = n['phase_str'],
			airmass_range   = list(n['airmass_range']),
			obs_start       = n['obs_start'].iso,
			obs_end         = n['obs_end'].iso,
		))

	with open(NIGHTS_FILE, 'w') as f:
		json.dump(nights_json, f, indent=2)
	print(f'\\nNight list saved to {{NIGHTS_FILE}}')
	return nights


# ---------------------------------------------------------------------------
# Simulate step: load saved night list and simulate chosen ranks
# ---------------------------------------------------------------------------

def simulate(nights=None):
	if nights is None:
		if not os.path.exists(NIGHTS_FILE):
			print(f'Night list not found: {{NIGHTS_FILE}}')
			print('Run with --plan first, or without arguments to do both.')
			return
		with open(NIGHTS_FILE) as f:
			nights_json = json.load(f)

		nights = []
		for n in nights_json:
			nights.append(dict(
				rank            = n['rank'],
				night           = n['night'],
				evening_date    = n['evening_date'],
				total_hours     = n['total_hours'],
				total_hours_all = n['total_hours_all'],
				phase_str       = n['phase_str'],
				airmass_range   = tuple(n['airmass_range']),
				obs_start       = Time(n['obs_start'], scale='utc'),
				obs_end         = Time(n['obs_end'],   scale='utc'),
			))

		print(f'\\nLoaded {{len(nights)}} qualifying nights from {{NIGHTS_FILE}}:')
		print(f'  {{"Rank":<5}} {{"Night":<16}} {{"Obs. Hours (Total)":<22}} {{"Phase Coverage":<48}} {{"Airmass"}}')
		print('  ' + '-' * 105)
		for n in nights:
			hrs_str   = f\'{{n["total_hours"]:.1f}} ({{n["total_hours_all"]:.1f}})\'
			phase_str = n['phase_str'].ljust(50)
			print(
				f\'  {{n["rank"]:<5}} \'
				f\'{{n["night"]:<16}} \'
				f\'{{hrs_str:<22}}\'
				f\'{{phase_str:<52}}\'
				f\'[{{n["airmass_range"][0]:.2f}}, {{n["airmass_range"][1]:.2f}}]\'
			)

	print(f'\\nEnter rank(s) to simulate — 0-indexed, space-separated:')
	raw = input('  Ranks: ').strip()
	try:
		ranks = [int(r) for r in raw.split()]
	except ValueError:
		print('Invalid input.')
		return
	ranks = [r for r in ranks if 0 <= r < len(nights)]
	if not ranks:
		print(f'No valid ranks in range [0, {{len(nights)-1}}].')
		return

	site_default = _OBSERVATORIES[INST]['default_qual']
	tier_str     = '/'.join(_PWV_TIERS.keys())
	night_quals  = {{}}
	for rank in ranks:
		night_label = nights[rank]['night']
		raw_q = input(
			f\'  Rank {{rank}} ({{night_label}}) — PWV quality \'
			f\'[{{tier_str}}, default={{site_default}}]: \'
		).strip().lower()
		night_quals[rank] = raw_q if raw_q in _PWV_TIERS else site_default
		chosen_q = night_quals[rank]
		if not raw_q:
			print(f\'    Using site default: {{chosen_q}}\')
		elif raw_q not in _PWV_TIERS:
			print(f\'    Unrecognised tier "{{raw_q}}", using site default: {{chosen_q}}\')
		else:
			print(f\'    PWV quality set to: {{chosen_q}}\')

	for rank in ranks:
		evening_date = nights[rank]['evening_date']
		savedir      = f\'{{SAVEDIR_BASE}}{{evening_date}}/\'

		print(f\'\\n{{"="*60}}\')
		print(f\'Simulating rank {{rank}}: night {{nights[rank]["night"]}}\')
		print(f\'Output directory: {{savedir}}\')
		print(f\'{{"="*60}}\')

		results = ratri_auto(
			**_common_kwargs(savedir=savedir, qual=night_quals[rank]),
			chosen_night = nights[rank],
		)

		if results is None:
			print(f\'Rank {{rank}} returned no results — skipping.\')
			continue

		_print_summary(results, savedir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description=f\'ratri runner for {planet_name} ({ob.upper()})\'
	)
	parser.add_argument('--plan',     action='store_true',
	                    help='Search for observable nights and save to JSON only')
	parser.add_argument('--simulate', action='store_true',
	                    help='Load saved night list and simulate (skips search)')
	args = parser.parse_args()

	if args.plan and not args.simulate:
		plan()
	elif args.simulate and not args.plan:
		simulate()
	else:
		nights = plan()
		if nights:
			simulate(nights=nights)
'''

	with open(out_filename, 'w') as f:
		f.write(script)

	print(f'\nDone. Generated: {out_filename}')
	print()
	print('Usage:')
	print(f'  python {out_filename}             # plan + simulate')
	print(f'  python {out_filename} --plan      # search only')
	print(f'  python {out_filename} --simulate  # simulate from saved night list')


if __name__ == '__main__':
	main()
