#!/usr/bin/env python3
"""
stellar.py - Stellar parameter lookup and atmosphere model downloader for ratri

Queries NASA Exoplanet Archive (primary) and SIMBAD (fallback) for stellar
parameters of a named planet host, snaps to the nearest model grid point,
and downloads the appropriate spectrum if not already cached.

Supported model libraries:
	PHOENIX/ACES  (Husser et al. 2013)     — best for T_eff 2300-7000 K
	BT-Settl      (Allard et al.)          — best for T_eff < 3500 K (M dwarfs, brown dwarfs)
	ATLAS9/Kurucz (Castelli & Kurucz 2004) — best for T_eff > 7000 K (A, F stars)
	MARCS         (Gustafsson et al. 2008) — good for FGK giants

Usage:
	from stellar import get_stellar_params, recommend_library, download_spectrum

	params = get_stellar_params('55 Cnc')
	lib    = recommend_library(params['teff'])
	paths  = download_spectrum(params, library=lib, cache_dir='/path/to/cache')
"""

import os
import urllib.request
import numpy as np


# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

PHOENIX_TEFF_GRID = np.array(
	list(range(2300, 7001, 100)) + list(range(7200, 12001, 200))
)
PHOENIX_LOGG_GRID = np.arange(0.0, 6.5, 0.5)
PHOENIX_FEH_GRID  = np.array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])

BTSETTL_TEFF_GRID = np.arange(400, 7001, 100)
BTSETTL_LOGG_GRID = np.arange(2.5, 5.6, 0.5)
BTSETTL_FEH_GRID  = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5])

ATLAS9_TEFF_GRID  = np.arange(3500, 50001, 250)
ATLAS9_LOGG_GRID  = np.arange(0.0, 5.1, 0.5)
ATLAS9_FEH_GRID   = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.2, 0.5])

MARCS_TEFF_GRID   = np.array(
	list(range(2500, 4001, 100)) + list(range(4000, 8001, 250))
)
MARCS_LOGG_GRID   = np.arange(-0.5, 5.6, 0.5)
MARCS_FEH_GRID    = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

LIBRARY_GRIDS = {
	'phoenix':  (PHOENIX_TEFF_GRID, PHOENIX_LOGG_GRID, PHOENIX_FEH_GRID),
	'btsettl':  (BTSETTL_TEFF_GRID, BTSETTL_LOGG_GRID, BTSETTL_FEH_GRID),
	'atlas9':   (ATLAS9_TEFF_GRID,  ATLAS9_LOGG_GRID,  ATLAS9_FEH_GRID),
	'marcs':    (MARCS_TEFF_GRID,   MARCS_LOGG_GRID,   MARCS_FEH_GRID),
}

LIBRARY_DESCRIPTIONS = {
	'phoenix':  'PHOENIX/ACES (Husser et al. 2013)     — T_eff 2300-7000 K, FGK/M stars',
	'btsettl':  'BT-Settl     (Allard et al.)          — T_eff < 3500 K, M dwarfs / brown dwarfs',
	'atlas9':   'ATLAS9/Kurucz (Castelli & Kurucz 2004)— T_eff > 7000 K, A/F/B stars',
	'marcs':    'MARCS        (Gustafsson et al. 2008)  — FGK cool giants, 2500-8000 K',
}


# ---------------------------------------------------------------------------
# Library recommendation
# ---------------------------------------------------------------------------

def recommend_library(teff):
	"""
	Recommend a stellar atmosphere library based on T_eff and print
	suggestions for alternatives.

	Parameters
	----------
	teff : float   effective temperature in K

	Returns
	-------
	str   recommended library key ('phoenix', 'btsettl', 'atlas9', 'marcs')
	"""
	if teff < 3500:
		default = 'btsettl'
	elif teff <= 7000:
		default = 'phoenix'
	else:
		default = 'atlas9'

	print(f'\nStellar atmosphere library recommendation for T_eff = {teff:.0f} K:')
	print(f'  Default : {LIBRARY_DESCRIPTIONS[default]}')
	print('  Alternatives:')
	for key, desc in LIBRARY_DESCRIPTIONS.items():
		if key != default:
			teff_grid = LIBRARY_GRIDS[key][0]
			if teff_grid.min() <= teff <= teff_grid.max():
				coverage = '(T_eff in range)'
			else:
				coverage = f'(T_eff range: {teff_grid.min():.0f}-{teff_grid.max():.0f} K — out of range)'
			print(f'    {desc}  {coverage}')

	return default


# ---------------------------------------------------------------------------
# Grid snapping
# ---------------------------------------------------------------------------

def _snap(value, grid):
	"""Return the nearest value in grid to value."""
	return grid[np.argmin(np.abs(grid - value))]


def snap_to_grid(teff, logg, feh, library='phoenix'):
	"""
	Snap stellar parameters to the nearest grid point for the given library.

	Parameters
	----------
	teff    : float
	logg    : float
	feh     : float
	library : str

	Returns
	-------
	dict with keys: teff, logg, feh (all snapped)
	"""
	teff_grid, logg_grid, feh_grid = LIBRARY_GRIDS[library]

	teff_snap = float(_snap(teff, teff_grid))
	logg_snap = float(_snap(logg, logg_grid))
	feh_snap  = float(_snap(feh,  feh_grid))

	print(f'\nGrid snapping ({library}):')
	print(f'  T_eff : {teff:.1f} K  ->  {teff_snap:.0f} K')
	print(f'  log g : {logg:.2f}   ->  {logg_snap:.1f}')
	print(f'  [Fe/H]: {feh:.2f}    ->  {feh_snap:+.1f}')

	return dict(teff=teff_snap, logg=logg_snap, feh=feh_snap)


# ---------------------------------------------------------------------------
# Stellar parameter lookup
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Stellar parameter lookup
# ---------------------------------------------------------------------------

def _safe_float(val):
	"""Return float(val) or None if val is None, masked, or NaN.

	Handles astropy MaskedQuantity objects (which carry units) by
	extracting the numeric value before conversion.
	"""
	try:
		if val is None:
			return None
		# MaskedQuantity: strip units via .value
		if hasattr(val, 'value'):
			v = float(val.value)
		else:
			v = float(val)
		import math
		return None if (math.isnan(v) or math.isinf(v)) else v
	except Exception:
		return None


def _logg_from_mass_radius(mass, radius):
	"""
	Compute log10(g / cm s^-2) from stellar mass (M_sun) and radius (R_sun).
	Returns None if either input is None.
	"""
	if mass is None or radius is None:
		return None
	import math
	G      = 6.674e-8     # cm^3 g^-1 s^-2
	M_sun  = 1.989e33     # g
	R_sun  = 6.957e10     # cm
	g      = G * (mass * M_sun) / (radius * R_sun) ** 2
	return round(math.log10(g), 3)


def _best_row(table, key_cols):
	"""Return the row from table with the most non-null values in key_cols."""
	best_row   = None
	best_count = -1
	for row in table:
		count = sum(1 for k in key_cols if _safe_float(row[k]) is not None)
		if count > best_count:
			best_count = count
			best_row   = row
	return best_row, best_count


def _query_nasa_astroquery(planet_name):
	"""
	Query NASA Exoplanet Archive via astroquery (primary method).

	Table query order:
	  1. stellarhosts — dedicated stellar parameters table, best radius coverage
	  2. pscomppars   — composite planet/star parameters
	  3. ps           — individual publications

	log g is derived from mass + radius if st_logg is missing.

	Returns dict or None.
	"""
	try:
		try:
			from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
		except ImportError:
			from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
	except ImportError:
		print('  astroquery not installed. Run: pip install astroquery')
		return None

	key_cols = ['st_teff', 'st_logg', 'st_met', 'st_rad', 'st_mass', 'sy_dist']

	# stellarhosts uses hostname instead of pl_name
	# Extract host name by stripping trailing planet letter
	host_name = planet_name.rsplit(' ', 1)[0]

	# Query each table in order. For stellarhosts we don't require st_teff
	# to be non-null since some rows have radius/mass but not teff.
	# We merge results across tables to get the most complete parameter set.
	result = {}

	for table_name, where_col, where_val, require_teff in [
		('pscomppars',   'pl_name',  planet_name, True),
		('ps',           'pl_name',  planet_name, True),
		('stellarhosts', 'hostname', host_name,   False),
	]:
		try:
			print(f'  Trying astroquery ({table_name})...', end=' ', flush=True)
			where = f"{where_col}='{where_val}'"
			if require_teff:
				where += ' and st_teff is not null'
			select_cols = ','.join(key_cols)
			table = NasaExoplanetArchive.query_criteria(
				table  = table_name,
				where  = where,
				select = select_cols,
			)
			if table is None or len(table) == 0:
				print('no results.')
				continue

			row, n_found = _best_row(table, key_cols)
			if row is None:
				print('no usable rows.')
				continue

			print(f'ok. ({n_found}/{len(key_cols)} params found)')

			# Fill in any still-missing parameters from this table
			for col, key in [
				('st_teff',  'teff'),
				('st_logg',  'logg'),
				('st_met',   'feh'),
				('st_rad',   'radius'),
				('st_mass',  'mass'),
				('sy_dist',  'distance'),
			]:
				if result.get(key) is None:
					v = _safe_float(row[col])
					if v is not None:
						result[key] = v

			result['source'] = result.get('source', f'NASA Exoplanet Archive (astroquery/{table_name})')

			# Stop early if we have all key params
			if all(result.get(k) is not None for k in ['teff', 'radius', 'mass']):
				break

		except Exception as e:
			print(f'failed: {e}')
			continue

	if not result or result.get('teff') is None:
		return None

	# Derive log g from mass + radius if still missing
	if result.get('logg') is None:
		logg = _logg_from_mass_radius(result.get('mass'), result.get('radius'))
		if logg is not None:
			result['logg'] = logg

	return dict(
		teff     = result.get('teff'),
		logg     = result.get('logg'),
		feh      = result.get('feh'),
		radius   = result.get('radius'),
		mass     = result.get('mass'),
		distance = result.get('distance'),
		source   = result.get('source', 'NASA Exoplanet Archive (astroquery)'),
	)


def _query_nasa_tap(planet_name, timeout=30, n_retries=3):
	"""
	Query NASA Exoplanet Archive TAP endpoint via urllib.
	Retries with increasing timeout. Used as fallback if astroquery fails.

	Returns dict or None.
	"""
	import urllib.parse
	import json
	import time

	adql = (
		"SELECT TOP 1 "
		"st_teff, st_logg, st_met, st_rad, st_mass, sy_dist "
		"FROM ps "
		f"WHERE pl_name = '{planet_name}' "
		"AND st_teff IS NOT NULL "
		"ORDER BY pl_pubdate DESC"
	)
	# TAP sync endpoint — POST request required per IVOA TAP standard
	# Ref: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
	tap_url  = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync'
	tap_body = urllib.parse.urlencode({'query': adql, 'format': 'json'}).encode()

	for attempt in range(1, n_retries + 1):
		try:
			print(f'  TAP attempt {attempt}/{n_retries} (timeout={timeout}s)...', end=' ', flush=True)
			req  = urllib.request.Request(tap_url, data=tap_body, headers={'User-Agent': 'Mozilla/5.0'})
			with urllib.request.urlopen(req, timeout=timeout) as resp:
				data = json.loads(resp.read().decode())
			if not data:
				print('no results.')
				return None
			row    = data[0]
			teff   = _safe_float(row.get('st_teff'))
			logg   = _safe_float(row.get('st_logg'))
			feh    = _safe_float(row.get('st_met'))
			radius = _safe_float(row.get('st_rad'))
			mass   = _safe_float(row.get('st_mass'))
			dist   = _safe_float(row.get('sy_dist'))

			if logg is None:
				logg = _logg_from_mass_radius(mass, radius)

			print('ok.')
			return dict(
				teff     = teff,
				logg     = logg,
				feh      = feh,
				radius   = radius,
				mass     = mass,
				distance = dist,
				source   = 'NASA Exoplanet Archive (TAP)',
			)
		except Exception as e:
			print(f'failed: {e}')
			if attempt < n_retries:
				wait = 5 * attempt
				print(f'  Waiting {wait}s before retry...')
				time.sleep(wait)
			timeout = int(timeout * 1.5)

	return None


def _query_simbad(star_name):
	"""
	Query SIMBAD via astroquery for stellar parameters.
	Used as last-resort fallback.

	Returns dict or None.
	"""
	try:
		from astroquery.simbad import Simbad
	except ImportError:
		print('  astroquery not installed. Run: pip install astroquery')
		return None

	simbad = Simbad()
	simbad.add_votable_fields('fe_h', 'plx', 'sp')

	try:
		result = simbad.query_object(star_name)
		if result is None:
			return None

		plx      = result['PLX_VALUE'][0]
		distance = 1000.0 / float(plx) if (plx and float(plx) > 0) else None
		feh_val  = result['Fe_H_Fe_H'][0]
		feh      = float(feh_val) if feh_val else 0.0
		sp_type  = result['SP_TYPE'][0] if 'SP_TYPE' in result.colnames else None
		teff     = _teff_from_sptype(str(sp_type)) if sp_type else None

		return dict(
			teff     = teff,
			logg     = None,
			feh      = feh,
			radius   = None,
			mass     = None,
			distance = distance,
			source   = 'SIMBAD',
		)
	except Exception as e:
		print(f'  SIMBAD query failed: {e}')
		return None


def _teff_from_sptype(sptype):
	"""Rough T_eff estimate from spectral type string."""
	sp_map = {
		'O': 40000, 'B': 20000, 'A': 9000,
		'F': 7000,  'G': 5500,  'K': 4500, 'M': 3200,
	}
	sptype   = sptype.strip().upper()
	sp_class = sptype[0] if sptype else ''
	if sp_class not in sp_map:
		return None
	try:
		subtype    = float(sptype[1])
		base       = sp_map[sp_class]
		next_class = {'O':'B','B':'A','A':'F','F':'G','G':'K','K':'M','M':'M'}
		next_teff  = sp_map[next_class[sp_class]]
		return base + (subtype / 10.0) * (next_teff - base)
	except (IndexError, ValueError):
		return sp_map.get(sp_class)



# ---------------------------------------------------------------------------
# Stellar parameter cache
# ---------------------------------------------------------------------------

_CACHE_FILENAME = 'stellar_params_cache.json'
_CACHE_REQUIRED_KEYS = ['teff', 'logg', 'feh', 'radius', 'mass', 'distance']


def _load_cache(cache_dir):
	"""Load the stellar parameter cache from cache_dir. Returns a dict."""
	import json
	path = os.path.join(cache_dir, _CACHE_FILENAME)
	if os.path.exists(path):
		with open(path) as f:
			return json.load(f)
	return {}


def _save_cache(cache, cache_dir):
	"""Save the stellar parameter cache to cache_dir."""
	import json
	os.makedirs(cache_dir, exist_ok=True)
	path = os.path.join(cache_dir, _CACHE_FILENAME)
	with open(path, 'w') as f:
		json.dump(cache, f, indent=2)


def _cache_complete(entry):
	"""Return True if the cache entry has all required keys with non-null values."""
	return all(entry.get(k) is not None for k in _CACHE_REQUIRED_KEYS)


def update_cache(planet_name, params, cache_dir):
	"""
	Manually update a cache entry. Useful for correcting missing or wrong values.

	Example
	-------
	>>> update_cache('55 Cnc e', {'feh': 0.35}, './spectra_cache')
	"""
	cache = _load_cache(cache_dir)
	if planet_name not in cache:
		cache[planet_name] = {}
	cache[planet_name].update(params)
	_save_cache(cache, cache_dir)
	print(f'Cache updated for "{planet_name}": {params}')


def show_cache(cache_dir):
	"""Print the current contents of the stellar parameter cache."""
	import json
	cache = _load_cache(cache_dir)
	if not cache:
		print(f'Cache is empty: {os.path.join(cache_dir, _CACHE_FILENAME)}')
		return
	print(f'Stellar parameter cache ({os.path.join(cache_dir, _CACHE_FILENAME)}):')
	for planet, entry in cache.items():
		print(f'  {planet}:')
		for k, v in entry.items():
			print(f'    {k}: {v}')


def get_stellar_params(planet_name, star_name=None, cache_dir=None):
	"""
	Look up stellar parameters for a planet host.

	If cache_dir is provided, checks the local cache first and only queries
	the archive if parameters are missing or incomplete. Results are saved
	back to the cache after a successful query.

	To manually correct a cached value (e.g. missing [Fe/H]):
	  update_cache('55 Cnc e', {'feh': 0.35}, cache_dir)

	Query order (if cache miss):
	  1. astroquery.NasaExoplanetArchive (pscomppars, then ps)
	  2. NASA Exoplanet Archive TAP endpoint (urllib, with retries)
	  3. SIMBAD (last resort, limited parameter coverage)

	log g is derived from stellar mass + radius if not directly available.

	Parameters
	----------
	planet_name : str   e.g. '55 Cnc e', 'HD 189733 b'
	star_name   : str or None   host star name for SIMBAD fallback
	cache_dir   : str or None   directory for stellar_params_cache.json

	Returns
	-------
	dict with keys: teff, logg, feh, radius, mass, distance, source
	"""
	# Check cache first
	if cache_dir is not None:
		cache = _load_cache(cache_dir)
		if planet_name in cache:
			entry = cache[planet_name]
			if _cache_complete(entry):
				print(f'Using cached stellar parameters for "{planet_name}".')
				print(f'  (Edit {os.path.join(cache_dir, _CACHE_FILENAME)} to update manually)')
				return dict(entry)
			else:
				missing = [k for k in _CACHE_REQUIRED_KEYS if entry.get(k) is None]
				print(f'Cache entry for "{planet_name}" is incomplete (missing: {missing}). Re-querying...')

	print(f'Querying stellar parameters for "{planet_name}"...')

	params = _query_nasa_astroquery(planet_name)

	if params is None or params.get('teff') is None:
		print('  astroquery failed or returned no T_eff. Trying TAP...')
		params = _query_nasa_tap(planet_name)

	if params is None or params.get('teff') is None:
		fallback = star_name if star_name else planet_name
		print(f'  TAP failed. Falling back to SIMBAD for "{fallback}"...')
		params = _query_simbad(fallback)

	if params is None or params.get('teff') is None:
		raise RuntimeError(
			f'Could not retrieve stellar parameters for "{planet_name}". '
			'Check the planet/star name or provide parameters manually.'
		)

	# If radius or mass are still missing, do a dedicated TAP query with no
	# st_teff constraint — some rows have radius/mass but not teff.
	# We try pscomppars, ps, and stellarhosts in turn.
	if params.get('radius') is None or params.get('mass') is None:
		host_name = star_name if star_name else planet_name.rsplit(' ', 1)[0]
		print('  R_star or M_star missing. Running dedicated radius/mass query...')
		tab_queries = [
			('pscomppars',   f"pl_name='{planet_name}'",  'pl_name'),
			('ps',           f"pl_name='{planet_name}'",  'pl_name'),
			('stellarhosts', f"hostname='{host_name}'", 'hostname'),
		]
		for tab, where_expr, _ in tab_queries:
			if params.get('radius') is not None and params.get('mass') is not None:
				break
			try:
				import urllib.parse, json
				adql = (
					f"SELECT TOP 20 st_rad, st_mass, st_logg "
					f"FROM {tab} "
					f"WHERE {where_expr} "
					f"AND st_rad IS NOT NULL "
					f"ORDER BY st_rad DESC"
				)
				print(f'    {tab}...', end=' ', flush=True)
				# Ref: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
				tap_url2  = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync'
				tap_body2 = urllib.parse.urlencode({'query': adql, 'format': 'json'}).encode()
				req2 = urllib.request.Request(tap_url2, data=tap_body2, headers={'User-Agent': 'Mozilla/5.0'})
				with urllib.request.urlopen(req2, timeout=30) as resp:
					data = json.loads(resp.read().decode())
				if not data:
					print('no rows with st_rad.')
					continue
				# Pick median radius value to avoid outliers
				radii = [_safe_float(r['st_rad']) for r in data if _safe_float(r['st_rad']) is not None]
				masses = [_safe_float(r['st_mass']) for r in data if _safe_float(r['st_mass']) is not None]
				loggs  = [_safe_float(r['st_logg']) for r in data if _safe_float(r['st_logg']) is not None]
				if radii:
					import statistics
					rad = statistics.median(radii)
					if params.get('radius') is None:
						params['radius'] = rad
					if params.get('mass') is None and masses:
						params['mass'] = statistics.median(masses)
					if params.get('logg') is None and loggs:
						params['logg'] = statistics.median(loggs)
					print(f'ok. R_star = {rad:.3f} R_sun ({tab}, median of {len(radii)} values)')
				else:
					print('no valid st_rad values.')
			except Exception as e:
				print(f'failed: {e}')
				continue

	# Final fallback: derive radius from Stefan-Boltzmann if we have teff and luminosity
	# (not attempted here — user should provide manually if all else fails)
	if params.get('radius') is None:
		print(
			f'  Warning: R_star could not be retrieved automatically for "{planet_name}".\n'
			f'  Please set it manually in ratri_auto() via the returned stellar_params dict,\n'
			f'  or check the NASA Exoplanet Archive page for this star.'
		)

	# Fill defaults for parameters that are still missing after all queries
	if params.get('logg') is None:
		# Last-resort estimate: main-sequence log g from T_eff
		teff = params['teff']
		if teff >= 6000:
			default_logg = 4.3
		elif teff >= 5000:
			default_logg = 4.5
		else:
			default_logg = 4.7
		params['logg'] = default_logg
		print(
			f'  log g not found anywhere. Using T_eff-based estimate: {default_logg} '
			f'(override manually if known)'
		)
	if params.get('feh') is None:
		params['feh'] = 0.0
		print('  [Fe/H] not found, using default 0.0')

	print(f'\nStellar parameters ({params["source"]}):')
	print(f'  T_eff    : {params["teff"]:.0f} K')
	print(f'  log g    : {params["logg"]:.2f}')
	print(f'  [Fe/H]   : {params["feh"]:+.2f}')
	if params.get('radius'):
		print(f'  R_star   : {params["radius"]:.3f} R_sun')
	if params.get('mass'):
		print(f'  M_star   : {params["mass"]:.3f} M_sun')
	if params.get('distance'):
		print(f'  Distance : {params["distance"]:.2f} pc')

	# Save to cache
	if cache_dir is not None:
		cache = _load_cache(cache_dir)
		cache[planet_name] = {k: params.get(k) for k in _CACHE_REQUIRED_KEYS + ['source']}
		_save_cache(cache, cache_dir)
		print(f'  Cached to {os.path.join(cache_dir, _CACHE_FILENAME)}')
		if params.get('feh') is None or params['feh'] == 0.0:
			msg = (
				'  Note: [Fe/H] may be missing or defaulted. '
				f'Correct with: update_cache("{planet_name}", {{"feh": VALUE}}, cache_dir)'
			)
			print(msg)

	return params


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def _progress_hook(block_num, block_size, total_size):
	downloaded = block_num * block_size
	if total_size > 0:
		pct = min(downloaded / total_size * 100, 100)
		print(f'\r  {downloaded/1e6:.1f} MB / {total_size/1e6:.1f} MB ({pct:.1f}%)', end='')


def _download_with_retry(url, local_path, label, n_retries=3, wait_base=10):
	"""
	Download url to local_path with retry logic.

	Skips if the file already exists. Removes partial files on failure
	before retrying. Raises RuntimeError if all attempts fail.

	Parameters
	----------
	url        : str
	local_path : str
	label      : str   description for progress messages
	n_retries  : int   (default 3)
	wait_base  : int   seconds between attempts, multiplied by attempt index
	"""
	import time

	if os.path.exists(local_path):
		print(f'  Already cached: {os.path.basename(local_path)}')
		return

	for attempt in range(1, n_retries + 1):
		print(f'\nDownloading {label} (attempt {attempt}/{n_retries}):')
		print(f'  {url}')
		try:
			urllib.request.urlretrieve(url, local_path, reporthook=_progress_hook)
			print()
			print(f'  Saved to {local_path}')
			return
		except Exception as e:
			print(f'\n  Failed: {e}')
			if os.path.exists(local_path):
				os.remove(local_path)
			if attempt < n_retries:
				wait = wait_base * attempt
				print(f'  Waiting {wait}s before retry...')
				time.sleep(wait)

	raise RuntimeError(
		f'Failed to download {label} after {n_retries} attempts.\n'
		f'URL: {url}\n'
		f'Check your internet connection or try again later.'
	)


def _phoenix_feh_dir(feh):
	"""
	Return the directory name for a given [Fe/H] value on the PHOENIX server.

	The server always uses Z-X.X format (minus sign convention), even for
	solar and super-solar metallicities:
	  [Fe/H] =  0.0  ->  Z-0.0
	  [Fe/H] = +0.5  ->  Z+0.5
	  [Fe/H] = -0.5  ->  Z-0.5
	Solar metallicity is Z-0.0, not Z+0.0.
	"""
	if feh == 0.0:
		return 'Z-0.0'
	return f'Z{feh:+.1f}'


def _phoenix_feh_str(feh):
	"""
	Return the [Fe/H] string as it appears in PHOENIX filenames.
	Solar is written as -0.0 in filenames, matching the directory convention.
	"""
	if feh == 0.0:
		return '-0.0'
	return f'{feh:+.1f}'


def _download_phoenix(teff, logg, feh, cache_dir):
	"""Download PHOENIX/ACES spectrum and wavelength grid. Returns (spec_path, wave_path)."""
	# v2.0 is the current server path; the old path without v2.0 is stale
	# PHOENIX/ACES spectra — Husser et al. 2013, A&A 553, A6
	# https://phoenix.astro.physik.uni-goettingen.de
	BASE       = 'https://phoenix.astro.physik.uni-goettingen.de/data/v2.0/HiResFITS'
	feh_dir    = _phoenix_feh_dir(feh)
	feh_str    = _phoenix_feh_str(feh)
	filename   = f'lte{int(teff):05d}-{logg:.2f}{feh_str}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
	spec_url   = f'{BASE}/PHOENIX-ACES-AGSS-COND-2011/{feh_dir}/{filename}'
	spec_local = os.path.join(cache_dir, filename)
	wave_url   = f'{BASE}/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
	wave_local = os.path.join(cache_dir, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')

	_download_with_retry(spec_url, spec_local, f'PHOENIX spectrum ({filename})')
	_download_with_retry(wave_url, wave_local, 'PHOENIX wavelength grid')

	return spec_local, wave_local


def _download_btsettl(teff, logg, feh, cache_dir):
	"""Download BT-Settl spectrum. Returns (spec_path, wave_path)."""
	# PHOENIX/ACES spectra — Husser et al. 2013, A&A 553, A6
	# https://phoenix.astro.physik.uni-goettingen.de
	BASE       = 'https://phoenix.astro.physik.uni-goettingen.de/data/v2.0/HiResFITS'
	feh_dir    = _phoenix_feh_dir(feh)
	feh_str    = _phoenix_feh_str(feh)
	filename   = f'lte{int(teff):05d}-{logg:.2f}{feh_str}.BT-Settl.spec.fits'
	spec_url   = f'{BASE}/BT-Settl/{feh_dir}/{filename}'
	spec_local = os.path.join(cache_dir, filename)
	wave_url   = f'{BASE}/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
	wave_local = os.path.join(cache_dir, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')

	_download_with_retry(spec_url, spec_local, f'BT-Settl spectrum ({filename})')
	_download_with_retry(wave_url, wave_local, 'PHOENIX wavelength grid (shared with BT-Settl)')

	return spec_local, wave_local


def _download_atlas9(teff, logg, feh, cache_dir):
	"""Download ATLAS9/Kurucz spectrum. Returns (spec_path, None)."""
	BASE      = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models'
	feh_str   = f'{"p" if feh >= 0 else "m"}{abs(int(feh * 10)):02d}'
	filename  = f'ck{feh_str}_{int(teff)}.fits'
	spec_url  = f'{BASE}/ck{feh_str}/{filename}'
	spec_local = os.path.join(cache_dir, filename)

	_download_with_retry(spec_url, spec_local, f'ATLAS9 spectrum ({filename})')

	return spec_local, None


def _download_marcs(teff, logg, feh, cache_dir):
	"""Download MARCS spectrum. Returns (spec_path, None)."""
	BASE     = 'https://marcs.astro.uu.se/documents/auxiliary/fitsspectra'
	geom     = 's' if logg < 3.5 else 'p'
	feh_str  = f'{"p" if feh >= 0 else "m"}{abs(feh):.2f}'.replace('.', '')
	filename = (
		f'{geom}{int(teff):04d}_g{logg:.1f}_m0.00_t02_st_z{feh_str}'
		f'_a+00.00_c+00.00_n+00.00_o+00.00_r+00.00_s+00.00.PHXTEFF.fits'
	)
	spec_url  = f'{BASE}/{filename}'
	spec_local = os.path.join(cache_dir, filename)

	_download_with_retry(spec_url, spec_local, f'MARCS spectrum ({filename})')

	return spec_local, None


_DOWNLOADERS = {
	'phoenix': _download_phoenix,
	'btsettl': _download_btsettl,
	'atlas9':  _download_atlas9,
	'marcs':   _download_marcs,
}


# ---------------------------------------------------------------------------
# Main public interface
# ---------------------------------------------------------------------------

def download_spectrum(params, library='phoenix', cache_dir='.'):
	"""
	Snap stellar parameters to the nearest grid point for the chosen library
	and download the corresponding spectrum if not already cached.

	Parameters
	----------
	params    : dict   output of get_stellar_params()
	library   : str   'phoenix', 'btsettl', 'atlas9', or 'marcs'
	cache_dir : str   directory to store downloaded files

	Returns
	-------
	dict with keys: spec_path, wave_path, library, teff, logg, feh
	"""
	os.makedirs(cache_dir, exist_ok=True)

	snapped    = snap_to_grid(params['teff'], params['logg'], params['feh'], library)
	downloader = _DOWNLOADERS.get(library)
	if downloader is None:
		raise ValueError(
			f'Unknown library "{library}". '
			f'Choose from {list(_DOWNLOADERS.keys())}.'
		)

	spec_path, wave_path = downloader(
		snapped['teff'], snapped['logg'], snapped['feh'], cache_dir
	)

	return dict(
		spec_path = spec_path,
		wave_path = wave_path,
		library   = library,
		teff      = snapped['teff'],
		logg      = snapped['logg'],
		feh       = snapped['feh'],
	)


def get_spectrum_for_planet(planet_name, cache_dir,
                            star_name=None, library=None):
	"""
	One-shot convenience function: look up stellar parameters for a planet
	host, recommend a library, and download the spectrum.

	Stellar parameters are cached in stellar_params_cache.json inside
	cache_dir. To manually fix a missing value after the first run:
	  from stellar import update_cache
	  update_cache('55 Cnc e', {'feh': 0.35}, './spectra_cache')

	Parameters
	----------
	planet_name : str   e.g. '55 Cnc e'
	cache_dir   : str   directory to cache downloaded spectra and parameters
	star_name   : str or None   host star name for SIMBAD fallback
	library     : str or None   force a specific library; if None, auto-recommend

	Returns
	-------
	tuple: (stellar_params dict, spectrum_paths dict)
	"""
	params = get_stellar_params(planet_name, star_name=star_name, cache_dir=cache_dir)

	if library is None:
		library = recommend_library(params['teff'])
		print(f'\nUsing library: {library}')
		print('(Override by passing library="phoenix"|"btsettl"|"atlas9"|"marcs")')

	paths = download_spectrum(params, library=library, cache_dir=cache_dir)

	return params, paths


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
	stellar_params, spec_paths = get_spectrum_for_planet(
		planet_name = '55 Cnc e',
		cache_dir   = './spectra_cache',
		star_name   = '55 Cnc',
		library     = None,
	)

	print(f'\nSpectrum file : {spec_paths["spec_path"]}')
	if spec_paths['wave_path']:
		print(f'Wavelength file: {spec_paths["wave_path"]}')
