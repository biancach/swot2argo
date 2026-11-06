# swot2argo data

The inputs of *Reconstructing 3D upper-ocean density from SWOT's wide-swath sea surface height using machine
learning*. Compressed NumPy archives (`np.load(path, allow_pickle=True)`); every density profile is potential
density (kg/m3) on the same 40 levels, 0 to 195 m every 5 m (`depth`, in every file). Density anomalies in the paper
are `dens - clim`.

Common columns of the profile files:

| column | content |
|---|---|
| `time` | profile time (UTC) |
| `lat`, `lon` | profile position (degrees) |
| `dens` | observed potential density, (n, 40) |
| `clim` | ECCO V4r4 climatology at the profile's position and date, (n, 40) |
| `ssh_swot` | SWOT L3 v3.0 unfiltered SSHA at the nearest pixel, internal tide restored, calibrated against DUACS (m) |
| `ssh_duacs` | DUACS L4 SSHA at the profile (m) |
| `sst` | OSTIA sea surface temperature (degC) |
| `mdt` | mean dynamic topography (m) |
| `sss`, `bathy`, `era5_*` | inputs tried in the selection and not kept (SI Text S3, Table S8): satellite sea surface salinity (SMOS/SMAP optimal interpolation, NaN before June 2010 and after December 2025), the ECCO bottom depth (m), and ERA5 surface buoyancy flux `era5_b0_<d>` and friction velocity cubed `era5_ustar3_<d>` averaged over the d = 7, 30 and 60 days before the profile |

## Files

| file | content | rows |
|---|---|---|
| `pairs.npz` | the training set: Argo profiles within 24 h and 3 km of a SWOT overpass, July 2023 to September 2026. Also `wmo` (float), `cycle`, `dt_h` (profile time minus overpass time, hours) | 424 |
| `history.npz` | the pretraining set: Argo profiles of the region before 2023 with DUACS SSHA (no SWOT), from floats that have no training pair. Also `wmo`, `cycle` | 14,448 |
| `tests.npz` | the zero-shot test sets, columns prefixed `moorings_`, `gliders_`, `s2_`. Moorings: the 11 Cal/Val moorings interpolated to each SWOT pass time, April to July 2023 (`station`, `pass_num`). Gliders: the two gliders of the campaign (`glider`; `dt_h`, overpass time minus profile time, hours). S2: the mooring record at the S2 site on the SWOT passes of October 2023 to October 2024 (the NOPP-GIW mooring there; called S2 in the paper) | 1,546 / 784 / 33 |
| `benchmarks.npz` | the gridded products at the test samples, `<test>_<product>` (n, 40): GLORYS12, GLO12, HYCOM, ARMOR3D and the PWP column model; NaN where a product does not cover the sample | as the tests |
| `geo.npz` | the maps: `stations`, `station_lat`, `station_lon`; the coastline (`coast`); the SWOT swath edges (`swath_<pass>_lat/lon`); the region box | |
| `s2_hourly.npz` | the hourly record at the S2 site, March 2023 to October 2024 (`time`, `dens`): the Cal/Val S2 mooring until September 2023, then the NOPP-GIW mooring at the site | 14,270 |
| `moorings_hourly.npz` | the hourly record of each Cal/Val mooring, 20 March to 31 July 2023: `stations`, and `<station>_time`, `<station>_dens` | 1,010 to 3,216 per mooring |
| `swath_crossovers.csv` | SWOT minus DUACS SSHA where two SWOT passes less than 2 days apart cross, July 2023 to September 2026: the two passes (`cycle_a`, `pass_a`, `cycle_b`, `pass_b`), the crossing (`lat`, `lon`, mid `time`, `gap_days`), `var_cm2` the variance of SWOT - DUACS on the crossing averaged over the two passes and `cov_cm2` its covariance between them, both summed over scales of 2 to 240 km (cm2). Computed from the full SWOT L3 and DUACS archives, which are too large to include | 819 |
| `swaths/swath_<YYYYMMDD_HH>.npz` | one SWOT overpass on its 2 km grid (n_lines, n_pixels): `lat`, `lon`, `ssh_swot` (calibrated, outliers removed), `ssh_raw` (calibrated with the plane `tilt` = offset, across-track and along-track slope not yet subtracted), `ssh_duacs` at the pixels, `time`, `pass_num`. The array overpasses also carry `sst`, `mdt` and the `clim` of the array (40,), so the network can reconstruct every pixel; the 8 May 2023 one carries GLORYS12 on the swath (`glorys12`, (n_lines, n_pixels, 40), with its `glorys12_time`) | |

The overpasses: 8 May 2023 (Figures 1b and 4), 30 May and 5 July 2023 and, at S2, 18 March and 2 October 2024
(Figure 2).

## Sources

Argo (https://doi.org/10.17882/42182); the Cal/Val moorings and gliders
(https://github.com/jinbow/swot-mission-validation-sub100km, https://doi.org/10.5281/zenodo.14675995) and the mooring
files, including the NOPP-GIW record at the S2 site from September 2023 to October 2024 (SWOT Postlaunch Oceanography
Field Campaign L2 Moorings, PO.DAAC, https://doi.org/10.5067/SWTPO-MOOR2); SWOT L3 v3.0 Expert (AVISO/DUACS,
https://doi.org/10.24400/527896/A01-2023.018); DUACS L4 and OSTIA SST (Copernicus Marine Service); ECCO V4r4
(https://podaac.jpl.nasa.gov/ECCO); GLORYS12, GLO12 and ARMOR3D (Copernicus Marine Service); HYCOM GOFS 3.1
(https://tds.hycom.org); sea surface salinity MULTIOBS_GLO_PHY_SSS_L4_MY_015_015 (Copernicus Marine Service);
ERA5 (Copernicus Climate Change Service). Densities are computed from temperature and salinity with TEOS-10
(GSW-Python). SI Table S1 lists every product with its identifier and period.

## License and credit

The processed data in this folder are released under the Creative Commons Attribution 4.0 license (CC BY 4.0). They are
derived from the products above, which keep their own terms; please also credit them when you reuse the data:

- Argo: these data were collected and made freely available by the International Argo Program and the national
  programs that contribute to it (https://argo.ucsd.edu, https://www.ocean-ops.org). The Argo Program is part of the
  Global Ocean Observing System. https://doi.org/10.17882/42182
- Moorings and gliders: SWOT Ocean CalVal Team (2025), SWOT Postlaunch Oceanography Field Campaign L2 Moorings, NASA
  PO.DAAC, https://doi.org/10.5067/SWTPO-MOOR2; Wang (2025), https://doi.org/10.5281/zenodo.14675995. The NOPP-GIW
  mooring was supported by the U.S. Office of Naval Research (awards N000142212215 and N000142212575).
- SWOT L3: AVISO/DUACS (2024), SWOT Level-3 KaRIn Low Rate SSH Expert, v3.0 (SSALTO/DUACS; CLS, CNES),
  https://doi.org/10.24400/527896/A01-2023.018, licensed by CNES under the AVISO License Agreement
  (https://www.aviso.altimetry.fr/fileadmin/documents/data/License_Aviso.pdf). Modified here: sampled at the profiles
  and cut out around the sites, internal tide restored, calibrated against DUACS and outliers removed (`ssh_swot`,
  `ssh_raw` and the swath files above).
- DUACS L4, OSTIA, sea surface salinity, GLORYS12, GLO12, ARMOR3D: generated using E.U. Copernicus Marine Service
  Information (product identifiers and DOIs in SI Table S1).
- ERA5: contains modified Copernicus Climate Change Service information (2026).
- ECCO V4r4 (https://podaac.jpl.nasa.gov/ECCO) and HYCOM GOFS 3.1 (https://tds.hycom.org).
