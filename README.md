# ai-pestriskmapping

Risk maps for the 10 most reported rice pests and diseases in the Philippines. The maps combine field pest reports, climate and vegetation from satellites, and the rice growth stage from the AI-CropGrowth pipeline.

**Pests covered:** brown planthopper, stem borer, rodents, bacterial leaf blight, leaffolder, rice blast, rice black bug, rice bug complex, bacterial leaf streak, green leafhopper.

## How it works

```
pest reports (2015-2025) ──► clean + one point per pest/place/month
                                      │
climate, vegetation, terrain ──► MaxEnt (Earth Engine) ──► suitability, 1 km, per pest x quarter
                                      │                                   │
reported growth stage ──► pest x stage weights                            │
                                      │                                   │
AI-CropGrowth stage map (30 m, per period) ───────────────────────────────┤
                                                                          ▼
                     risk = suitability(1 km, quarter) x weight[pest, stage of the 30 m pixel]
```

1. **Environmental suitability.** For each pest and quarter, a MaxEnt model compares report locations with background points. Background points are random cropland plus places surveyed for other pests in the same period, which accounts for where surveys happen. Predictors:
   - CHIRPS rainfall: 30 and 60 days before the month, and the month itself
   - MODIS NDVI and EVI
   - TerraClimate: precipitation, max/min temperature, soil moisture, VPD, AET, PET
   - SRTM elevation and slope

   Training years are 2021–2024. The output is relative suitability from 0 to 1, not a probability.
2. **Stage weights.** The growth-stage text of each report is read again, including `45 DAT` / `30 DAS` style numbers. The weight compares how often a pest is reported at a stage with how often all pests are. It is blended with an expert starting table; pests with few reports lean on the starting table.
3. **Risk for each stage period.** Each 30 m rice pixel gets suitability × the weight for its current stage. The results are summed up to 1 km cells and to provinces.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/PEST_2026_RISK_MAPPING_STAGES.ipynb` | Colab notebook: data cleaning, Earth Engine models, stage weights, operational run |
| `pestrisk/stage.py` | Combines the growth-stage product with the suitability maps (numpy, pandas, rasterio; no Earth Engine) |
| `tests/` | Tests for `pestrisk/stage.py` on small synthetic rasters |

### Notebook cells

| Cell | What it does | When to run |
|---|---|---|
| 0 | Mounts Drive. Cleans `Pest/pests_2015_2025 new.csv`: parses dates and degree/minute/second coordinates, standardises pest names, keeps rice and the top 10 pests, one point per year/month/place/pest | Model update |
| 1 | Earth Engine setup, predictors, background sampling, MaxEnt helper | Model update |
| 2 | Trains 10 pests × 4 quarters and shows them in a map viewer | Model update |
| 3 | Exports the suitability maps to Drive (`PestRiskGEE_Rice_YearAgnostic_Quarterly/`) | Model update |
| 4 | Introduction to the stage cells | — |
| 5 | Stage of each report, and the pest × stage weight table (`rice_top10_pest_stage_weights.csv`) | Model update |
| 6–9 | *Optional, historical.* Stage from MODIS flooding (LSWI + 0.05 ≥ EVI), stage as a MaxEnt input with an AUC comparison, monthly maps, export | Testing only |
| 10 | **Operational:** risk for every new growth-stage period | After every growth-stage run |

## Growth-stage input (AI-CropGrowth)

The AI-CropGrowth pipeline writes to `AI-CropGrowth/outputs/products/growth_stage/auto/` on Drive:

```
rice_<PROVINCE>_<YYYYMM>.tiff          30 m (HLS), EPSG:4326, int16, one per province   <- used here
mosaic/rice_PHILIPPINES_<YYYYMM>.tiff  national mosaic
summary/rice_<PROVINCE>_<YYYYMM>.csv   hectares per class
runs/periodic_<timestamp>.csv          status of each province in a run
```

| Value | Class | Weight used |
|---|---|---|
| -1 | outside the rice mask | nodata |
| 0 | no rice cycle detected | off-season |
| 1 | pre-planting / transplanting | establishment |
| 2 | vegetative | vegetative |
| 3 | reproductive | reproductive |
| 4 | ripening / maturity | ripening |
| 5 | harvested / post-harvest | off-season |

- **Rice area:** every pixel ≥ 0 is inside the rice mask, so the product also gives the rice area.
- **Off-season weight:** 0 for most pests, 0.3 for Rodents.
- **Class check:** a file whose `classes` GeoTIFF tag differs from this table is refused, so a change in the coding can't slip through unnoticed.

## Outputs (cell 10)

`Pest/pest_risk_stage/<YYYYMM>/`:

| File | Contents |
|---|---|
| `rice_pest_risk_<pest>_<YYYYMM>_1km.tif` | Bands: `risk_standing_crop`, `risk_all_rice`, `suitability_env`, `stage_weight_all_rice`, `standing_crop_ha`, `rice_ha` |
| `rice_stage_fractions_<YYYYMM>_1km.tif` | Share of rice area in each class, `rice_ha`, `standing_crop_ha` |
| `rice_pest_risk_by_province_<YYYYMM>.csv` | Per province and pest: rice and standing-crop hectares, mean risk, hectares per risk class and per stage |
| `30m/rice_pest_risk_<pest>_<province>_<YYYYMM>_30m.tif` | Optional (`WRITE_30M`): risk × 100, uint8, nodata 255 |
| `manifest.json` | Input files, suitability maps and weights used |

- **Risk bands:**
  - `risk_standing_crop` is the hectare-weighted mean over pixels with a crop in the field (classes 1–4). Use it for advisories.
  - `risk_all_rice` also counts fallow and harvested pixels, at their off-season weight.

  Without this split, "no crop in the field" would look like "low risk".
- **Risk classes:** low < 0.25 ≤ moderate < 0.5 ≤ high < 0.75 ≤ very high.
- **Resuming:** periods that already have a `manifest.json` are skipped. A period is redone when its province files change, for example when a province that failed is produced later.

## Setup (Colab)

1. Copy this repository to `MyDrive/ai-pestriskmapping`.
2. The growth-stage `auto` folder is shared from another account. Add a shortcut to it in My Drive (right-click > Organize > Add shortcut). Cell 10 expects `MyDrive/auto`; change `STAGE_DIR` otherwise.
3. Put the pest report CSV at `MyDrive/Pest/pests_2015_2025 new.csv`.
4. Earth Engine project: `sensesaka2024` (cells 1 and 3).
5. **Model update** (yearly, or when new reports arrive): run cells 0, 1, 2, 3, 5. Wait for the Drive exports to finish.
6. **Each growth-stage run:** run cell 0 (Drive and paths), then cell 10.

Local tests:

```bash
pip install -r requirements.txt pytest
```

```bash
python -m pytest -q
```

## Known limitations

- **Missing provinces.** In the 24 Nov 2026 growth-stage run, 12 provinces failed with `KeyError('trough')`: Antique, Basilan, Bohol, Camarines Sur, Davao Occidental, Davao Oriental, Iloilo, Palawan, Sulu, Surigao del Norte, Tawi-Tawi and Zamboanga del Sur. They have no risk until they are produced. Cell 10 lists missing provinces each run.
- **Stage is not yet a model input.** The stage product starts in 2026, but the models train on 2021–2024. Stage enters only through the weight table until past periods are produced. After that, `pestrisk.stage.sample_stage_at_points` gives the product's stage at each report, so the weights can come from satellite stages and stage can become a model input.
- **Cropland mask.** The suitability maps are clipped to the ESA WorldCover cropland mask, so some rice pixels have no suitability. The province table reports these hectares in `ha_no_suitability`. Exporting the suitability without that mask fixes it.
- **Expert weight table.** `STAGE_PRIOR` in cell 5 is an approximate starting point from general rice IPM knowledge. Review it with entomologists and pathologists before operational use.
- **Model evaluation.** The validation split is random, so nearby points fall in both sets and AUCs are optimistic. Cell 2 does not report AUC; cell 7 compares the env-only and env+stage models on the same samples.
- **Mixed 1 km cells.** 1 km cells mix fields at different stages; in sampled provinces the main stage covers about 70% of a cell's rice. Use the stage fractions or the 30 m maps for field-level detail.
