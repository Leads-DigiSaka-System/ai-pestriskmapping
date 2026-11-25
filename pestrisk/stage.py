"""Rice pest risk from the AI-CropGrowth growth-stage product.

The growth-stage pipeline writes, for each period ``YYYYMM``::

    <stage_dir>/rice_<PROVINCE>_<YYYYMM>.tiff          one per province, 30 m, EPSG:4326, int16
    <stage_dir>/mosaic/rice_PHILIPPINES_<YYYYMM>.tiff  national mosaic (not needed here)
    <stage_dir>/summary/rice_<PROVINCE>_<YYYYMM>.csv   hectares per class

Pixel values (GeoTIFF tag ``classes``)::

    -1 outside the province / rice mask (nodata)
     0 no rice cycle detected        3 reproductive
     1 pre-planting / transplanting  4 ripening / maturity
     2 vegetative                    5 harvested / post-harvest

Every pixel >= 0 is inside the rice mask, so the product also gives the rice area.

This module combines it with the quarterly 1 km MaxEnt suitability maps of the pest notebook::

    risk(pixel) = suitability(1 km cell, quarter of the period) x weight[pest, stage of the pixel]

and writes, per period and pest, a 1 km GeoTIFF (rice-area weighted mean risk), a 1 km stage
fraction GeoTIFF, and a province table of rice hectares per risk class. Optional 30 m risk
GeoTIFFs per province.

Only numpy, pandas and rasterio are needed (no Earth Engine).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Product definition
# ------------------------------------------------------------------
NODATA = -1
CLASS_NAMES = {
    0: "no rice cycle detected",
    1: "pre-planting / transplanting",
    2: "vegetative",
    3: "reproductive",
    4: "ripening / maturity",
    5: "harvested / post-harvest",
}
N_CLASSES = len(CLASS_NAMES)
EXPECTED_CLASSES_TAG = "; ".join(f"{k}={v}" for k, v in CLASS_NAMES.items())

# Stage columns of the notebook's weight table (rice_top10_pest_stage_weights.csv).
WEIGHT_STAGES = ["establishment", "vegetative", "reproductive", "ripening"]
# Product class -> weight-table stage. Classes 0 and 5 (no crop in the field) use the off-season weight.
CLASS_TO_STAGE = {1: "establishment", 2: "vegetative", 3: "reproductive", 4: "ripening"}
STANDING_CLASSES = sorted(CLASS_TO_STAGE)   # crop in the field

QUARTER_OF_MONTH = {m: f"Q{(m - 1) // 3 + 1}" for m in range(1, 13)}

RISK_BREAKS = (0.25, 0.5, 0.75)
RISK_CLASS_NAMES = ("low", "moderate", "high", "very_high")

_FILE_RE = re.compile(r"^rice_(?P<area>.+)_(?P<period>\d{6})\.tiff?$", re.I)

M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQUATOR = 111320.0


# ------------------------------------------------------------------
# Finding products
# ------------------------------------------------------------------
def list_stage_products(stage_dir: str) -> pd.DataFrame:
    """Province GeoTIFFs in ``stage_dir`` (not sub-folders): area, period, path, size, mtime."""
    rows = []
    for name in sorted(os.listdir(stage_dir)):
        m = _FILE_RE.match(name)
        path = os.path.join(stage_dir, name)
        if not m or not os.path.isfile(path):
            continue
        area = m.group("area")
        if area.upper() == "PHILIPPINES":
            continue
        st = os.stat(path)
        rows.append({"area": area, "period": m.group("period"), "path": path,
                     "size": st.st_size, "mtime": int(st.st_mtime)})
    return pd.DataFrame(rows, columns=["area", "period", "path", "size", "mtime"])


def period_month(period: str) -> int:
    month = int(period[4:6])
    if not 1 <= month <= 12:
        raise ValueError(f"Bad period {period!r}; expected YYYYMM")
    return month


def read_product_info(path: str) -> dict:
    """GeoTIFF tags of a stage product. Raises if the class coding is not the expected one."""
    import rasterio

    with rasterio.open(path) as src:
        tags = src.tags()
        info = {
            "crs": src.crs.to_string() if src.crs else None,
            "res": src.res,
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
            **{k: tags.get(k) for k in ("period", "period_dates", "window", "method",
                                         "data_source", "province", "classes")},
        }
    if info["classes"] is not None and _norm(info["classes"]) != _norm(EXPECTED_CLASSES_TAG):
        raise ValueError(
            f"{os.path.basename(path)}: class coding changed.\n  found:    {info['classes']}\n"
            f"  expected: {EXPECTED_CLASSES_TAG}\nUpdate CLASS_NAMES / CLASS_TO_STAGE first.")
    return info


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# ------------------------------------------------------------------
# Weights
# ------------------------------------------------------------------
def load_stage_weights(path: str) -> pd.DataFrame:
    """Weight table written by the notebook (index = pest group, columns = WEIGHT_STAGES)."""
    w = pd.read_csv(path, index_col=0)
    missing = [s for s in WEIGHT_STAGES if s not in w.columns]
    if missing:
        raise ValueError(f"{path}: missing stage columns {missing}")
    return w[WEIGHT_STAGES].astype(float)


def class_weights(weights: pd.DataFrame, pest: str, offseason: float = 0.0) -> np.ndarray:
    """Weight for each product class 0..5 for one pest."""
    vec = np.full(N_CLASSES, float(offseason), dtype=np.float32)
    for code, stage in CLASS_TO_STAGE.items():
        vec[code] = float(weights.loc[pest, stage])
    return vec


# ------------------------------------------------------------------
# Target grid (the 1 km suitability grid)
# ------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    x0: float     # west edge
    y0: float     # north edge
    dx: float     # > 0
    dy: float     # > 0 (rows go south)
    width: int
    height: int
    crs: str = "EPSG:4326"

    @classmethod
    def from_raster(cls, path: str) -> "Grid":
        import rasterio

        with rasterio.open(path) as src:
            t = src.transform
            if t.b != 0 or t.d != 0:
                raise ValueError(f"{path}: rotated grids are not supported")
            return cls(t.c, t.f, t.a, -t.e, src.width, src.height,
                       src.crs.to_string() if src.crs else "EPSG:4326")

    @property
    def transform(self):
        from affine import Affine

        return Affine(self.dx, 0.0, self.x0, 0.0, -self.dy, self.y0)

    def cell_index(self, lon: np.ndarray, lat: np.ndarray):
        """Row, col of the cell holding each point, and a mask of points inside the grid."""
        col = np.floor((lon - self.x0) / self.dx).astype(np.int64)
        row = np.floor((self.y0 - lat) / self.dy).astype(np.int64)
        inside = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        return row, col, inside


def read_grid_array(path: str, grid: Grid | None = None) -> np.ndarray:
    """Band 1 as float32 with nodata -> NaN. Checks it is on ``grid`` when one is given."""
    import rasterio

    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            a[a == src.nodata] = np.nan
        if grid is not None:
            g = Grid.from_raster(path)
            if (g.width, g.height) != (grid.width, grid.height) or not np.allclose(
                    [g.x0, g.y0, g.dx, g.dy], [grid.x0, grid.y0, grid.dx, grid.dy], rtol=0, atol=1e-9):
                raise ValueError(f"{path} is not on the reference grid")
    return a


# ------------------------------------------------------------------
# Core: 30 m stage pixels -> 1 km counts and hectares at risk
# ------------------------------------------------------------------
def _pixel_area_ha(lat: np.ndarray, dlon: float, dlat: float) -> np.ndarray:
    return (dlon * M_PER_DEG_LON_EQUATOR * np.cos(np.radians(lat))) * (dlat * M_PER_DEG_LAT) / 1e4


def _iter_chunks(path: str, chunk_rows: int):
    """Yield (stage int array, lon of columns, lat of rows) in row chunks."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        t = src.transform
        if t.b != 0 or t.d != 0:
            raise ValueError(f"{path}: rotated grids are not supported")
        if src.crs is not None and src.crs.to_epsg() != 4326:
            raise ValueError(f"{path}: expected EPSG:4326, found {src.crs}")
        nodata = src.nodata if src.nodata is not None else NODATA
        lon = t.c + (np.arange(src.width) + 0.5) * t.a
        for r0 in range(0, src.height, chunk_rows):
            n = min(chunk_rows, src.height - r0)
            a = src.read(1, window=Window(0, r0, src.width, n))
            a = np.where(a == nodata, NODATA, a).astype(np.int16)
            lat = t.f + (np.arange(r0, r0 + n) + 0.5) * t.e
            yield a, lon, lat, (abs(t.a), abs(t.e))


@dataclass
class PeriodAccumulator:
    """Running totals for one period over all province files."""
    grid: Grid
    pests: list
    area_by_class: np.ndarray = None   # (N_CLASSES, H, W) hectares per class
    risk_sum: dict = None              # pest -> (H, W) sum of risk x hectares, all rice pixels
    risk_sum_standing: dict = None     # pest -> (H, W) same, standing crop only (classes 1-4)
    province_rows: list = field(default_factory=list)

    def __post_init__(self):
        h, w = self.grid.height, self.grid.width
        self.area_by_class = np.zeros((N_CLASSES, h, w), dtype=np.float64)
        self.risk_sum = {p: np.zeros((h, w), dtype=np.float64) for p in self.pests}
        self.risk_sum_standing = {p: np.zeros((h, w), dtype=np.float64) for p in self.pests}

    @property
    def area_ha(self) -> np.ndarray:
        return self.area_by_class.sum(axis=0)

    @property
    def standing_ha(self) -> np.ndarray:
        return self.area_by_class[STANDING_CLASSES].sum(axis=0)


def accumulate_province(acc: PeriodAccumulator, path: str, area: str, suitability: dict,
                        weights: dict, chunk_rows: int = 2048, write_30m_dir: str | None = None,
                        period: str = "") -> None:
    """Adds one province file to ``acc``.

    suitability: pest -> (H, W) float32 on acc.grid (NaN = no model value)
    weights:     pest -> length-6 class weight vector
    """
    grid = acc.grid
    per_pest = {p: {"ha_by_class": np.zeros(len(RISK_CLASS_NAMES)), "risk_ha": 0.0,
                    "risk_ha_standing": 0.0, "standing_scored_ha": 0.0, "ha_no_suit": 0.0}
                for p in acc.pests}
    ha_stage = np.zeros(N_CLASSES)

    writers = {}
    if write_30m_dir:
        writers = _open_30m_writers(path, area, period, acc.pests, write_30m_dir)

    try:
        row_offset = 0
        for a, lon, lat, (dlon, dlat) in _iter_chunks(path, chunk_rows):
            valid = (a >= 0) & (a < N_CLASSES)
            risk_chunks = {p: None for p in writers}
            if valid.any():
                rr, cc = np.nonzero(valid)
                cls = a[rr, cc].astype(np.int64)
                prow, pcol, inside = grid.cell_index(lon[cc], lat[rr])
                px_ha = _pixel_area_ha(lat[rr], dlon, dlat)

                ha_stage += np.bincount(cls, weights=px_ha, minlength=N_CLASSES)

                sel = inside
                cls_in = cls[sel]
                ha_in = px_ha[sel]
                prow_in, pcol_in = prow[sel], pcol[sel]
                # Bin into the small block of 1 km cells this chunk touches, then add it in.
                if prow_in.size:
                    r0, r1 = int(prow_in.min()), int(prow_in.max()) + 1
                    c0, c1 = int(pcol_in.min()), int(pcol_in.max()) + 1
                else:
                    r0 = r1 = c0 = c1 = 0
                bh, bw = r1 - r0, c1 - c0
                local = (prow_in - r0) * bw + (pcol_in - c0)
                block = (slice(r0, r1), slice(c0, c1))
                standing = np.isin(cls_in, STANDING_CLASSES)
                if bh and bw:
                    acc.area_by_class[:, block[0], block[1]] += np.bincount(
                        cls_in * (bh * bw) + local, weights=ha_in,
                        minlength=N_CLASSES * bh * bw).reshape(N_CLASSES, bh, bw)

                for p in acc.pests:
                    s = suitability[p][prow_in, pcol_in]
                    r = s * weights[p][cls_in]
                    ok = ~np.isnan(r)
                    oks = ok & standing
                    if bh and bw:
                        acc.risk_sum[p][block] += np.bincount(
                            local[ok], weights=r[ok] * ha_in[ok], minlength=bh * bw).reshape(bh, bw)
                        acc.risk_sum_standing[p][block] += np.bincount(
                            local[oks], weights=r[oks] * ha_in[oks], minlength=bh * bw).reshape(bh, bw)
                    stats = per_pest[p]
                    stats["risk_ha"] += float((r[ok] * ha_in[ok]).sum())
                    stats["risk_ha_standing"] += float((r[oks] * ha_in[oks]).sum())
                    stats["standing_scored_ha"] += float(ha_in[oks].sum())
                    stats["ha_no_suit"] += float(ha_in[~ok].sum()) + float(px_ha[~sel].sum())
                    k = np.digitize(r[ok], RISK_BREAKS)
                    stats["ha_by_class"] += np.bincount(k, weights=ha_in[ok], minlength=len(RISK_CLASS_NAMES))

                    if p in writers:
                        out = np.full(a.shape, 255, dtype=np.uint8)
                        rr_in, cc_in = rr[sel][ok], cc[sel][ok]
                        out[rr_in, cc_in] = np.clip(np.round(r[ok] * 100), 0, 100).astype(np.uint8)
                        risk_chunks[p] = out
            for p, wr in writers.items():
                from rasterio.windows import Window

                out = risk_chunks[p] if risk_chunks[p] is not None else np.full(a.shape, 255, np.uint8)
                wr.write(out, 1, window=Window(0, row_offset, a.shape[1], a.shape[0]))
            row_offset += a.shape[0]
    finally:
        for wr in writers.values():
            wr.close()

    rice_ha = float(ha_stage.sum())
    standing_ha = float(ha_stage[STANDING_CLASSES].sum())
    for p in acc.pests:
        stats = per_pest[p]
        scored_ha = float(stats["ha_by_class"].sum())
        row = {
            "period": period, "province": area.replace("_", " "), "pest_group": p,
            "rice_ha": round(rice_ha, 1),
            "standing_crop_ha": round(standing_ha, 1),
            "ha_no_suitability": round(stats["ha_no_suit"], 1),
            "mean_risk_standing_crop": (round(stats["risk_ha_standing"] / stats["standing_scored_ha"], 4)
                                        if stats["standing_scored_ha"] > 0 else np.nan),
            "mean_risk_all_rice": round(stats["risk_ha"] / scored_ha, 4) if scored_ha > 0 else np.nan,
            "risk_weighted_ha": round(stats["risk_ha"], 1),
        }
        for name, ha in zip(RISK_CLASS_NAMES, stats["ha_by_class"]):
            row[f"ha_{name}"] = round(float(ha), 1)
        for code, ha in enumerate(ha_stage):
            row[f"ha_stage_{code}"] = round(float(ha), 1)
        acc.province_rows.append(row)


def _open_30m_writers(path, area, period, pests, out_dir):
    import rasterio

    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(path) as src:
        profile = src.profile.copy()
    profile.update(dtype="uint8", nodata=255, count=1, compress="deflate", tiled=True,
                   blockxsize=512, blockysize=512)
    writers = {}
    for p in pests:
        name = f"rice_pest_risk_{slug(p)}_{area.lower()}_{period}_30m.tif"
        wr = rasterio.open(os.path.join(out_dir, name), "w", **profile)
        wr.update_tags(1, units="risk x 100", pest_group=p, period=period, province=area)
        writers[p] = wr
    return writers


def slug(x: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(x).lower())).strip("_")


# ------------------------------------------------------------------
# Writing 1 km outputs
# ------------------------------------------------------------------
def write_grid_tif(path: str, grid: Grid, bands: dict, nodata=np.nan, tags: dict | None = None) -> None:
    import rasterio

    arrays = list(bands.values())
    with rasterio.open(
        path, "w", driver="GTiff", width=grid.width, height=grid.height, count=len(arrays),
        dtype="float32", crs=grid.crs, transform=grid.transform, nodata=nodata,
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        for i, (name, arr) in enumerate(bands.items(), start=1):
            dst.write(arr.astype(np.float32), i)
            dst.set_band_description(i, name)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


def finish_period(acc: PeriodAccumulator, suitability: dict, weights: dict, out_dir: str,
                  period: str, tags: dict) -> dict:
    """Writes the 1 km stage fractions, per-pest risk maps and the province table."""
    os.makedirs(out_dir, exist_ok=True)
    area = acc.area_ha
    standing = acc.standing_ha
    has_rice = area > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(has_rice, acc.area_by_class / np.where(has_rice, area, 1), np.nan)

    stage_bands = {f"frac_{k}_{slug(v)}": frac[k] for k, v in CLASS_NAMES.items()}
    stage_bands["rice_ha"] = np.where(has_rice, area, np.nan)
    stage_bands["standing_crop_ha"] = np.where(has_rice, standing, np.nan)
    stage_path = os.path.join(out_dir, f"rice_stage_fractions_{period}_1km.tif")
    write_grid_tif(stage_path, acc.grid, stage_bands, tags=tags)
    written = {"stage_fractions": stage_path, "risk": {}}

    for p in acc.pests:
        has_suit = ~np.isnan(suitability[p])
        with np.errstate(invalid="ignore", divide="ignore"):
            weight_mean = np.where(has_rice, np.tensordot(weights[p], frac, axes=1), np.nan)
            risk_standing = np.where((standing > 0) & has_suit,
                                     acc.risk_sum_standing[p] / np.where(standing > 0, standing, 1), np.nan)
            risk_all = np.where(has_rice & has_suit, acc.risk_sum[p] / np.where(has_rice, area, 1), np.nan)
        path = os.path.join(out_dir, f"rice_pest_risk_{slug(p)}_{period}_1km.tif")
        write_grid_tif(path, acc.grid, {
            "risk_standing_crop": risk_standing,
            "risk_all_rice": risk_all,
            "suitability_env": np.where(has_rice, suitability[p], np.nan),
            "stage_weight_all_rice": weight_mean,
            "standing_crop_ha": np.where(has_rice, standing, np.nan),
            "rice_ha": np.where(has_rice, area, np.nan),
        }, tags={**tags, "pest_group": p,
                 "class_weights": ",".join(f"{v:.3f}" for v in weights[p]),
                 "formula": "risk(pixel) = suitability(1 km cell) x weight[pest, stage]; bands are "
                            "hectare-weighted means over standing crop (classes 1-4) or all rice pixels"})
        written["risk"][p] = path

    table = pd.DataFrame(acc.province_rows)
    table_path = os.path.join(out_dir, f"rice_pest_risk_by_province_{period}.csv")
    table.to_csv(table_path, index=False)
    written["province_table"] = table_path
    return written


# ------------------------------------------------------------------
# Suitability inputs (exported by the notebook, cell 3)
# ------------------------------------------------------------------
def find_suitability(suit_dir: str, pest: str, quarter: str) -> str | None:
    """rice_maxent_occurrence_suitability_<pest>_<years>_<qN>_ph_cropland_1km(.tif)"""
    pat = re.compile(rf"^rice_maxent_occurrence_suitability_{re.escape(slug(pest))}_\d{{4}}_\d{{4}}_"
                     rf"{quarter.lower()}_.*\.tif$")
    hits = sorted(n for n in os.listdir(suit_dir) if pat.match(n))
    return os.path.join(suit_dir, hits[-1]) if hits else None


# ------------------------------------------------------------------
# One period, and all pending periods
# ------------------------------------------------------------------
def run_period(period: str, products: pd.DataFrame, suit_dir: str, weights_table: pd.DataFrame,
               out_dir: str, offseason: dict | None = None, pests: list | None = None,
               write_30m: bool = False, chunk_rows: int = 2048, log=print) -> dict:
    files = products[products["period"] == period].sort_values("area")
    if files.empty:
        raise ValueError(f"No stage products for period {period}")
    quarter = QUARTER_OF_MONTH[period_month(period)]
    pests = list(pests or weights_table.index)
    offseason = offseason or {}

    suit_paths, missing = {}, []
    for p in pests:
        sp = find_suitability(suit_dir, p, quarter)
        if sp:
            suit_paths[p] = sp
        else:
            missing.append(p)
    if missing:
        log(f"  no {quarter} suitability map for: {', '.join(missing)} (skipped)")
    pests = [p for p in pests if p in suit_paths]
    if not pests:
        raise ValueError(f"No suitability maps for {quarter} in {suit_dir}")

    grid = Grid.from_raster(next(iter(suit_paths.values())))
    suitability = {p: read_grid_array(suit_paths[p], grid) for p in pests}
    weights = {p: class_weights(weights_table, p, offseason.get(p, 0.0)) for p in pests}

    info = read_product_info(files.iloc[0]["path"])
    acc = PeriodAccumulator(grid, pests)
    period_out = os.path.join(out_dir, period)
    for _, f in files.iterrows():
        read_product_info(f["path"])     # stops if a province uses another class coding
        accumulate_province(acc, f["path"], f["area"], suitability, weights, chunk_rows=chunk_rows,
                            write_30m_dir=os.path.join(period_out, "30m") if write_30m else None,
                            period=period)
        log(f"  {f['area']}")

    tags = {"period": period, "quarter": quarter, "period_dates": info.get("period_dates"),
            "stage_window": info.get("window"), "stage_source": info.get("data_source"),
            "n_provinces": len(files)}
    written = finish_period(acc, suitability, weights, period_out, period, tags)

    manifest = {
        "period": period, "quarter": quarter,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "stage_files": {r["area"]: {"size": int(r["size"]), "mtime": int(r["mtime"])}
                        for _, r in files.iterrows()},
        "suitability_files": {p: os.path.basename(v) for p, v in suit_paths.items()},
        "class_weights": {p: [round(float(x), 4) for x in v] for p, v in weights.items()},
        "outputs": written,
    }
    with open(os.path.join(period_out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def pending_periods(products: pd.DataFrame, out_dir: str) -> list:
    """Periods with no manifest yet, or whose province files changed since the last run
    (for example provinces that failed before and were produced later)."""
    todo = []
    for period, files in products.groupby("period"):
        mpath = os.path.join(out_dir, period, "manifest.json")
        if not os.path.exists(mpath):
            todo.append(period)
            continue
        with open(mpath) as fh:
            done = json.load(fh).get("stage_files", {})
        now = {r["area"]: {"size": int(r["size"]), "mtime": int(r["mtime"])} for _, r in files.iterrows()}
        if now != done:
            todo.append(period)
    return sorted(todo)


def run_pending(stage_dir: str, suit_dir: str, weights_csv: str, out_dir: str,
                offseason: dict | None = None, pests: list | None = None, periods: list | None = None,
                write_30m: bool = False, log=print) -> list:
    products = list_stage_products(stage_dir)
    if products.empty:
        log(f"No stage products in {stage_dir}")
        return []
    todo = periods or pending_periods(products, out_dir)
    if not todo:
        log("Nothing to do: every period is up to date.")
        return []
    weights_table = load_stage_weights(weights_csv)
    all_areas = set(products["area"])
    results = []
    for period in todo:
        present = set(products.loc[products["period"] == period, "area"])
        log(f"Period {period}: {len(present)} provinces")
        absent = sorted(all_areas - present)
        if absent:
            log(f"  not in this period (failed or not run yet): {', '.join(absent)}")
        results.append(run_period(period, products, suit_dir, weights_table, out_dir,
                                  offseason=offseason, pests=pests, write_30m=write_30m, log=log))
    return results


# ------------------------------------------------------------------
# Stage at pest report points (needs products for the report periods)
# ------------------------------------------------------------------
def sample_stage_at_points(points: pd.DataFrame, products: pd.DataFrame) -> pd.Series:
    """Product class at each point for its own year and month (NaN where no product covers it).
    ``points`` needs year, month, latitude, longitude. Use it to check reported stages and to build
    the weight table from satellite stages once past periods are produced."""
    import rasterio

    out = pd.Series(np.nan, index=points.index, dtype=float)
    period_key = points["year"].astype(int).astype(str) + points["month"].astype(int).map("{:02d}".format)
    for period, idx in points.groupby(period_key).groups.items():
        files = products[products["period"] == period]
        if files.empty:
            continue
        pts = points.loc[idx]
        coords = list(zip(pts["longitude"].astype(float), pts["latitude"].astype(float)))
        for _, f in files.iterrows():
            with rasterio.open(f["path"]) as src:
                b = src.bounds
                inside = [(lo, la) for lo, la in coords if b.left <= lo < b.right and b.bottom < la <= b.top]
                if not inside:
                    continue
                vals = {xy: int(v[0]) for xy, v in zip(inside, src.sample(inside))}
            for i, xy in zip(idx, coords):
                v = vals.get(xy)
                if v is not None and v >= 0 and math.isnan(out.at[i]):
                    out.at[i] = v
    return out
