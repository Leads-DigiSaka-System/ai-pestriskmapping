import json
import os

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402

from pestrisk import stage as ps  # noqa: E402

PESTS = ["Rice bug complex", "Stem borer"]


def _weights(tmp_path):
    w = pd.DataFrame(
        {"establishment": [0.0, 0.3], "vegetative": [0.1, 1.0],
         "reproductive": [0.6, 1.0], "ripening": [1.0, 0.3]},
        index=PESTS,
    )
    path = tmp_path / "weights.csv"
    w.to_csv(path)
    return str(path)


def _write_stage(path, arr, x0, y0, d, classes=ps.EXPECTED_CLASSES_TAG, period="202610"):
    with rasterio.open(
        path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="int16",
        crs="EPSG:4326", transform=Affine(d, 0, x0, 0, -d, y0), nodata=-1,
    ) as dst:
        dst.write(arr.astype(np.int16), 1)
        dst.update_tags(classes=classes, period=period, period_dates="2026-10-01..2026-11-01",
                        window="2026-06-01..2026-11-16", data_source="hls", method="dominant")


def _write_suit(path, arr, x0, y0, d):
    with rasterio.open(
        path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="float32",
        crs="EPSG:4326", transform=Affine(d, 0, x0, 0, -d, y0), nodata=np.nan,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


@pytest.fixture
def setup(tmp_path):
    """1 km grid of 2x2 cells (0.01 deg). Stage pixels are 0.001 deg: 10x10 per cell.
    Province A covers the two top cells, province B the bottom-left cell."""
    stage_dir = tmp_path / "auto"
    suit_dir = tmp_path / "suit"
    stage_dir.mkdir()
    suit_dir.mkdir()
    x0, y0, D, d = 120.0, 15.0, 0.01, 0.001

    a = np.full((10, 20), -1)
    a[:, :10] = 4          # top-left cell: all ripening
    a[:, 10:15] = 2        # top-right cell: half vegetative,
    a[:, 15:] = 5          #                 half harvested
    _write_stage(stage_dir / "rice_PROV_A_202610.tiff", a, x0, y0, d)

    b = np.full((10, 10), -1)
    b[:5, :] = 3           # bottom-left cell: half reproductive, half outside rice mask
    _write_stage(stage_dir / "rice_PROV_B_202610.tiff", b, x0, y0 - D, d)

    # national mosaic is ignored
    _write_stage(stage_dir / "rice_PHILIPPINES_202610.tiff", np.zeros((2, 2)), x0, y0, d)

    for pest, val in [("rice_bug_complex", 0.8), ("stem_borer", 0.5)]:
        s = np.full((2, 2), val)
        s[1, 1] = np.nan
        _write_suit(suit_dir / f"rice_maxent_occurrence_suitability_{pest}_2021_2024_q4_ph_cropland_1km.tif",
                    s, x0, y0, D)
    return stage_dir, suit_dir, tmp_path


def test_list_products_skips_mosaic(setup):
    stage_dir, _, _ = setup
    p = ps.list_stage_products(str(stage_dir))
    assert sorted(p["area"]) == ["PROV_A", "PROV_B"]
    assert set(p["period"]) == {"202610"}


def test_class_weights_off_season():
    w = pd.DataFrame({"establishment": [0.2], "vegetative": [0.4], "reproductive": [0.6], "ripening": [0.8]},
                     index=["Rodents"])
    v = ps.class_weights(w, "Rodents", offseason=0.3)
    assert np.allclose(v, [0.3, 0.2, 0.4, 0.6, 0.8, 0.3])


def test_run_period_outputs(setup):
    stage_dir, suit_dir, tmp = setup
    out = tmp / "out"
    res = ps.run_pending(str(stage_dir), str(suit_dir), _weights(tmp), str(out),
                         offseason={"Stem borer": 0.1}, write_30m=True, log=lambda *_: None)
    assert len(res) == 1 and res[0]["quarter"] == "Q4"

    with rasterio.open(res[0]["outputs"]["stage_fractions"]) as src:
        frac = src.read()
    assert np.isclose(frac[4, 0, 0], 1.0)                                # all ripening
    assert np.isclose(frac[2, 0, 1], 0.5) and np.isclose(frac[5, 0, 1], 0.5)
    assert np.isclose(frac[3, 1, 0], 1.0)                                # nodata pixels not counted
    assert np.isnan(frac[0, 1, 1])                                       # no rice

    with rasterio.open(res[0]["outputs"]["risk"]["Rice bug complex"]) as src:
        assert src.descriptions[0] == "risk_standing_crop"
        standing, risk_all = src.read(1), src.read(2)
    assert np.isclose(standing[0, 0], 0.8 * 1.0, atol=1e-4)
    assert np.isclose(standing[0, 1], 0.8 * 0.1, atol=1e-4)            # harvested half left out
    assert np.isclose(risk_all[0, 1], 0.8 * (0.5 * 0.1 + 0.5 * 0.0), atol=1e-4)
    assert np.isclose(standing[1, 0], 0.8 * 0.6, atol=1e-4)
    assert np.isnan(standing[1, 1])

    with rasterio.open(res[0]["outputs"]["risk"]["Stem borer"]) as src:
        risk_sb = src.read(2)
    assert np.isclose(risk_sb[0, 1], 0.5 * (0.5 * 1.0 + 0.5 * 0.1), atol=1e-4)   # off-season weight 0.1

    table = pd.read_csv(res[0]["outputs"]["province_table"])
    rb = table[(table.province == "PROV A") & (table.pest_group == "Rice bug complex")].iloc[0]
    # 100 ripening px at risk 0.8 -> very_high; 50 vegetative at 0.08 and 50 harvested at 0 -> low
    px_ha = rb["rice_ha"] / 200
    assert np.isclose(rb["ha_very_high"], 100 * px_ha, rtol=1e-3)
    assert np.isclose(rb["ha_low"], 100 * px_ha, rtol=1e-3)
    assert np.isclose(rb["ha_stage_4"], 100 * px_ha, rtol=1e-3)
    assert np.isclose(rb["standing_crop_ha"], 150 * px_ha, rtol=1e-3)
    assert np.isclose(rb["mean_risk_standing_crop"], (100 * 0.8 + 50 * 0.08) / 150, atol=1e-3)

    tif30 = out / "202610" / "30m" / "rice_pest_risk_rice_bug_complex_prov_a_202610_30m.tif"
    with rasterio.open(tif30) as src:
        r30 = src.read(1)
    assert r30[0, 0] == 80 and r30[0, 12] == 8 and r30[0, 17] == 0

    manifest = json.loads((out / "202610" / "manifest.json").read_text())
    assert set(manifest["stage_files"]) == {"PROV_A", "PROV_B"}


def test_pending_reruns_when_a_province_is_added(setup):
    stage_dir, suit_dir, tmp = setup
    out = tmp / "out"
    ps.run_pending(str(stage_dir), str(suit_dir), _weights(tmp), str(out), log=lambda *_: None)
    products = ps.list_stage_products(str(stage_dir))
    assert ps.pending_periods(products, str(out)) == []

    _write_stage(stage_dir / "rice_PROV_C_202610.tiff", np.full((10, 10), 3), 120.01, 14.99, 0.001)
    products = ps.list_stage_products(str(stage_dir))
    assert ps.pending_periods(products, str(out)) == ["202610"]


def test_changed_class_coding_is_refused(tmp_path):
    p = tmp_path / "rice_X_202610.tiff"
    _write_stage(p, np.zeros((2, 2)), 120, 15, 0.001, classes="0=bare; 1=rice")
    with pytest.raises(ValueError, match="class coding changed"):
        ps.read_product_info(str(p))


def test_sample_stage_at_points(setup):
    stage_dir, _, _ = setup
    products = ps.list_stage_products(str(stage_dir))
    pts = pd.DataFrame({
        "year": [2026, 2026, 2026, 2025],
        "month": [10, 10, 10, 10],
        "latitude": [14.9995, 14.9895, 14.9805, 14.9995],
        "longitude": [120.0005, 120.0005, 120.0105, 120.0005],
    })
    s = ps.sample_stage_at_points(pts, products)
    assert s.iloc[0] == 4          # province A, ripening
    assert s.iloc[1] == 3          # province B, reproductive
    assert np.isnan(s.iloc[2])     # outside both rice masks
    assert np.isnan(s.iloc[3])     # no product for 2025-10
