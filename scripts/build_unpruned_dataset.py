#!/usr/bin/env python
"""Build the unpruned megamaser spot-table dataset.

The published tables define the spot list and astrometry. Fiducial
accelerations replace published values for velocity-matched spots, and the 19
NGC6323 spots added by P20 take their astrometry from Kuo et al. (2011).
"""
import csv
import shutil
from pathlib import Path

from candel.pvdata.megamaser_data import (load_megamaser_spots,
                                          maser_data_root)


GALAXY_FILES = {
    "CGCG074-064": "CGCG074-064_Pesce2020_mrt.txt",
    "NGC4258": "N4258_disk_data_MarkReid.final",
    "NGC5765b": "NGC5765b_Gao2016_table6.dat",
    "NGC6264": "NGC6264_Kuo2013_table2.txt",
    "NGC6323": "NGC6323_Kuo2015_table2.txt",
    "UGC3789": "UGC3789_Reid2013_table1.txt",
}
UNCHANGED = ("CGCG074-064", "NGC4258")


def _load(dataset, galaxy):
    return load_megamaser_spots(
        maser_data_root(dataset), galaxy, v_sys_obs=0.0)


def _rows(galaxy):
    published = _load("original_published", galaxy)
    fiducial = _load("fiducial", galaxy)
    fid_index = {v: i for i, v in enumerate(fiducial["velocity"])}
    if len(fid_index) != fiducial["n_spots"]:
        raise ValueError(f"{galaxy}: duplicate velocity in fiducial table")

    rows = []
    for i, velocity in enumerate(published["velocity"]):
        row = {key: published[key][i] for key in
               ("velocity", "x", "sigma_x", "y", "sigma_y", "a",
                "sigma_a", "accel_measured")}
        if velocity in fid_index:
            j = fid_index[velocity]
            for key in ("a", "sigma_a", "accel_measured"):
                row[key] = fiducial[key][j]
            row["acceleration_source"] = "fiducial"
        else:
            row["acceleration_source"] = "original_published"
        row["astrometry_source"] = "original_published"
        row["clipped_by_pesce"] = velocity not in fid_index
        rows.append(row)

    if galaxy == "NGC6323":
        published_velocities = set(published["velocity"])
        kuo = _load_kuo2011_ngc6323()
        for j, velocity in enumerate(fiducial["velocity"]):
            if velocity in published_velocities:
                continue
            if velocity not in kuo:
                raise ValueError(
                    f"NGC6323: fiducial-only velocity {velocity} is absent "
                    "from Kuo et al. (2011) Table 3")
            row = dict(kuo[velocity])
            for key in ("a", "sigma_a", "accel_measured"):
                row[key] = fiducial[key][j]
            row["astrometry_source"] = "Kuo2011_table3"
            row["acceleration_source"] = "fiducial"
            row["clipped_by_pesce"] = False
            rows.append(row)

    if len({row["velocity"] for row in rows}) != len(rows):
        raise ValueError(f"{galaxy}: duplicate velocity in unpruned table")
    return rows


def _load_kuo2011_ngc6323():
    path = (Path(maser_data_root("original_published")).parent /
            "Kuo2011_MCP_III_table3.dat")
    rows = {}
    with path.open() as f:
        for line in f:
            if not line.startswith("NGC 6323"):
                continue
            _, _, velocity, x, sigma_x, y, sigma_y, *_ = line.split()
            velocity = float(velocity)
            rows[velocity] = {
                "velocity": velocity,
                "x": 1000.0 * float(x),
                "sigma_x": 1000.0 * float(sigma_x),
                "y": 1000.0 * float(y),
                "sigma_y": 1000.0 * float(sigma_y),
            }
    return rows


def _write_gao(path, rows):
    with path.open("w") as f:
        for row in rows:
            placeholder = row["a"] == 1.0 and row["sigma_a"] == 1.0
            a = row["a"] if row["accel_measured"] or placeholder else 0.0
            sigma_a = (row["sigma_a"]
                       if row["accel_measured"] or placeholder else 0.2)
            values = (row["velocity"], row["x"] / 1000,
                      row["sigma_x"] / 1000, row["y"] / 1000,
                      row["sigma_y"] / 1000, a, sigma_a)
            f.write("\t".join(f"{value:.10g}" for value in values) + "\n")


def _write_kuo(path, rows):
    with path.open("w") as f:
        for row in rows:
            values = (row["velocity"], row["x"] / 1000,
                      row["sigma_x"] / 1000, row["y"] / 1000,
                      row["sigma_y"] / 1000)
            fields = [f"{value:.10g}" for value in values]
            if row["accel_measured"]:
                fields += [f"{row['a']:.10g}", f"{row['sigma_a']:.10g}"]
            else:
                fields += ["...", "..."]
            f.write("\t".join(fields) + "\n")


def main():
    published_root = Path(maser_data_root("original_published"))
    output_root = published_root.parent / "unpruned"
    output_root.mkdir(parents=True, exist_ok=True)

    for galaxy in UNCHANGED:
        filename = GALAXY_FILES[galaxy]
        shutil.copy2(published_root / filename, output_root / filename)

    master_rows = {}
    for galaxy in (name for name in GALAXY_FILES if name not in UNCHANGED):
        rows = _rows(galaxy)
        master_rows[galaxy] = rows
        path = output_root / GALAXY_FILES[galaxy]
        (_write_gao if galaxy == "NGC5765b" else _write_kuo)(path, rows)

    with (output_root / "provenance.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("galaxy", "spot_index", "velocity_km_s",
                         "astrometry_source", "acceleration_source",
                         "clipped_by_pesce"))
        for galaxy in GALAXY_FILES:
            if galaxy in UNCHANGED:
                rows = _load("original_published", galaxy)
                velocities = rows["velocity"]
                sources = [("original_published", "original_published",
                            False)
                           for _ in velocities]
            else:
                velocities = [row["velocity"] for row in master_rows[galaxy]]
                sources = [(row["astrometry_source"],
                            row["acceleration_source"],
                            row["clipped_by_pesce"])
                           for row in master_rows[galaxy]]
            for i, (velocity, source) in enumerate(zip(velocities, sources)):
                writer.writerow((galaxy, i, f"{velocity:.10g}", *source))

    counts = {galaxy: (_load("original_published", galaxy)["n_spots"]
                       if galaxy in UNCHANGED else len(master_rows[galaxy]))
              for galaxy in GALAXY_FILES}
    with (output_root / "README.md").open("w") as f:
        f.write(
            "# Unpruned megamaser dataset\n\n"
            "This dataset preserves every `original_published` spot and its "
            "published astrometry. For velocity-matched rows, acceleration, "
            "acceleration uncertainty, and measurement status come from the "
            "fiducial table. Published accelerations are retained when a spot "
            "was pruned from the fiducial table.\n\n"
            "NGC6323 additionally contains the 19 fiducial-only spots "
            "identified in the MCP response. Their astrometry is taken "
            "directly from Kuo et al. (2011), Table 3, while their acceleration "
            "fields come from the fiducial table. No other Kuo et al. (2011) "
            "spots are added. `provenance.csv` records the source used for "
            "every row and whether its published velocity was absent from "
            "the Pesce fiducial table (`clipped_by_pesce`).\n\n"
            "Spot counts: " + ", ".join(
                f"{galaxy}={count}" for galaxy, count in counts.items()) +
            ".\n")
    print(f"Wrote unpruned dataset to {output_root}")


if __name__ == "__main__":
    main()
