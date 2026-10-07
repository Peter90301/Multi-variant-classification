"""Generate fictional CSV, BIOM, TSV, and QZA inputs for public tests."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np
from biom import Table
from biom.util import biom_open


DNA = [
    "ACGTACGTACGTACGT", "CGTACGTACGTACGTA", "GTACGTACGTACGTAC",
    "TACGTACGTACGTACG", "AAAACCCCGGGGTTTT", "CCCCGGGGTTTTAAAA",
    "GGGGTTTTAAAACCCC", "TTTTAAAACCCCGGGG", "ACACACACGTGTGTGT",
    "CACACACATGTGTGTG", "AGCTAGCTAGCTAGCT", "TCGATCGATCGATCGA",
]
HMTOL_FEATURES = [f"G{index:05d}" for index in range(len(DNA))]


def write_biom(path: Path, sample_ids: list[str], feature_ids: list[str], classes: list[int]) -> None:
    rng = np.random.default_rng(2026)
    data = rng.integers(1, 4, size=(len(feature_ids), len(sample_ids))).astype(float)
    block = max(1, len(feature_ids) // (max(classes) + 1))
    for sample, class_index in enumerate(classes):
        start = class_index * block
        data[start:start + block, sample] += 20
    table = Table(data, feature_ids, sample_ids)
    with biom_open(path, "w") as handle:
        table.to_hdf5(handle, "Fictional public test data")


def write_qza(path: Path, feature_ids: list[str]) -> None:
    fasta = "".join(
        f">{feature}\n{DNA[index % len(DNA)]}\n"
        for index, feature in enumerate(feature_ids)
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("synthetic-uuid/data/dna-sequences.fasta", fasta)


def write_marine(path: Path) -> None:
    fields = [
        "sample", "count", "domain", "phylum", "class", "order", "family",
        "genus", "species", "ASV_sequence", "geo_loc_name",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in range(30):
            group = sample % 3
            for replicate in range(2):
                writer.writerow({
                    "sample": f"PUBLIC_MARINE_{sample:03d}",
                    "count": 10 + group + replicate,
                    "domain": "Eukarya", "phylum": f"SyntheticPhylum{group}",
                    "class": f"SyntheticClass{group}", "order": f"SyntheticOrder{group}",
                    "family": f"SyntheticFamily{group}", "genus": f"SyntheticGenus{group}",
                    "species": f"Synthetic species {group}",
                    "ASV_sequence": DNA[group * 2 + replicate],
                    "geo_loc_name": f"Synthetic Bay {group}",
                })


def write_emp(root: Path) -> tuple[Path, Path]:
    sample_ids = [f"PUBLIC_EMP_{index:03d}" for index in range(60)]
    classes = [index % 3 for index in range(60)]
    biom_path = root / "synthetic_emp.biom"
    metadata_path = root / "synthetic_emp.tsv"
    write_biom(biom_path, sample_ids, DNA, classes)
    with metadata_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["#SampleID", "empo_1", "empo_2", "empo_3"])
        for sample, group in zip(sample_ids, classes):
            writer.writerow([sample, f"EMPO1_{group}", f"EMPO2_{group}", f"EMPO3_{group}"])
    return biom_path, metadata_path


def write_hmtol_before(root: Path) -> Path:
    sample_ids = [f"PUBLIC_PRE_{index:03d}" for index in range(60)]
    classes = [index % 3 for index in range(60)]
    write_biom(root / "feature-table.biom", sample_ids, HMTOL_FEATURES, classes)
    with (root / "metadata.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["SampleID", "Country"])
        writer.writerows((sample, f"Synthetic Country {group}") for sample, group in zip(sample_ids, classes))
    qza = root / "synthetic_sequences.qza"
    write_qza(qza, HMTOL_FEATURES)
    return qza


def write_hmtol_qc(root: Path) -> Path:
    sample_ids, studies, regions, continents, classes = [], [], [], [], []
    for study in range(6):
        for region in range(3):
            sample_ids.append(f"PUBLIC_QC_S{study}_R{region}")
            studies.append(f"Synthetic Study {study}")
            regions.append(f"Synthetic Region {region}")
            continents.append("Synthetic Continent A" if region == 0 else "Synthetic Continent B")
            classes.append(region)
    write_biom(root / "feature-table.qc.min3.biom", sample_ids, HMTOL_FEATURES, classes)
    with (root / "metadata.qc.min3.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["SampleID", "study", "Continent", "region"])
        writer.writerows(zip(sample_ids, studies, continents, regions))
    qza = root / "synthetic_sequences.qza"
    write_qza(qza, HMTOL_FEATURES)
    return qza
