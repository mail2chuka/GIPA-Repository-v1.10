# Verification of public version 1.10

Verified on 23 November 2026 before upload.

- Source package: `ITS-2026-08-0579_R1_Supplementary_Data_1-7.zip`; its SHA-256 is recorded in `SOURCE_ARCHIVE.json`. All 574 entries of its checksum manifest matched before packaging.
- The 423 files under `src/`, `experiments/` and `results/` are unchanged from that package.
- All 56 distributed Python files passed syntax parsing. All 20 Bash runners passed `bash -n`.
- In Python 3.12 with the pinned NumPy 2.4.4 and SciPy 1.17.1, the planned prior at Regime A, N = 160, z = 2 reproduced all 30 archived seed results exactly: 317 numeric values, maximum difference 0. Mean latency was 16.694631944444446 ms and both class deadline-miss rates were zero.
- The backlog-priority variant at N = 160 also reproduced its plan, screen and 30 matched seed results: 79 numeric values agreed within a maximum difference of 3.55e-15. Mean latency was 17.11490324074074 ms with no observed deadline misses.
- Rebuilding the main statistics from archived seed-level data reproduced the structure and all 3,983 numeric values within a largest scaled difference of 3.5e-15. The comparison passed tolerance 1e-9; byte identity was not expected across the recorded environments.
- The complete learning experiments were not retrained for this upload. Their archived records and reproduction commands are included.

To check the distributed files after downloading, run `python scripts/verify_archive.py`. The manifest checks file integrity, not whether every experiment has been rerun.
