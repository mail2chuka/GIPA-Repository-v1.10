# Computational Environment

## Public repository version 1.10

This release republishes the archived first-revision experiments. The environment descriptions below are the original experiment records, not a claim that every experiment was rerun for this upload. Use the pinned `requirements.txt` for verification; training and timing results can vary across library versions and machines.

- Operating system: Linux x86-64
- Python: CPython 3.12.3
- NumPy: 2.4.4
- SciPy: 1.17.1
- Matplotlib: 3.10.8

The exact archived JSON outputs are included so figures and reported statistics can be regenerated without rerunning the simulator.

## Revision (v1.1.0) reference machine

The revision experiments (E1–E12), the rebuilt statistics and the revision figures were run on the following machine. The decision timings in the manuscript (Section VII-H, Fig. R8) are from this machine; the complete record is stored in `results/res_complexity.json` under `hardware` → `full` (`src/sysinfo.py`).

- Processor: Intel Xeon Processor @ 2.10 GHz (model name as exposed by the hypervisor), x86-64, SSE4.2, AVX, AVX2, FMA, AVX-512F
- Cores: 2 vCPU (2 cores, 1 thread per core); L2 2 MiB per core; L3 260 MiB (as reported)
- Virtualisation: KVM virtual machine
- Memory: 7.8 GiB
- Operating system: Ubuntu 24.04.5 LTS, Linux kernel 6.18.44, glibc 2.39
- Python: CPython 3.13.17 (Clang 23.1.3)
- NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2; BLAS: OpenBLAS 0.3.34 (scipy-openblas)
- Timing runs: one thread (OPENBLAS_NUM_THREADS = OMP_NUM_THREADS = MKL_NUM_THREADS = 1)

All simulation results are deterministic given their seeds. MAPPO training (E7) was repeated on a second machine (Intel Core i5-4210M, 8 GB RAM, Ubuntu 24.04); group means agree within 0.010 ms.

