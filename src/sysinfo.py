"""
sysinfo.py
==========
Records the computing environment that produced a result file, in enough
detail to state it in the manuscript: processor model and features, core
and thread counts, memory, virtualisation, operating system, kernel, C
library, Python build, numerical-library versions and the BLAS in use.

STATUS: written for the first revision of manuscript ITS-2026-08-0579.
Used by experiments/exp_complexity.py (decision timings, Section VII-H).
Nothing else in the repository depends on it.

Only the standard library and the installed numerical packages are used.
Every field falls back to None when the information is not available on
the host (for example on a non-Linux system), so the function never fails.

Run on its own to print the record for the current machine:

    PYTHONPATH=src python src/sysinfo.py
"""

import json
import os
import platform
import sys


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def _cpuinfo():
    """Model name, vendor, clock, cache and selected flags from /proc/cpuinfo."""
    txt = _read('/proc/cpuinfo')
    out = dict(model=platform.processor() or None, vendor=None, mhz=None, cache=None,
               flags=None, hypervisor=False, sockets=None, physical_cores=None)
    if not txt:
        return out
    first = txt.split('\n\n')[0]
    fields = {}
    for line in first.splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            fields[k.strip()] = v.strip()
    out['model'] = fields.get('model name', out['model'])
    out['vendor'] = fields.get('vendor_id')
    out['mhz'] = float(fields['cpu MHz']) if 'cpu MHz' in fields else None
    out['cache'] = fields.get('cache size')
    flags = set(fields.get('flags', '').split())
    keep = ['sse4_2', 'avx', 'avx2', 'fma', 'avx512f']
    out['flags'] = [f for f in keep if f in flags]
    out['hypervisor'] = 'hypervisor' in flags
    phys = {(b.get('physical id'), b.get('core id'))
            for b in ({k.strip(): v.strip() for k, v in (l.split(':', 1) for l in blk.splitlines() if ':' in l)}
                      for blk in txt.strip().split('\n\n'))}
    out['sockets'] = len({p for p, _ in phys if p is not None}) or None
    out['physical_cores'] = len(phys) or None
    return out


def _hypervisor_vendor():
    """Hypervisor vendor (e.g. 'KVM') from lscpu, else the DMI platform vendor."""
    try:
        import subprocess
        txt = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=10).stdout
        for line in txt.splitlines():
            if line.startswith('Hypervisor vendor:'):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    for p in ('/sys/class/dmi/id/sys_vendor', '/sys/class/dmi/id/product_name'):
        v = _read(p)
        if v and v.strip():
            return v.strip()
    return None


def _memory_gib():
    txt = _read('/proc/meminfo')
    if not txt:
        return None
    for line in txt.splitlines():
        if line.startswith('MemTotal:'):
            return round(int(line.split()[1]) / 1024 ** 2, 1)
    return None


def _os_release():
    txt = _read('/etc/os-release')
    if not txt:
        return platform.platform()
    for line in txt.splitlines():
        if line.startswith('PRETTY_NAME='):
            return line.split('=', 1)[1].strip().strip('"')
    return platform.platform()


def _versions():
    out = {}
    for name in ('numpy', 'scipy', 'matplotlib'):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def _blas():
    """BLAS library and version reported by NumPy's build configuration."""
    try:
        import numpy as np
        cfg = np.__config__.CONFIG
        b = cfg['Build Dependencies']['blas']
        return dict(name=b.get('name'), version=b.get('version'),
                    configuration=b.get('openblas configuration'))
    except Exception:
        return None


def system_info():
    """Return a JSON-serialisable record of the computing environment."""
    cpu = _cpuinfo()
    libc = platform.libc_ver()
    return dict(
        processor=dict(model=cpu['model'], vendor=cpu['vendor'], reported_mhz=cpu['mhz'],
                       cache_reported=cpu['cache'], instruction_sets=cpu['flags'],
                       logical_cpus=os.cpu_count(), physical_cores=cpu['physical_cores'],
                       sockets=cpu['sockets']),
        virtualised=cpu['hypervisor'],
        hypervisor_vendor=_hypervisor_vendor(),
        memory_gib=_memory_gib(),
        os=_os_release(),
        kernel=platform.release(),
        architecture=platform.machine(),
        libc=f'{libc[0]} {libc[1]}' if libc[0] else None,
        python=dict(implementation=platform.python_implementation(), version=platform.python_version(),
                    compiler=platform.python_compiler(), build=' '.join(platform.python_build())),
        packages=_versions(),
        blas=_blas(),
        thread_env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        argv=sys.argv,
    )


if __name__ == '__main__':
    print(json.dumps(system_info(), indent=1))
