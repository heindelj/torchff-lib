import os, glob
from setuptools import setup, find_packages

# Allow CPU-only installs (e.g. CI, doc builds) when CUDA toolkit is absent.
# Set TORCHFF_NO_CUDA=1 to skip CUDA extension compilation.
_BUILD_CUDA = os.environ.get("TORCHFF_NO_CUDA", "0") != "1"

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    _BUILD_CUDA = False

if _BUILD_CUDA:
    if 'CXX' not in os.environ:
        os.environ['CXX'] = 'g++'
        print("Set C++ compiler: g++")


    def build_cuda_extension(name, exclude_files=list()):
        sources = []
        for file in glob.glob(os.path.join(os.path.dirname(__file__), f"csrc/{name}/*")):
            if (file.endswith('.cpp') or file.endswith('.cu') or file.endswith('.c') or file.endswith('.C')) and (os.path.basename(file) not in exclude_files):
                sources.append(file)
        
        return CUDAExtension(
            name=f'torchff_{name}',
            sources=sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            },
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")]
        )


    # Which extensions to build. Every one of the 15 takes minutes under nvcc and they are
    # compiled one after another, so `TORCHFF_EXTENSIONS=ffterms,nblist` (comma-separated
    # names from the list below, or "all") builds only what a project needs. The python
    # modules whose extension is missing are skipped by torchff/__init__.py.
    _ALL_EXTENSIONS = {
        'bond': lambda: build_cuda_extension('bond'),
        'angle': lambda: build_cuda_extension('angle'),
        'torsion': lambda: build_cuda_extension('torsion'),
        'vdw': lambda: build_cuda_extension('vdw'),
        'dispersion': lambda: build_cuda_extension('dispersion'),
        'slater': lambda: build_cuda_extension('slater'),
        'coulomb': lambda: build_cuda_extension('coulomb'),
        'multipoles': lambda: build_cuda_extension('multipoles'),
        'amoeba': lambda: build_cuda_extension('amoeba'),
        'ewald': lambda: build_cuda_extension('ewald', ['ewald_optimized.cu']),
        'pme': lambda: build_cuda_extension('pme'),
        'cmm': lambda: build_cuda_extension('cmm'),
        'ffterms': lambda: build_cuda_extension('ffterms'),
        # only compile the fused nonbonded atom-pair kernel in csrc/nonbonded
        'nb': lambda: CUDAExtension(
            name='torchff_nb',
            sources=['csrc/nonbonded/nonbonded_interface.cpp',
                     'csrc/nonbonded/nonbonded_atom_pairs_cuda.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")],
        ),
        'nblist': lambda: CUDAExtension(
            name='torchff_nblist',
            sources=['csrc/nblist/nblist_interface.cpp',
                     'csrc/nblist/nblist_nsquared_cuda.cu',
                     'csrc/nblist/nblist_clist_cuda.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")],
        ),
    }
    _selected = os.environ.get("TORCHFF_EXTENSIONS", "all").strip()
    if _selected in ("", "all"):
        _names = list(_ALL_EXTENSIONS)
    else:
        _names = [n.strip() for n in _selected.split(",") if n.strip()]
        _unknown = [n for n in _names if n not in _ALL_EXTENSIONS]
        if _unknown:
            raise SystemExit(f"TORCHFF_EXTENSIONS: unknown {_unknown}; choose from {list(_ALL_EXTENSIONS)}")
    print(f"torchff: building extensions {_names}")
    ext_modules = [_ALL_EXTENSIONS[n]() for n in _names]
    cmdclass = {'build_ext': BuildExtension}
else:
    print("CUDA not available — installing Python-only package (no CUDA extensions).")
    ext_modules = []
    cmdclass = {}


setup(
    name='torchff',
    classifiers=[
        'Development Status :: Beta',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.12',
    ],
    packages=find_packages(exclude=['csrc', 'tests', 'docs'], include=['torchff*']),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
