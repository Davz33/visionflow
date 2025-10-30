"""
setup.py for building C extensions
Run: python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from pathlib import Path
import platform
import numpy

# Define C extension
extensions = []

if platform.system() == "Darwin":
    # macOS - use Accelerate framework
    c_ext = Extension(
        'visionflow_c_extensions',
        sources=[
            str(Path(__file__).parent / 'visionflow/services/generation/visionflow_c_extensions.c')
        ],
        include_dirs=[
            numpy.get_include(),  # Add NumPy include directory
            '/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers'
        ],
        extra_link_args=['-framework', 'Accelerate'],  # Use framework linking
        extra_compile_args=['-O3', '-march=native', '-DACCELERATE_NEW_LAPACK'],  # Use new LAPACK API
        language='c'
    )
    extensions.append(c_ext)

setup(
    name='visionflow-c-extensions',
    version='0.1.0',
    ext_modules=extensions,
    zip_safe=False,
)
