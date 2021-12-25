#! /usr/bin/env python

import os
import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup

descr = """Sinkformers:  Transformers with Doubly Stochastic Attention"""

version = None
with open(os.path.join('sinkformers', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


DISTNAME = 'sinkformers'
DESCRIPTION = descr
MAINTAINER = 'XX XXXX'
MAINTAINER_EMAIL = 'XX.XXXX@XX.fr'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/XXX/XXX.git'
VERSION = version
URL = 'https://github.com/XXX/XXXX'

def get_requirements():
    """Return the requirements of the projects in requirements.txt"""
    with open('requirements.txt') as f:
        requirements = [r.strip() for r in f.readlines()]
    return [r for r in requirements if r != '']

def package_tree(pkgroot):
    """Get the submodule list."""
    # Adapted from VisPy
    path = os.path.dirname(__file__)
    subdirs = [os.path.relpath(i[0], path).replace(os.path.sep, '.')
               for i in os.walk(os.path.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return sorted(subdirs)


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          long_description_content_type='text/x-rst',
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS',
                       'Programming Language :: Python :: 3',
                       ],
          platforms='any',
          python_requires='>=3.8',
          packages=package_tree('sinkformers'),
          install_requires=get_requirements()
          )
