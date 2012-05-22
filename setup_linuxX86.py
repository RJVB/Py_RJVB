from distutils.core import setup, Extension
from numpy import get_include as numpy_get_include

try:
	from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
	from distutils.command.build_py import build_py

def doall():
	setup(name='HRTime', version='1.0',
		 description = 'a package providing access to the high resolution timer',
		 ext_modules = [Extension('HRTime',
							 sources=['HRTime.c'],
							 libraries=['rt'],
							 )])

	setup(name='rtsched', version='1.0',
		 description = 'a package providing (limited) real-time processing/priority functionality',
		 ext_modules = [Extension('rtsched',
							 sources=['rtsched/rtsched.c'],
							 )])

# these come from the SciPy cookbook:
	setup(name='rebin', version='1.0',
		 description = 'a package providing the rebin "resampling function',
		 py_modules=['rebin'], cmdclass={'build_py':build_py}, )
	setup(name='DataFrame', version='1.0',
		 description = 'a package providing R-like data.frame functionality',
		 py_modules=['DataFrame'], cmdclass={'build_py':build_py} )

# these depend on numpy:
	setup(name='rjvbFilt', version='1.0',
		 description = 'a package providing some filtring functions',
		 ext_modules = [Extension('rjvbFilt',
							 sources=['rjvbFilt.c','rjvbFilters.c'],
							 depends=['ParsedSequence.h'],
							 include_dirs = [numpy_get_include()],
							 extra_compile_args =['-mfpmath=sse','-msse','-msse2','-msse3','-msse4','-ftree-vectorize']
							 )])

	setup(name='sincos', version='1.0',
		 description = 'a package providing a sincos function',
		 ext_modules = [Extension('sincos',
							 sources=['sincos.c','mips_sincos.c','cephes_sincos.c'],
							 include_dirs = [numpy_get_include()],
							 extra_compile_args =['-mfpmath=sse','-msse','-msse2','-msse3','-msse4','-ftree-vectorize']
							 )])

	setup(name='fmadd', version='1.0',
		 description = 'a package providing an fmadd function',
		 ext_modules = [Extension('fmadd',
							 sources=['fmadd.c'],
							 include_dirs = [numpy_get_include()]
							 )])

doall()

import os
os.system("rm -rf build/lib")
