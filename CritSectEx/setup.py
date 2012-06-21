import os,sys
from distutils.core import setup, Extension
from numpy import get_include as numpy_get_include

try:
	from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
	from distutils.command.build_py import build_py

def doall():
	if sys.platform.find('linux') >= 0:
		extra_link_args=['-g','-lrt','-lpthread']
	else:
		extra_link_args=['-g']
	setup(name='_CritSectEx', version='1.0',
		 description = 'a package providing access to c++ CritSectEx',
		 ext_modules = [Extension('_CritSectEx',
							 sources=['CritSectEx_wrap.c','CritSectEx.cpp','msemul.cpp','timing.c'],
							 depends=['CritSectEx.h','msemul.h','timing.h'],
							 extra_compile_args=['-g','-I../..'],
							 extra_link_args=extra_link_args
							 )])

	setup(name='CritSectEx', version='1.0',
		 description = 'a package providing access to c++ CritSectEx',
		 py_modules=['CritSectEx','MSEmul'], cmdclass={'build_py':build_py}, )

	setup(name='_MSEmul', version='1.0',
		 description = 'a package providing access to c++ msemul',
		 ext_modules = [Extension('_MSEmul',
							 sources=['msemul_wrap.cxx','msemul.cpp','timing.c'],
							 depends=['msemul.h','timing.h'],
							 extra_compile_args=['-g','-I../..'],
							 extra_link_args=extra_link_args
							 )])


doall()

os.system("rm -rf build/lib")
