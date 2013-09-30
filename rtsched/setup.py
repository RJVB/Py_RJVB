from distutils.core import setup, Extension

setup(name='rtsched', version='1.0',
	 description = 'a package providing (limited) real-time processing/priority functionality',
      ext_modules = [Extension('rtsched',
                               sources=['rtsched.c'],
                               )])
