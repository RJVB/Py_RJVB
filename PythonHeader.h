#ifndef _PYTHON_HEADER_H

#include <Python.h>
#if (PY_MAJOR_VERSION >= 3) || (PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >=6)
#	include <bytesobject.h>
#endif

#if PY_MAJOR_VERSION >= 3
#	define IS_PY3K
#	define PyInt_FromLong	PyLong_FromLong
#	define PyInt_Check	PyLong_Check
#elif (PY_MAJOR_VERSION >= 3) || (PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >=6)
#	include <intobject.h>
#endif

#define _PYTHON_HEADER_H
#endif
