/*
 * Python interface to high(er) resolution timer(s).
 \ (c) 2005 R.J.V. Bertin, Mac OS X version.
 */
 
/*
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/

#include "PythonHeader.h"
#include "Py_InitModule.h"
#if PY_MAJOR_VERSION >= 2
#	ifndef MS_WINDOWS
#		include <numpy/arrayobject.h>
#	else
// #		include <../lib/site-packages/numpy/core/include/numpy/arrayobject.h>
#		include <numpy/arrayobject.h>
#	endif // MS_WINDOWS
#else
#	error "not yet configured for this Python version"
#endif


#include <stdio.h>
#include <math.h>

#include <errno.h>

#ifdef __cplusplus
#	include <valarray>
#	include <macstl/valarray.h>
#endif

static PyObject *FMError=NULL;

    /*DOC*/ static char doc_fmadd[] =
    /*DOC*/    "fmadd.fmadd(x,y,z) -> double\n"
    /*DOC*/    "returns x * y + z\n"
    /*DOC*/ ;

static PyObject *fmadd_fmadd( PyObject *self, PyObject *args )
{ double x, y, z;
	
	if(!PyArg_ParseTuple(args, "ddd:fmadd", &x, &y, &z )){
		return NULL;
	}
	return Py_BuildValue( "d", (x * y + z) );
}

    /*DOC*/ static char doc_fmadd2[] =
    /*DOC*/    "fmadd.fmadd2(x,y,z[,inplace]) -> double\n"
    /*DOC*/    "returns x * y + z\n"
    /*DOC*/    "arguments may be scalar, list, tuple or ndarray but must be of identical size.\n"
    /*DOC*/    "Returns an object of the same type as the 1st element (but 1D in case of ndarrays)\n"
    /*DOC*/    "when x, y, z are ndarrays, this is about twice as fast as the direct array operation (on a PowerPC)\n"
    /*DOC*/    "When inplace is specified and True (default: False), the operation is performed in-place in the ndarray X.\n"
    /*DOC*/ ;

static PyObject *fmadd_fmadd2( PyObject *self, PyObject *args )
{ PyObject *X, *Y, *Z, *ret=NULL;
  PyArrayObject *Xarr=NULL, *Yarr=NULL, *Zarr=NULL;
  PyArrayIterObject *Xit=NULL, *Yit=NULL, *Zit=NULL;
  double x, y, z, *array= NULL;
  double *Xarray= NULL, *Yarray= NULL, *Zarray= NULL;
  int i, yN, zN, isTuple= 0, inplace= 0;
  npy_intp xN;
	
	if(!PyArg_ParseTuple(args, "OOO|i:fmadd2", &X, &Y, &Z, &inplace )){
		return NULL;
	}

	PyErr_Clear();
	x= PyFloat_AsDouble(X);
	y= PyFloat_AsDouble(Y);
	z= PyFloat_AsDouble(Z);
	if( !PyErr_Occurred() ){
		return Py_BuildValue( "d", (x * y + z) );
	}
	else{
		PyErr_Clear();
	}

	if( PyArray_Check(X) ){
		xN= PyArray_Size(X);
		if( (Xarr= (PyArrayObject*) PyArray_ContiguousFromObject( X, PyArray_DOUBLE, 0,0)) ){
			Xarray= (double*) PyArray_DATA(Xarr);
		}
		else{
			Xarr= (PyArrayObject*) X;
			Xit= (PyArrayIterObject*) PyArray_IterNew(X);
		}
	}
	else if( PyList_Check(X) ){
		if( !(X= PyList_AsTuple(X)) ){
			PyErr_SetString( FMError, "Unexpected failure converting X list to tuple" );
			goto FMA_ESCAPE;
		}
		xN= PyTuple_Size(X);
	}
	else if( PyTuple_Check(X) ){
		xN= PyTuple_Size(X);
		isTuple= 1;
	}
	else{
		PyErr_SetString( FMError, "X: arguments may be scalar, tuple, list or ndarray" );
		goto FMA_ESCAPE;
	}
	if( PyArray_Check(Y) ){
		yN= PyArray_Size(Y);
		if( (Yarr= (PyArrayObject*) PyArray_ContiguousFromObject( Y, PyArray_DOUBLE, 0,0)) ){
			Yarray= (double*) PyArray_DATA(Yarr);
		}
		else{
			Yarr= (PyArrayObject*) Y;
			Yit= (PyArrayIterObject*) PyArray_IterNew(Y);
		}
	}
	else if( PyList_Check(Y) ){
		if( !(Y= PyList_AsTuple(Y)) ){
			PyErr_SetString( FMError, "Unexpected failure converting Y list to tuple" );
			goto FMA_ESCAPE;
		}
		yN= PyTuple_Size(Y);
	}
	else if( PyTuple_Check(Y) ){
		yN= PyTuple_Size(Y);
	}
	else{
		PyErr_SetString( FMError, "Y: arguments may be scalar, tuple, list or ndarray" );
		goto FMA_ESCAPE;
	}
	if( PyArray_Check(Z) ){
		zN= PyArray_Size(Z);
		if( (Zarr= (PyArrayObject*) PyArray_ContiguousFromObject( Z, PyArray_DOUBLE, 0,0)) ){
			Zarray= (double*) PyArray_DATA(Zarr);
		}
		else{
			Zarr= (PyArrayObject*) Z;
			Zit= (PyArrayIterObject*) PyArray_IterNew(Z);
		}
	}
	else if( PyList_Check(Z) ){
		if( !(Z= PyList_AsTuple(Z)) ){
			PyErr_SetString( FMError, "Unexpected failure converting Z list to tuple" );
			goto FMA_ESCAPE;
		}
		zN= PyTuple_Size(Z);
	}
	else if( PyTuple_Check(Z) ){
		zN= PyTuple_Size(Z);
	}
	else{
		PyErr_SetString( FMError, "Z: arguments may be scalar, tuple, list or ndarray" );
		goto FMA_ESCAPE;
	}

	if( xN!= yN || xN!= zN || yN!= zN ){
		PyErr_SetString( FMError, "Arguments must all have the same size" );
		goto FMA_ESCAPE;
	}

	if( Xarr ){
		if( inplace && Xarray ){
			array= Xarray;
		}
		else{
			if( !(array= (double*) PyMem_New( double, xN )) ){
				PyErr_NoMemory();
				goto FMA_ESCAPE;
			}
			inplace= 0;
		}
	}
	else{
		if( !(ret= PyList_New(xN)) ){
			PyErr_NoMemory();
			goto FMA_ESCAPE;
		}
		inplace= 0;
	}

	if( inplace && !Xarray ){
		PyErr_Warn( PyExc_Warning, "inplace operation requires the 1st argument to be a contiguous (numpy) ndarray" );
	}

#ifdef __cplusplus
	if( Xarray && Yarray && Zarray && array ){
	  stdext::refarray<double> vx(Xarray,xN);
	  stdext::refarray<double> vy(Yarray,xN), vz(Zarray,xN);
	  stdext::refarray<double> vr(array,xN);
		vr = vx * vy + vz;
	}
	else
#endif
	for( i= 0; i< xN; i++ ){
		if( Xarr ){
			if( Xarray ){
				x= Xarray[i];
			}
			else{
				x= PyFloat_AsDouble( Xarr->descr->f->getitem( Xit->dataptr, X ) );
				PyArray_ITER_NEXT(Xit);
			}
		}
		else{
			x= PyFloat_AsDouble( PyTuple_GetItem( X, i ) );
		}
		if( Yarr ){
			if( Yarray ){
				y= Yarray[i];
			}
			else{
				y= PyFloat_AsDouble( Yarr->descr->f->getitem( Yit->dataptr, Y ) );
				PyArray_ITER_NEXT(Yit);
			}
		}
		else{
			y= PyFloat_AsDouble( PyTuple_GetItem( Y, i ) );
		}
		if( Zarr ){
			if( Zarray ){
				z= Zarray[i];
			}
			else{
				z= PyFloat_AsDouble( Zarr->descr->f->getitem( Zit->dataptr, Z ) );
				PyArray_ITER_NEXT(Zit);
			}
		}
		else{
			z= PyFloat_AsDouble( PyTuple_GetItem( Z, i ) );
		}

		if( array ){
			array[i]= x * y + z;
		}
		else{
			PyList_SetItem(ret, i, PyFloat_FromDouble( x * y + z ) );
		}
	}
	if( array ){
		if( array== Xarray ){
			ret= X;
			Py_INCREF(ret);
		}
		else{
		  npy_intp dim[1] = {xN};
// 			ret= PyArray_FromDimsAndData( 1, &xN, PyArray_DOUBLE, (char*) array );
			ret= PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (char*) array );
			((PyArrayObject*)ret)->flags|= NPY_OWNDATA;
		}
	}
	else if( isTuple ){
	  PyObject *r= PyList_AsTuple(ret);
		if( r ){
			ret= r;
		}
		else{
			if( PyErr_Occurred() ){
				PyErr_Print();
			}
			PyErr_Warn( PyExc_Warning, "conversion of result to tuple failed; returning list" );
		}
	}

FMA_ESCAPE:;
	if( Xit ){
		Py_XDECREF(Xit);
	}
	else if( Xarr ){
		Py_XDECREF(Xarr);
	}
	if( Yit ){
		Py_XDECREF(Yit);
	}
	else if( Yarr ){
		Py_XDECREF(Yarr);
	}
	if( Zit ){
		Py_XDECREF(Zit);
	}
	else if( Zarr ){
		Py_XDECREF(Zarr);
	}
	return(ret);
}


static PyMethodDef fmadd_methods[] =
{
	{ "fmadd", fmadd_fmadd2, METH_VARARGS, doc_fmadd2 },
	{ NULL, NULL }
};

#ifdef __cplusplus
extern "C" {
#endif

#ifndef IS_PY3K
void initfmadd()
#else
PyObject *PyInit_fmadd()
#endif
{
	PyObject *mod, *dict;

	mod=Py_InitModule3("fmadd", fmadd_methods,
		"fmadd: multiply add, x*y+z in C." );

	FMError= PyErr_NewException( "fmadd.error", NULL, NULL );
	Py_XINCREF(FMError);
	PyModule_AddObject( mod, "error", FMError );
	if( PyErr_Occurred() ){
		PyErr_Print();
	}

	dict=PyModule_GetDict(mod);

	import_array();
#ifdef IS_PY3K
	return mod;
#endif
}
#ifdef __cplusplus
}
#endif
