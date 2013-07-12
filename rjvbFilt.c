/*
 * Python interface to filtering routines
 \ (c) 2005-2010 R.J.V. Bertin
 */
 
static const char modName[] = "rjvbFilt";

#ifdef __CYGWIN__
#	undef _WINDOWS
#	undef WIN32
#	undef MS_WINDOWS
#	undef _MSC_VER
#endif
#if defined(_WINDOWS) || defined(WIN32) || defined(MS_WINDOWS) || defined(_MSC_VER)
#	define MS_WINDOWS
#	define _USE_MATH_DEFINES
#endif

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

#if defined(__GNUC__) && !defined(_GNU_SOURCE)
#	define _GNU_SOURCE
#endif

#include <stdio.h>
#include <math.h>
#include "NaN.h"

#include <errno.h>

#include "rjvbFilters.h"

#ifndef False
#	define False	0
#endif
#ifndef True
#	define True	1
#endif

#ifndef StdErr
#	define StdErr	stderr
#endif

#ifndef CLIP
#	define CLIP(var,low,high)	if((var)<(low)){\
	(var)=(low);\
}else if((var)>(high)){\
	(var)=(high);}
#endif
#define CLIP_EXPR(var,expr,low,high)	{ double l, h; if(((var)=(expr))<(l=(low))){\
	(var)=l;\
}else if((var)>(h=(high))){\
	(var)=h;}}

PyObject *FMError;

#include "ParsedSequence.h"

static PyObject *python_convolve( PyObject *self, PyObject *args )
{ PyObject *dataArg, *maskArg, *ret= NULL;
  ParsedSequences dataSeq, maskSeq;
  int nan_handling = True;
	
	if(!PyArg_ParseTuple(args, "OO|i:convolve", &dataArg, &maskArg, &nan_handling )){
		return NULL;
	}
	dataSeq.type = maskSeq.type = (PSTypes) 0;
	if( !ParseSequence( dataArg, &dataSeq, FMError ) ){
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		PyErr_SetString( FMError, "Error occurred while parsing data argument" );
	}
	if( !ParseSequence( maskArg, &maskSeq, FMError ) ){
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		PyErr_SetString( FMError, "Error occurred while parsing mask argument" );
	}
	if( dataSeq.type && maskSeq.type ){
	  double *output;
		output = convolve( dataSeq.array, dataSeq.N, maskSeq.array, maskSeq.N, nan_handling );
		if( output ){
		  npy_intp dim[1]= {dataSeq.N};
			ret= PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) output );
			((PyArrayObject*)ret)->flags|= NPY_OWNDATA;
		}
	}
	if( dataSeq.type ){
		if( dataSeq.dealloc_array ){
			PyMem_Free(dataSeq.array);
		}
		if( dataSeq.type == ContArray ){
			Py_XDECREF(dataSeq.ContArray);
		}
	}
	if( maskSeq.type ){
		if( maskSeq.dealloc_array ){
			PyMem_Free(maskSeq.array);
		}
		if( maskSeq.type == ContArray ){
			Py_XDECREF(maskSeq.ContArray);
		}
	}
	return ret;
}

static PyObject *python_SavGolayCoeffs( PyObject *self, PyObject *args, PyObject *kw )
{ PyObject *ret = NULL;
  int fw, fo, deriv = 0;
  char *kws[] = { "halfwidth", "order", "deriv", NULL };
  unsigned long N;
  double *coeffs, *output;
	
	if(!PyArg_ParseTupleAndKeywords(args, kw, "ii|i:SavGolayCoeffs", kws, &fw, &fo, &deriv )){
		return NULL;
	}
	if( fw < 0 || fw > (MAXINT-1)/2 ){
		PyErr_Warn( FMError, "halfwidth must be positive and <= (MAXINT-1)/2; clipping" );
		CLIP( fw, 0, (MAXINT-1)/2 );
	}
	if( fo < 0 || fo > 2*fw ){
		PyErr_Warn( FMError, "order must be positive and <= the filter width" );
		  /* Put arbitrary upper limit on the order of the smoothing polynomial	*/
		CLIP( fo, 0, 2* fw );
	}
	if( deriv < -fo || deriv > fo ){
		PyErr_Warn( FMError, "derivative must be between -order and +order" );
		CLIP( deriv, -fo, fo );
	}
	N = fw*2+3;
	errno = 0;
	if( (coeffs= (double*) PyMem_New(double, (N+ 1) ))
	    && (output= (double*) PyMem_New(double, (N+ 1) ))
	){
	  int i;
	  npy_intp dim[1]= {N};
		if( !(fw== 0 && fo== 0 && deriv==0) ){
			if( savgol( &(coeffs)[-1], N, fw, fw, deriv, fo ) ){
					fprintf( StdErr, "coeffs[%lu,%d,%d,%d]={%g", N, fw, deriv, fo, coeffs[0] );
					for( i = 1; i < N; i++ ){
						fprintf( StdErr, ",%g", coeffs[i] );
					}
					fputs( "}\n", StdErr );
				  /* Unwrap the coefficients into the target memory:	*/
				output[N/2] = coeffs[0];
				for( i = 1; i <= N/2; i++ ){
					output[N/2-i]= coeffs[i];
					output[N/2+i]= coeffs[N-i];
				}
			}
			else{
				PyMem_Free(output);
				output = NULL;
			}
		}
		else{
			memset( output, 0, N* sizeof(double) );
		}
		PyMem_Free(coeffs);
		if( output ){
			ret = PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) output );
			((PyArrayObject*)ret)->flags|= NPY_OWNDATA;
		}
	}
	else{
		PyErr_NoMemory();
	}
	return( ret );
}

static PyObject *python_SavGolayGain( PyObject *self, PyObject *args, PyObject *kw )
{ int deriv = 0;
  double delta;
  char *kws[] = { "delta", "deriv", NULL };
	
	if(!PyArg_ParseTupleAndKeywords(args, kw, "d|i:SavGolayGain", kws, &delta, &deriv )){
		return NULL;
	}
// 	return( Py_BuildValue("d", ((deriv>0)? pow(delta,deriv)/deriv : 1) ) );
	return( Py_BuildValue("d", ((deriv>0)? deriv/pow(delta,deriv) : 1) ) );
}

static PyObject *python_SavGolay2DCoeffs( PyObject *self, PyObject *args, PyObject *kw )
{ PyObject *ret = NULL;
  int fw, fo, deriv = 0;
  char *kws[] = { "halfwidth", "order", "deriv", NULL };
  unsigned long N;
  double *output;
	
	if(!PyArg_ParseTupleAndKeywords(args, kw, "ii|i:SavGolay2DCoeffs", kws, &fw, &fo, &deriv )){
		return NULL;
	}
	if( fw < 0 || fw > (MAXINT-1)/2 ){
		PyErr_Warn( FMError, "halfwidth must be positive and <= (MAXINT-1)/2; clipping" );
		CLIP( fw, 0, (MAXINT-1)/2 );
	}
	if( fo < 0 || fo > 2*fw ){
		PyErr_Warn( FMError, "order must be positive and <= the filter width" );
		  /* Put arbitrary upper limit on the order of the smoothing polynomial	*/
		CLIP( fo, 0, 2* fw );
	}
	if( deriv < -fo || deriv > fo ){
		PyErr_Warn( FMError, "derivative must be between -order and +order" );
		CLIP( deriv, -fo, fo );
	}
	N = savgol2D_dim( fw, NULL );
	errno = 0;
	if( (output= (double*) PyMem_New(double, (N*N+ 1) )) ){
	  int i;
	  npy_intp dim[2]= {N, N};
		if( !(fw== 0 && fo== 0 && deriv==0) ){
			if( !savgol2D( output, N*N, fw, deriv, fo ) ){
				PyMem_Free(output);
				output = NULL;
			}
//			if( !altSavGol2D( output, N, N, fo, fo ) ){
//				PyMem_Free(output);
//				output = NULL;
//			}
		}
		else{
			memset( output, 0, N * N * sizeof(double) );
		}
		if( output ){
			ret = PyArray_SimpleNewFromData( 2, dim, PyArray_DOUBLE, (void*) output );
			((PyArrayObject*)ret)->flags|= NPY_OWNDATA;
		}
	}
	else{
		PyErr_NoMemory();
	}
	return( ret );
}

static PyObject *__python_Spline_Resample(int use_PWLInt, PyObject *OrgX, PyObject *OrgY, PyObject *ResampledX,
					int pad, int returnX, int returnY, int returnCoeffs)
{ PyObject *ret = NULL, *ResampledY = NULL, *RetOX = NULL, *RetOY = NULL, *RetCoeffs = NULL;
  ParsedSequences orgXSeq, orgYSeq, resampledXSeq;
	if( !ParseSequence( OrgX, &orgXSeq, FMError ) ){
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		PyErr_SetString( FMError, "Error occurred while parsing orgX argument" );
	}
	if( !ParseSequence( OrgY, &orgYSeq, FMError ) ){
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		PyErr_SetString( FMError, "Error occurred while parsing orgY argument" );
	}
	if( !ParseSequence( ResampledX, &resampledXSeq, FMError ) ){
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		PyErr_SetString( FMError, "Error occurred while parsing ResampledX argument" );
	}
	if( orgXSeq.N != orgYSeq.N ){
		PyErr_SetString( FMError, "orgX and orgY sequences must be of equal length!" );
	}
	else if( orgXSeq.type && orgYSeq.type && resampledXSeq.type ){
	  double *resampledY, *retOX=NULL, *retOY=NULL, *coeffs=NULL, **_retOX=NULL, **_retOY=NULL, **_coeffs=NULL;
		if( returnX ){
			_retOX = &retOX;
		}
		if( returnY ){
			_retOY = &retOY;
		}
		if( returnCoeffs ){
			_coeffs = &coeffs;
		}
		resampledY = SplineResample( use_PWLInt, orgXSeq.array, orgXSeq.N, orgYSeq.array, orgYSeq.N,
					resampledXSeq.array, resampledXSeq.N,
					pad, _retOX, _retOY, _coeffs );
		if( resampledY ){
		  npy_intp dim[1]= {resampledXSeq.N};
			ResampledY= PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) resampledY );
			((PyArrayObject*)ResampledY)->flags|= NPY_OWNDATA;
			if( retOX || retOY || coeffs ){
			  int i, N = 1;
				if( retOX ){
					N += 1;
				}
				if( retOY ){
					N += 1;
				}
				if( coeffs ){
					N += 1;
				}
				if( !(ret = PyTuple_New(N)) ){
					PyErr_NoMemory();
					PyMem_Free(resampledY);
					if( retOX ){
						PyMem_Free(retOX);
					}
					if( retOY ){
						PyMem_Free(retOY);
					}
					if( coeffs ){
						PyMem_Free(coeffs);
					}
				}
				else{
					PyTuple_SetItem(ret, 0, ResampledY );
					i = 1;
					if( retOX ){
						dim[0] = (pad)? orgXSeq.N + 2 : orgXSeq.N;
						RetOX = PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) retOX );
						((PyArrayObject*)RetOX)->flags |= NPY_OWNDATA;
						PyTuple_SetItem(ret, i, RetOX );
						i += 1;
					}
					if( retOY ){
						dim[0] = (pad)? orgYSeq.N + 2 : orgYSeq.N;
						RetOY = PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) retOY );
						((PyArrayObject*)RetOY)->flags |= NPY_OWNDATA;
						PyTuple_SetItem(ret, i, RetOY );
						i += 1;
					}
					if( coeffs ){
						dim[0] = (pad)? orgXSeq.N + 2 : orgXSeq.N;
						RetCoeffs = PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) coeffs );
						((PyArrayObject*)RetCoeffs)->flags |= NPY_OWNDATA;
						PyTuple_SetItem(ret, i, RetCoeffs );
						i += 1;
					}
				}
			}
			else{
				ret = ResampledY;
			}
		}
	}
	if( orgXSeq.type ){
		if( orgXSeq.dealloc_array ){
			PyMem_Free(orgXSeq.array);
		}
		if( orgXSeq.type == ContArray ){
			Py_XDECREF(orgXSeq.ContArray);
		}
	}
	if( orgYSeq.type ){
		if( orgYSeq.dealloc_array ){
			PyMem_Free(orgYSeq.array);
		}
		if( orgYSeq.type == ContArray ){
			Py_XDECREF(orgYSeq.ContArray);
		}
	}
	return( ret );
}

static PyObject *python_Spline_Resample( PyObject *self, PyObject *args, PyObject *kw )
{
  PyObject *OrgX=NULL, *OrgY=NULL, *ResampledX=NULL;
  int pad=1, returnX=0, returnY=0, returnCoeffs=0;
  char *kws[] = { "orgX", "orgY", "resampledX", "pad", "returnX", "returnY", "returnCoeffs", NULL };
	
	if(!PyArg_ParseTupleAndKeywords(args, kw, "OOO|iiii:Spline_Resample", kws,
		&OrgX, &OrgY, &ResampledX, &pad, &returnX, &returnY, &returnCoeffs )
	){
		return NULL;
	}
	else{
		return( __python_Spline_Resample(0, OrgX, OrgY, ResampledX, pad, returnX, returnY, returnCoeffs ) );
	}
}

static PyObject *python_PWLInt_Resample( PyObject *self, PyObject *args, PyObject *kw )
{ PyObject *OrgX=NULL, *OrgY=NULL, *ResampledX=NULL;
  int pad=1, returnX=0, returnY=0, returnCoeffs=0;
  char *kws[] = { "orgX", "orgY", "resampledX", "pad", "returnX", "returnY", "returnCoeffs", NULL };
	
	if(!PyArg_ParseTupleAndKeywords(args, kw, "OOO|iiii:PWLInt_Resample", kws,
		&OrgX, &OrgY, &ResampledX, &pad, &returnX, &returnY, &returnCoeffs )
	){
		return NULL;
	}
	else{
		return( __python_Spline_Resample(1, OrgX, OrgY, ResampledX, pad, returnX, returnY, returnCoeffs ) );
	}
}

static PyObject *python_EulerSum( PyObject *self, PyObject *args, PyObject *kw )
{ PyObject *ret = NULL, *vals = NULL, *t = NULL;
  char *kws[] = { "values", "t", "initialValue", "returnAll", "nanResets", NULL };
  double iVal = 0;
  int returnAll = 0, nanResets = 0;
  ParsedSequences valsSeq, tSeq;

	if(!PyArg_ParseTupleAndKeywords(args, kw, "OO|dii:EulerSum", kws,
		&vals, &t, &iVal, &returnAll, &nanResets)
	){
		return NULL;
	}
	if( !ParseSequence( vals, &valsSeq, FMError ) ){
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		PyErr_SetString( FMError, "Error occurred while parsing values argument" );
	}
	if( !ParseSequence( t, &tSeq, FMError ) ){
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		PyErr_SetString( FMError, "Error occurred while parsing t argument" );
	}
	if( valsSeq.N != tSeq.N ){
		PyErr_SetString( FMError, "values and t sequences must be of equal length!" );
	}
	else if( valsSeq.type && tSeq.type ){
	  double cumsum, *runsum = NULL;
	  extern int ascanf_arg_error;
		if( returnAll ){
			cumsum = EulerArrays( valsSeq.array, valsSeq.N, tSeq.array, tSeq.N, iVal, &runsum, nanResets );
			if( runsum && !ascanf_arg_error ){
			  npy_intp dim[1]= {valsSeq.N};
				ret= PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) runsum );
				((PyArrayObject*)ret)->flags|= NPY_OWNDATA;
			}
		}
		else{
			cumsum = EulerArrays( valsSeq.array, valsSeq.N, tSeq.array, tSeq.N, iVal, NULL, nanResets );
			if( !ascanf_arg_error ){
				ret= Py_BuildValue( "d", cumsum );
			}
		}
	}
	if( valsSeq.type ){
		if( valsSeq.dealloc_array ){
			PyMem_Free(valsSeq.array);
		}
		if( valsSeq.type == ContArray ){
			Py_XDECREF(valsSeq.ContArray);
		}
	}
	if( tSeq.type ){
		if( tSeq.dealloc_array ){
			PyMem_Free(tSeq.array);
		}
		if( tSeq.type == ContArray ){
			Py_XDECREF(tSeq.ContArray);
		}
	}
	return( ret );
}

static PyMethodDef Filt_methods[] =
{
	{ "convolve", (PyCFunction) python_convolve, METH_VARARGS,
		"convolve(Data,Mask[,nan_handling]): convolve the array pointed to by <Data> by <Mask>\n"
		" The result is returned in a numpy.ndarray equal in size to Data\n"
		" Data and Mask must be 1D homogenous, numerical sequences (numpy.ndarray, tuple, list)\n"
		" The <nan_handling> argument indicates what to do with gaps of NaN value(s) in the input:\n"
		" \t1: if possible, pad with the values surrounding the gap (step halfway)\n"
		" \t2: if possible, intrapolate linearly between the surrounding values\n"
		" \t(if not possible, simply pad with the first or last non-NaN value).\n"
		" The estimated values are replaced by the original NaNs after convolution.\n"
		" This routine uses direct (\"brain dead\") convolution.\n"
	},
	{ "SavGolayCoeffs", (PyCFunction) python_SavGolayCoeffs, METH_VARARGS|METH_KEYWORDS,
		"SavGolayCoeffs(halfwidth,order[,deriv=0]]): determine the coefficients\n"
		" for a Savitzky-Golay convolution filter. This returns a mask that can be used for\n"
		" convolution; the wider the filter, the more it smooths using a polynomial of the\n"
		" requested order (the higher the orde, the closer it will follow the input data). The\n"
		" deriv argument specifies an optional derivative that will be calculated during\n"
		" the smoothing; this is generally better than smoothing first and taking the derivative afterwards.\n"
		" After the convolution, the output is scaled by a gain factor; see SavGolayGain.\n"
	},
	{ "SavGolayGain", (PyCFunction) python_SavGolayGain, METH_VARARGS|METH_KEYWORDS,
		"SavGolayGain(delta,deriv): returns the gain factor by which the result of a convolution\n"
		" with a SavGolay mask has to be multiplied (1 for deriv==0). delta is the resolution of\n"
		" the dimension with respect to which the derivative is being taken, e.g. the sampling time interval.\n"
	},
	{ "SavGolay2DCoeffs", (PyCFunction) python_SavGolay2DCoeffs, METH_VARARGS|METH_KEYWORDS,
		"SavGolay2DCoeffs(halfwidth,order[,deriv=0]]): determine the coefficients\n"
		" for a 2D Savitzky-Golay convolution filter. This returns a mask that can be used for\n"
		" convolution; the wider the filter, the more it smooths using a polynomial of the\n"
		" requested order (the higher the orde, the closer it will follow the input data). The\n"
		" deriv argument specifies an optional derivative that will be calculated during\n"
		" the smoothing; this is generally better than smoothing first and taking the derivative afterwards.\n"
		" After the convolution, the output is scaled by a gain factor; see SavGolayGain.\n"
	},
	{ "Spline_Resample", (PyCFunction) python_Spline_Resample, METH_VARARGS|METH_KEYWORDS,
		docSplineResample
	},
	{ "PWLInt_Resample", (PyCFunction) python_PWLInt_Resample, METH_VARARGS|METH_KEYWORDS,
		docSplineResample
	},
	{ "EulerSum", (PyCFunction) python_EulerSum, METH_VARARGS|METH_KEYWORDS,
		docEulerSum
	},
	{ NULL, NULL }
};

#ifndef IS_PY3K
void initrjvbFilt()
#else
PyObject *PyInit_rjvbFilt()
#endif
{ PyObject *mod, *dict;

	import_array();

	mod = Py_InitModule3( modName, Filt_methods,
		"this module provides a number of functions (mostly filters) as they exist in XGraph's ascanf language"
	);

	FMError= PyErr_NewException( "rjvbFilt.error", NULL, NULL );
	Py_XINCREF(FMError);
	PyModule_AddObject( mod, "error", FMError );
	if( PyErr_Occurred() ){
		PyErr_Print();
	}

	dict=PyModule_GetDict(mod);

#ifdef IS_PY3K
	return mod;
#endif
}
