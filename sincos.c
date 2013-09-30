/*
 * Python interface to sincos routines.
 \ (c) 2005-2010 R.J.V. Bertin
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

#ifdef HAVE_VECLIB
#	include <vecLib/vecLib.h>
#endif

#if __GNUC__ >= 3
#	define pragma_likely(x)		__builtin_expect (!!(x),1)
#	define pragma_unlikely(x)	__builtin_expect (!!(x),0)
#else
#	define pragma_likely(x)		(x)
#	define pragma_unlikely(x)	(x)
#endif

#if defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__) || defined(_MSC_VER)

#	if defined(_MSC_VER) || defined(__CYGWIN__)
		extern void sincos( double, double*, double* );

#ifdef __SSE__
#		define USE_SSE2
#		include <xmmintrin.h>
#		include <emmintrin.h>
#		define SSE_MATHFUN_WITH_CODE
#		include "sse_mathfun/sse_mathfun.h"

		void sincos_sse(double x, double *s, double *c )
		{ v4sf xx, ss, cc;
//  			xx = _mm_set_ps1(x);
 			((float*)&xx)[0] = (float) x;
			sincos_ps(xx, &ss, &cc);
			*s = ((float*)&ss)[0];
			*c = ((float*)&cc)[0];
		}
#		define sincos(x,s,c)	sincos_sse((x),(s),(c))
#endif

#	else

#		if defined(__SSE__) || defined(__SSE2__)
#		warning "using sincos_sse!"

#			define USE_SSE2
#			include <xmmintrin.h>
#			include <emmintrin.h>
#			define SSE_MATHFUN_WITH_CODE
#			include "sse_mathfun/sse_mathfun.h"

			void sincos_sse(double x, double *s, double *c )
			{ v4sf xx, ss, cc;
// 				xx = _mm_set_ps1(x);
				((float*)&xx)[0] = (float) x;
				sincos_ps(xx, &ss, &cc);
				*s = ((float*)&ss)[0];
				*c = ((float*)&cc)[0];
			}

#			define sincos(x,s,c)	sincos_sse((x),(s),(c))

#		else

			void sincos_x86_fpu(double x, double *s, double *c )
			{
				asm( "fsincos;" : "=t" (*c), "=u" (*s) : "0" (x) : "st(7)" );
			}

#			define sincos(x,s,c)	sincos_x86_fpu((x),(s),(c))
#		endif

#	endif
#endif

#include <errno.h>

static double M_PI2;

static PyObject *FMError;

    /*DOC*/ static char doc_sincos[] =
    /*DOC*/    "sincos(angle[,base=2PI]) -> (double,double)\n"
    /*DOC*/    "returns sine and cosine of <angle>\n"
#ifdef HAVE_VECLIB
    			"(uses vecLib's vvsincos for contiguous ndarrays - double precision)\n"
#endif
#if defined(_SSE_MATHFUN_H)
    			"(uses SSE implementation 'sincos_ps' from sse_mathfun.h - float precision)\n"
#endif
    /*DOC*/ ;

static PyObject *sincos_sincos( PyObject *self, PyObject *args )
{ double angle, base= 0;
  double s, c;
	
	if(!PyArg_ParseTuple(args, "d|d:sincos", &angle, &base )){
		return NULL;
	}
	if( base && base != M_PI2 ){
		angle*= M_PI2/base;
	}
#if defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__) || defined(_MSC_VER)
	sincos( angle, &s, &c );
#else
	s= sin( angle );
	c= cos( angle );
#endif
	return Py_BuildValue( "(dd)", s, c );
}

static PyObject *sincos_sincos2( PyObject *self, PyObject *args )
{ double angle, base= 0;
  PyObject *angles;
  int isList= 0;
	
	if(!PyArg_ParseTuple(args, "O|d:sincos", &angles, &base )){
		return NULL;
	}

	if( PyList_Check(angles) ){
		if( !(angles= PyList_AsTuple(angles)) ){
			PyErr_SetString( FMError, "Unexpected failure converting angles list to tuple" );
			return(NULL);
		}
		isList= 1;
	}

	if( base && base != M_PI2 ){
		base = M_PI2/base;
	}

	if( PyTuple_Check(angles) ){
	  long i, N= PyTuple_Size(angles);
	  PyObject *sins= PyList_New(N);
	  PyObject *coss= PyList_New(N), *ret= NULL;
		if( sins && coss ){
			for( i= 0; i< N; i++ ){
			  double s, c;
				angle= PyFloat_AsDouble( PyTuple_GetItem(angles,i) );

				if( base && base != 1 ){
					angle*= base;
				}
				if( PyErr_Occurred() ){
					goto SC2t_ESCAPE;
				}
#if defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__) || defined(_MSC_VER)
				sincos( angle, &s, &c );
#else
				c= cos( angle ), s= sin( angle );
#endif
				PyList_SetItem( sins, i, PyFloat_FromDouble(s) );
				PyList_SetItem( coss, i, PyFloat_FromDouble(c) );
			}
			if( !isList ){
			  PyObject *r;
				if( (r= PyList_AsTuple(sins)) ){
					sins= r;
				}
				if( (r= PyList_AsTuple(coss)) ){
					coss= r;
				}
			}
			ret= Py_BuildValue( "(OO)", sins, coss );
		}
		else{
			PyErr_NoMemory();
		}
SC2t_ESCAPE:;
		Py_XDECREF(sins);
		Py_XDECREF(coss);
		return(ret);
	}
	else if( PyArray_Check(angles) ){
	  long i, N= PyArray_Size(angles), N_1 = N-1;
	  double *PyArrayBuf= NULL, *sins= PyMem_New( double, N);
	  double *coss= PyMem_New( double, N);
	  PyObject *r1= NULL, *r2= NULL, *ret= NULL;
		if( sins && coss ){
		  npy_intp dims[2]= {0,1};
		  PyArrayObject *parray= (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) angles, PyArray_DOUBLE, 0,0 );
		  PyArrayIterObject *it= NULL;
		  size_t stride = 1;
#ifdef _SSE_MATHFUN_H
#	if defined(USE_SINCOS_PD) || defined(HAVE_VECLIB)
		  v2df va, vs, vc, vbase;
		  double *v2;
		  int nn;
#	else
 		  v4sf va, vs, vc, vbase;
 		  float *v4;
#	endif
#endif
#ifdef _SSE_MATHFUN_H
#	if defined(USE_SINCOS_PD) || defined(HAVE_VECLIB)
			vbase = (v2df) _mm_set1_pd(base);
#	else
 		  	vbase = _mm_set_ps1(base);
#	endif
#endif
			if( parray ){
				PyArrayBuf = (double*) PyArray_DATA(parray);
#ifdef _SSE_MATHFUN_H
				if( PyArrayBuf ){
#	if defined(USE_SINCOS_PD) || defined(HAVE_VECLIB)
					stride = 2;
#	else
 					stride = 4;
#	endif
				}
#endif
			}
			else{
				parray= (PyArrayObject*) angles;
				it= (PyArrayIterObject*) PyArray_IterNew(angles);
			}
			dims[0]= N;
#ifdef HAVE_VECLIB
			if( PyArrayBuf && (base == 0 || base == 1) ){
			  int nn = N;
				vvsincos( sins, coss, PyArrayBuf, &nn );
				N = 0;
			}
#endif
			for( i= 0; i< N; i += stride ){
				if( PyArrayBuf ){
#ifdef _SSE_MATHFUN_H
#	if defined(USE_SINCOS_PD) || defined(HAVE_VECLIB)
					v2 = (double*) &va;
					if( pragma_unlikely(i==N_1) ){
						nn = 1;
						va = (v2df) _mm_set1_pd( PyArrayBuf[i] );
					}
					else{
						nn = 2;
						va = _MM_SET_PD( PyArrayBuf[i], PyArrayBuf[i+1] );
					}
#	else
					// NB: we ought to be checking for boundaries here too!!
 					v4 = (float*) &va;
// 					v4[0] = (float) PyArrayBuf[i], v4[1] = (float) PyArrayBuf[i+1],
// 						v4[2] = (float) PyArrayBuf[i+2], v4[3] = (float) PyArrayBuf[i+3];
 					va = _MM_SET_PS( (float) PyArrayBuf[i], (float) PyArrayBuf[i+1],
 							(float) PyArrayBuf[i+2], (float) PyArrayBuf[i+3] );
#	endif
#else
					angle= PyArrayBuf[i];
#endif
				}
				else{
					angle= PyFloat_AsDouble( PyArray_DESCR(parray)->f->getitem( it->dataptr, angles ) );
				}

				if( base && base != 1 ){
#ifdef _SSE_MATHFUN_H
					if( PyArrayBuf ){
#	if defined(USE_SINCOS_PD) || defined(HAVE_VECLIB)
						va = _mm_mul_pd( va, vbase );
#	else
 						va = _mm_mul_ps( va, vbase );
#	endif
					}
					else{
						angle*= base;
					}
#else
					angle*= base;
#endif
				}
				if( PyErr_Occurred() ){
					if( it ){
						Py_DECREF(it);
					}
					else if( parray ){
						Py_DECREF(parray);
					}
					goto SC2a_ESCAPE;
				}
#ifdef _SSE_MATHFUN_H
				if( PyArrayBuf ){
#	ifdef USE_SINCOS_PD
					sincos_pd( va, &vs, &vc );
					v2 = &vs;
					sins[1] = ((double*)&vs)[0];
					coss[i] = ((double*)&vs)[0];
					if( pragma_likely(i!=N_1) ){
						sins[i+1] = ((double*)&vs)[1];
						coss[i+1] = ((double*)&vs)[1];
					}
#	elif defined(HAVE_VECLIB)
					vvsincos( &sins[i], &coss[i], v2, &nn );
#	else
 					sincos_ps( va, &vs, &vc );
 					v4 = &vs;
 					sins[i] = v4[0], sins[i+1] = v4[1], sins[i+2] = v4[2], sins[i+3] = v4[3];
 					v4 = &vc;
 					coss[i] = v4[0], coss[i+1] = v4[1], coss[i+2] = v4[2], coss[i+3] = v4[3];
#	endif
				}
				else{
					sincos( angle, &sins[i], &coss[i] );
				}
#elif defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__) || defined(_MSC_VER)
				sincos( angle, &sins[i], &coss[i] );
#else
				sins[i]= sin( angle );
				coss[i]= cos( angle );
#endif
				if( it ){
					PyArray_ITER_NEXT(it);
				}
			}
// 			r1= PyArray_FromDimsAndData( 1, dims, PyArray_DOUBLE, (char*) sins );
// 			r2= PyArray_FromDimsAndData( 1, dims, PyArray_DOUBLE, (char*) coss );
			r1= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) sins );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r1, NPY_OWNDATA );
			r2= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) coss );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r2, NPY_OWNDATA );
			ret= Py_BuildValue( "(OO)", r1, r2 );
			Py_XDECREF(r1);
			Py_XDECREF(r2);
			if( it ){
				Py_DECREF(it);
			}
			else if( parray ){
				Py_DECREF(parray);
			}
			return(ret);
		}
		else{
			PyErr_NoMemory();
			goto SC2a_ESCAPE;
		}
SC2a_ESCAPE:;
		if( sins ){
			PyMem_Free(sins);
		}
		if( coss ){
			PyMem_Free(coss);
		}
		return(NULL);
	}
	else{
	  double s, c;
		if( base && base != 1 ){
			angle= PyFloat_AsDouble(angles) * base;
		}
		else{
			angle= PyFloat_AsDouble(angles);
		}
		if( PyErr_Occurred() ){
			return(NULL);
		}
#if defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__) || defined(_MSC_VER)
		sincos( angle, &s, &c );
#else
		c= cos( angle ), s= sin( angle );
#endif
		return Py_BuildValue( "(dd)", s, c );
	}
}

extern	void	mips_sincos(double, double *, double *);

static PyObject *sincos_mipssincos( PyObject *self, PyObject *args )
{ double angle, base= 0;
  PyObject *angles;
  int isList= 0;
	
	if(!PyArg_ParseTuple(args, "O|d:mips_sincos", &angles, &base )){
		return NULL;
	}

	if( PyList_Check(angles) ){
		if( !(angles= PyList_AsTuple(angles)) ){
			PyErr_SetString( FMError, "Unexpected failure converting angles list to tuple" );
			return(NULL);
		}
		isList= 1;
	}

	if( base && base != M_PI2 ){
		base = M_PI2/base;
	}

	if( PyTuple_Check(angles) ){
	  long i, N= PyTuple_Size(angles);
	  PyObject *sins= PyList_New(N);
	  PyObject *coss= PyList_New(N), *ret= NULL;
		if( sins && coss ){
			for( i= 0; i< N; i++ ){
			  double s, c;
				angle= PyFloat_AsDouble( PyTuple_GetItem(angles,i) );

				if( base && base != 1 ){
					angle*= base;
				}
				if( PyErr_Occurred() ){
					goto SC2t_ESCAPE;
				}
				mips_sincos( angle, &s, &c );
				PyList_SetItem( sins, i, PyFloat_FromDouble(s) );
				PyList_SetItem( coss, i, PyFloat_FromDouble(c) );
			}
			if( !isList ){
			  PyObject *r;
				if( (r= PyList_AsTuple(sins)) ){
					sins= r;
				}
				if( (r= PyList_AsTuple(coss)) ){
					coss= r;
				}
			}
			ret= Py_BuildValue( "(OO)", sins, coss );
		}
		else{
			PyErr_NoMemory();
		}
SC2t_ESCAPE:;
		Py_XDECREF(sins);
		Py_XDECREF(coss);
		return(ret);
	}
	else if( PyArray_Check(angles) ){
	  long i, N= PyArray_Size(angles);
	  double *PyArrayBuf= NULL, *sins= PyMem_New( double, N);
	  double *coss= PyMem_New( double, N);
	  PyObject *r1= NULL, *r2= NULL, *ret= NULL;
		if( sins && coss ){
		  npy_intp dims[2]= {0,1};
		  PyArrayObject *parray= (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) angles, PyArray_DOUBLE, 0,0 );
		  PyArrayIterObject *it= NULL;
			if( parray ){
				PyArrayBuf= (double*) PyArray_DATA(parray);
			}
			else{
				parray= (PyArrayObject*) angles;
				it= (PyArrayIterObject*) PyArray_IterNew(angles);
			}
			dims[0]= N;
			for( i= 0; i< N; i++ ){
				if( PyArrayBuf ){
					angle= PyArrayBuf[i];
				}
				else{
					angle= PyFloat_AsDouble( PyArray_DESCR(parray)->f->getitem( it->dataptr, angles ) );
				}

				if( base && base != 1 ){
					angle*= base;
				}
				if( PyErr_Occurred() ){
					if( it ){
						Py_DECREF(it);
					}
					else if( parray ){
						Py_DECREF(parray);
					}
					goto SC2a_ESCAPE;
				}
				mips_sincos( angle, &sins[i], &coss[i] );
				if( it ){
					PyArray_ITER_NEXT(it);
				}
			}
// 			r1= PyArray_FromDimsAndData( 1, dims, PyArray_DOUBLE, (char*) sins );
// 			r2= PyArray_FromDimsAndData( 1, dims, PyArray_DOUBLE, (char*) coss );
			r1= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) sins );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r1, NPY_OWNDATA );
			r2= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) coss );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r2, NPY_OWNDATA );
			ret= Py_BuildValue( "(OO)", r1, r2 );
			Py_XDECREF(r1);
			Py_XDECREF(r2);
			if( it ){
				Py_DECREF(it);
			}
			else if( parray ){
				Py_DECREF(parray);
			}
			return(ret);
		}
		else{
			PyErr_NoMemory();
			goto SC2a_ESCAPE;
		}
SC2a_ESCAPE:;
		if( sins ){
			PyMem_Free(sins);
		}
		if( coss ){
			PyMem_Free(coss);
		}
		return(NULL);
	}
	else{
	  double s, c;
		if( base && base != 1 ){
			angle= PyFloat_AsDouble(angles) * base;
		}
		else{
			angle= PyFloat_AsDouble(angles);
		}
		if( PyErr_Occurred() ){
			return(NULL);
		}
		mips_sincos( angle, &s, &c );
		return Py_BuildValue( "(dd)", s, c );
	}
}

#if 0
/* Copyright (C) 2004, 2005, 2006 Free Software Foundation, Inc. */
/* This file is part of GNU Modula-2.

GNU Modula-2 is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2, or (at your option) any later
version.

GNU Modula-2 is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.   See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License along
with gm2; see the file COPYING.   If not, write to the Free Software
Foundation, 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

This file was originally part of the University of Ulm library
*/



/* Ulm's Modula-2 Library
    Copyright (C) 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991,
    1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001,
    2002, 2003, 2004, 2005
    by University of Ulm, SAI, D-89069 Ulm, Germany
*/
#endif

#include "NaN.h"

static
#ifdef __GNUC__
inline
#endif
double CopySign( double x, double y )
{
	if( x< 0 ){
		x= -x;
	}
	return( (y<0)? -x : x );
}

static
#ifdef __GNUC__
inline
#endif
int Finite( double x )
{
	return( !NaNorInf(x) );
}

#define half	0.5

#define	thresh	2.6117239648121182150E-1
#define	PIo4	7.8539816339744827900E-1
#define	PIo2	1.5707963267948965580E0
#define	PI3o4	2.3561944901923448370E0
#define	PI	3.1415926535897931160E0
#define	PI2	6.2831853071795862320E0

#define	S0	-1.6666666666666463126E-1
#define	S1	8.3333333332992771264E-3
#define	S2	-1.9841269816180999116E-4
#define	S3	2.7557309793219876880E-6
#define	S4	-2.5050225177523807003E-8
#define	S5	1.5868926979889205164E-10

#define	C0	4.1666666666666504759E-2
#define	C1	-1.3888888888865301516E-3
#define	C2	2.4801587269650015769E-5
#define	C3	-2.7557304623183959811E-7
#define	C4	2.0873958177697780076E-9
#define	C5	-1.1250289076471311557E-11

#define	small	1.2E-8
#define	big	1.0E20

static
#ifdef __GNUC__
inline
#endif
double SinS( double x )
{
#if 0
	/* STATIC KERNEL FUNCTION OF SIN(X), COS(X), AND TAN(X)
	 * CODED IN C BY K.C. NG, 1/21/85;
	 * REVISED BY K.C. NG on 8/13/85.
	 *
	 *   		sin(x*k) - x
	 * RETURN	--------------- on [-PI/4,PI/4] ,
	 *   				x
	 *
	 * where k=pi/PI, PI is the rounded
	 * value of pi in machine precision:
	 *
	 *   	Decimal:
	 *   			pi = 3.141592653589793 23846264338327 .....
	 *      53 bits	PI = 3.141592653589793 115997963 ..... ,
	 *      56 bits	PI = 3.141592653589793 227020265 ..... ,
	 *
	 *   	Hexadecimal:
	 *   			pi = 3.243F6A8885A308D313198A2E....
	 *      53 bits	PI = 3.243F6A8885A30   =   2 * 1.921FB54442D18
	 *      56 bits	PI = 3.243F6A8885A308 =   4 * .C90FDAA22168C2
	 *
	 * Method:
	 *   	1. Let z=x*x. Create a polynomial approximation to
	 *   		(sin(k*x)-x)/x   =	z*(S0 + S1*z^1 + ... + S5*z^5).
	 *   	Then
	 *   	sin__S(x*x) = z*(S0 + S1*z^1 + ... + S5*z^5)
	 *
	 *   	The coefficient S's are obtained by a special Remez algorithm.
	 *
	 * Accuracy:
	 *   	In the absence of rounding error, the approximation has absolute error
	 *   	less than 2**(-61.11) for VAX D FORMAT, 2**(-57.45) for IEEE DOUBLE.
		 */
	 BEGIN
	RETURN x*(S0+x*(S1+x*(S2+x*(S3+x*(S4+x*S5)))))
	 END SinS;
#endif
	return( x*(S0+x*(S1+x*(S2+x*(S3+x*(S4+x*S5))))) );
}

static
#ifdef __GNUC__
inline
#endif
double CosC( double x )
{
#if 0
	/*
	 * STATIC KERNEL FUNCTION OF SIN(X), COS(X), AND TAN(X)
	 * CODED IN C BY K.C. NG, 1/21/85;
	 * REVISED BY K.C. NG on 8/13/85.
	 *
	 *   						x*x
	 * RETURN	cos(k*x) - 1 + ----- on [-PI/4,PI/4],	where k = pi/PI,
	 *   						 2
	 * PI is the rounded value of pi in machine precision :
	 *
	 *   	Decimal:
	 *   			pi = 3.141592653589793 23846264338327 .....
	 *      53 bits	PI = 3.141592653589793 115997963 ..... ,
	 *      56 bits	PI = 3.141592653589793 227020265 ..... ,
	 *
	 *   	Hexadecimal:
	 *   			pi = 3.243F6A8885A308D313198A2E....
	 *      53 bits	PI = 3.243F6A8885A30   =   2 * 1.921FB54442D18
	 *      56 bits	PI = 3.243F6A8885A308 =   4 * .C90FDAA22168C2
	 *
	 *
	 * Method:
	 *   	1. Let z=x*x. Create a polynomial approximation to
	 *   		cos(k*x)-1+z/2   =	z*z*(C0 + C1*z^1 + ... + C5*z^5)
	 *   	then
	 *   	cos__C(z) =   z*z*(C0 + C1*z^1 + ... + C5*z^5)
	 *
	 *   	The coefficient C's are obtained by a special Remez algorithm.
	 *
	 * Accuracy:
	 *   	In the absence of rounding error, the approximation has absolute error
	 *   	less than 2**(-64) for VAX D FORMAT, 2**(-58.3) for IEEE DOUBLE.
	 */
	 BEGIN
	RETURN x*x*(C0+x*(C1+x*(C2+x*(C3+x*(C4+x*C5)))))
	 END CosC;
#endif
	return( x*x*(C0+x*(C1+x*(C2+x*(C3+x*(C4+x*C5))))) );
}

static
#ifdef __GNUC__
inline
#endif
double cxsin( double x, double base )
{ double a, c, z, hbase;
	if( !Finite(x) ){
		return( x - x );
	}
	hbase= base * 0.5;

	x= fmod(x * PI2/base, base); /* reduce x into [-base/2,base/2] */
	a = CopySign(x, 1.0);
	if( a >= PIo4 ){
		if( a >= PI3o4 ){ /* ... in [3PI/4,PI] */
		   a = PI - a;
		   x = CopySign(a, x);
		}
		else{ /* ... in [PI/4,3PI/4] */
		   a = PIo2 - a; /* rtn. sign(x)*C(PI/2-|x|) */
		   z = a * a;
		   c = CosC(z);
		   z *= half;
		   if( z >= thresh ){
		   	a = half - ((z - half) - c);
		   }
		   else{
		   	a = 1.0 - (z - c);
		   }
		   return CopySign(a, x);
		}
	}

	if( a < small ){ /* return S(x) */
		// tmp = big + a;
		return x;
	}

	return( x + x * SinS(x * x) );
}

static
#ifdef __GNUC__
inline
#endif
double cxcos( double x, double base )
{ double a, c, z, s= 1.0, hbase;
	if( !Finite(x) ){
		return( x - x );
	}
	hbase= base * 0.5;
	
	x= fmod(x * PI2/base, base); /* reduce x into [-base/2,base/2] */
	a = CopySign(x, 1.0);
	if( a >= PIo4 ){
		if( a >= PI3o4 ){ /* ... in [3PI/4,PI] */
		   a = PI - a;
		   s = - 1.0;
		}
		else{			   /* ... in [PI/4,3PI/4] */
		   a = PIo2 - a;
		   return( a + a * SinS(a * a) ); /* rtn. S(PI/2-|x|) */
		}
	}
	if( a < small ){
		// tmp = big + a;
		return( s );/* rtn. s*C(a) */
	}

	z = a * a;
	c = CosC(z);
	z *= half;
	if( z >= thresh ){
		a = half - ((z - half) - c);
	}
	else{
		a = 1.0 - (z - c);
	}
	return( CopySign(a, s) );
}

static
#ifdef __GNUC__
inline
#endif
void cxsincos( double x, double base, double *sr, double *cr )
{ double a, c, z, s= 1, sx;
	if( !Finite(x) ){
		*sr = *cr= ( x - x );
		return;
	}

	x= sx= fmod(x * PI2/base, base); /* reduce x into [-base,base] */
	a = CopySign(x, 1.0);
	if( a >= PIo4 ){
		if( a >= PI3o4 ){ /* ... in [3PI/4,PI] */
			a = PI - a;
			sx = CopySign(a, x);
			s = -1;
		}
		else{ /* ... in [PI/4,3PI/4] */
			a = PIo2 - a; /* rtn. sign(x)*C(PI/2-|x|) */
			*cr = a + a * SinS(a * a);
			z = a * a;
			c = CosC(z);
			z *= half;
			if( z >= thresh ){
				a = half - ((z - half) - c);
			}
			else{
				a = 1.0 - (z - c);
			}
			*sr= CopySign(a, sx);
			return;
		}
	}

	if( a < small ){ /* return S(x) */
		// tmp = big + a;
		*sr=  x;
		*cr= 1;
		return;
	}

	*sr= ( sx + sx * SinS(sx * sx) );
	z = a * a;
	c = CosC(z);
	z *= half;
	if( z >= thresh ){
		a = half - ((z - half) - c);
	}
	else{
		a = 1.0 - (z - c);
	}
	*cr= CopySign(a, s);
	return;
}


static PyObject *sincos_cxsin( PyObject *self, PyObject *args )
{ double angle, base= PI2;
  PyObject *angles;
  int isList= 0;
	
	if(!PyArg_ParseTuple(args, "O|d:cxsin", &angles, &base )){
		return NULL;
	}

	if( PyList_Check(angles) ){
		if( !(angles= PyList_AsTuple(angles)) ){
			PyErr_SetString( FMError, "Unexpected failure converting angles list to tuple" );
			return(NULL);
		}
		isList= 1;
	}

	if( PyTuple_Check(angles) ){
	  long i, N= PyTuple_Size(angles);
	  PyObject *sins= PyList_New(N);
	  PyObject *ret= NULL;
		if( sins ){
			for( i= 0; i< N; i++ ){
			  double s, c;
				angle= PyFloat_AsDouble( PyTuple_GetItem(angles,i) );

				if( PyErr_Occurred() ){
					goto SC2t_ESCAPE;
				}
				s= cxsin( angle, base );
				PyList_SetItem( sins, i, PyFloat_FromDouble(s) );
			}
			if( !isList ){
			  PyObject *r;
				if( (r= PyList_AsTuple(sins)) ){
					sins= r;
				}
			}
			ret= Py_BuildValue( "O", sins );
		}
		else{
			PyErr_NoMemory();
		}
SC2t_ESCAPE:;
		Py_XDECREF(sins);
		return(ret);
	}
	else if( PyArray_Check(angles) ){
	  long i, N= PyArray_Size(angles);
	  double *PyArrayBuf= NULL, *sins= PyMem_New( double, N);
	  PyObject *r1= NULL, *ret= NULL;
		if( sins ){
		  npy_intp dims[2]= {0,1};
		  PyArrayObject *parray= (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) angles, PyArray_DOUBLE, 0,0 );
		  PyArrayIterObject *it= NULL;
			if( parray ){
				PyArrayBuf= (double*) PyArray_DATA(parray);
			}
			else{
				parray= (PyArrayObject*) angles;
				it= (PyArrayIterObject*) PyArray_IterNew(angles);
			}
			dims[0]= N;
			for( i= 0; i< N; i++ ){
				if( PyArrayBuf ){
					angle= PyArrayBuf[i];
				}
				else{
					angle= PyFloat_AsDouble( PyArray_DESCR(parray)->f->getitem( it->dataptr, angles ) );
				}

				if( PyErr_Occurred() ){
					if( it ){
						Py_DECREF(it);
					}
					else if( parray ){
						Py_DECREF(parray);
					}
					goto SC2a_ESCAPE;
				}
				sins[i]= cxsin( angle, base );
				if( it ){
					PyArray_ITER_NEXT(it);
				}
			}
// 			r1= PyArray_FromDimsAndData( 1, dims, PyArray_DOUBLE, (char*) sins );
			r1= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) sins );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r1, NPY_OWNDATA );
			ret= Py_BuildValue( "O", r1 );
			Py_XDECREF(r1);
			if( it ){
				Py_DECREF(it);
			}
			else if( parray ){
				Py_DECREF(parray);
			}
			return(ret);
		}
		else{
			PyErr_NoMemory();
			goto SC2a_ESCAPE;
		}
SC2a_ESCAPE:;
		if( sins ){
			PyMem_Free(sins);
		}
		return(NULL);
	}
	else{
	  double s;
		angle= PyFloat_AsDouble(angles);
		if( PyErr_Occurred() ){
			return(NULL);
		}
		s= cxsin( angle, base );
		return Py_BuildValue( "d", s );
	}
}

static PyObject *sincos_cxcos( PyObject *self, PyObject *args )
{ double angle, base= PI2;
  double c;
  PyObject *angles;
  int isList= 0;
	
	if(!PyArg_ParseTuple(args, "O|d:cxcos", &angles, &base )){
		return NULL;
	}

	if( PyList_Check(angles) ){
		if( !(angles= PyList_AsTuple(angles)) ){
			PyErr_SetString( FMError, "Unexpected failure converting angles list to tuple" );
			return(NULL);
		}
		isList= 1;
	}

	if( PyTuple_Check(angles) ){
	  long i, N= PyTuple_Size(angles);
	  PyObject *coss= PyList_New(N);
	  PyObject *ret= NULL;
		if( coss ){
			for( i= 0; i< N; i++ ){
			  double c;
				angle= PyFloat_AsDouble( PyTuple_GetItem(angles,i) );

				if( PyErr_Occurred() ){
					goto SC2t_ESCAPE;
				}
				c= cxcos( angle, base );
				PyList_SetItem( coss, i, PyFloat_FromDouble(c) );
			}
			if( !isList ){
			  PyObject *r;
				if( (r= PyList_AsTuple(coss)) ){
					coss= r;
				}
			}
			ret= Py_BuildValue( "O", coss );
		}
		else{
			PyErr_NoMemory();
		}
SC2t_ESCAPE:;
		Py_XDECREF(coss);
		return(ret);
	}
	else if( PyArray_Check(angles) ){
	  long i, N= PyArray_Size(angles);
	  double *PyArrayBuf= NULL, *coss= PyMem_New( double, N);
	  PyObject *r1= NULL, *ret= NULL;
		if( coss ){
		  npy_intp dims[2]= {0,1};
		  PyArrayObject *parray= (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) angles, PyArray_DOUBLE, 0,0 );
		  PyArrayIterObject *it= NULL;
			if( parray ){
				PyArrayBuf= (double*) PyArray_DATA(parray);
			}
			else{
				parray= (PyArrayObject*) angles;
				it= (PyArrayIterObject*) PyArray_IterNew(angles);
			}
			dims[0]= N;
			for( i= 0; i< N; i++ ){
				if( PyArrayBuf ){
					angle= PyArrayBuf[i];
				}
				else{
					angle= PyFloat_AsDouble( PyArray_DESCR(parray)->f->getitem( it->dataptr, angles ) );
				}

				if( PyErr_Occurred() ){
					if( it ){
						Py_DECREF(it);
					}
					else if( parray ){
						Py_DECREF(parray);
					}
					goto SC2a_ESCAPE;
				}
				coss[i]= cxcos( angle, base );
				if( it ){
					PyArray_ITER_NEXT(it);
				}
			}
// 			r1= PyArray_FromDimsAndData( 1, dims, PyArray_DOUBLE, (char*) coss );
			r1= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) coss );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r1, NPY_OWNDATA );
			ret= Py_BuildValue( "O", r1 );
			Py_XDECREF(r1);
			if( it ){
				Py_DECREF(it);
			}
			else if( parray ){
				Py_DECREF(parray);
			}
			return(ret);
		}
		else{
			PyErr_NoMemory();
			goto SC2a_ESCAPE;
		}
SC2a_ESCAPE:;
		if( coss ){
			PyMem_Free(coss);
		}
		return(NULL);
	}
	else{
		angle= PyFloat_AsDouble(angles);
		if( PyErr_Occurred() ){
			return(NULL);
		}
		c= cxcos( angle, base );
		return Py_BuildValue( "d", c );
	}
}

static PyObject *sincos_cxsincos( PyObject *self, PyObject *args )
{ double angle, base= PI2;
  PyObject *angles;
  int isList= 0;
	
	if(!PyArg_ParseTuple(args, "O|d:cxsincos", &angles, &base )){
		return NULL;
	}

	if( PyList_Check(angles) ){
		if( !(angles= PyList_AsTuple(angles)) ){
			PyErr_SetString( FMError, "Unexpected failure converting angles list to tuple" );
			return(NULL);
		}
		isList= 1;
	}

	if( PyTuple_Check(angles) ){
	  long i, N= PyTuple_Size(angles);
	  PyObject *sins= PyList_New(N);
	  PyObject *coss= PyList_New(N), *ret= NULL;
		if( sins && coss ){
			for( i= 0; i< N; i++ ){
			  double s, c;
				angle= PyFloat_AsDouble( PyTuple_GetItem(angles,i) );

				if( PyErr_Occurred() ){
					goto SC2t_ESCAPE;
				}
				cxsincos( angle, base, &s, &c );
				PyList_SetItem( sins, i, PyFloat_FromDouble(s) );
				PyList_SetItem( coss, i, PyFloat_FromDouble(c) );
			}
			if( !isList ){
			  PyObject *r;
				if( (r= PyList_AsTuple(sins)) ){
					sins= r;
				}
				if( (r= PyList_AsTuple(coss)) ){
					coss= r;
				}
			}
			ret= Py_BuildValue( "(OO)", sins, coss );
		}
		else{
			PyErr_NoMemory();
		}
SC2t_ESCAPE:;
		Py_XDECREF(sins);
		Py_XDECREF(coss);
		return(ret);
	}
	else if( PyArray_Check(angles) ){
	  long i, N= PyArray_Size(angles);
	  double *PyArrayBuf= NULL, *sins= PyMem_New( double, N);
	  double *coss= PyMem_New( double, N);
	  PyObject *r1= NULL, *r2= NULL, *ret= NULL;
		if( sins && coss ){
		  npy_intp dims[2]= {0,1};
		  PyArrayObject *parray= (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) angles, PyArray_DOUBLE, 0,0 );
		  PyArrayIterObject *it= NULL;
			if( parray ){
				PyArrayBuf= (double*) PyArray_DATA(parray);
			}
			else{
				parray= (PyArrayObject*) angles;
				it= (PyArrayIterObject*) PyArray_IterNew(angles);
			}
			dims[0]= N;
			for( i= 0; i< N; i++ ){
				if( PyArrayBuf ){
					angle= PyArrayBuf[i];
				}
				else{
					angle= PyFloat_AsDouble( PyArray_DESCR(parray)->f->getitem( it->dataptr, angles ) );
				}

				if( PyErr_Occurred() ){
					if( it ){
						Py_DECREF(it);
					}
					else if( parray ){
						Py_DECREF(parray);
					}
					goto SC2a_ESCAPE;
				}
				cxsincos( angle, base, &sins[i], &coss[i] );
				if( it ){
					PyArray_ITER_NEXT(it);
				}
			}
// 			r1= PyArray_FromDimsAndData( 1, dims, PyArray_DOUBLE, (char*) sins );
// 			r2= PyArray_FromDimsAndData( 1, dims, PyArray_DOUBLE, (char*) coss );
			r1= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) sins );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r1, NPY_OWNDATA );
			r2= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) coss );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r2, NPY_OWNDATA );
			ret= Py_BuildValue( "(OO)", r1, r2 );
			Py_XDECREF(r1);
			Py_XDECREF(r2);
			if( it ){
				Py_DECREF(it);
			}
			else if( parray ){
				Py_DECREF(parray);
			}
			return(ret);
		}
		else{
			PyErr_NoMemory();
			goto SC2a_ESCAPE;
		}
SC2a_ESCAPE:;
		if( sins ){
			PyMem_Free(sins);
		}
		if( coss ){
			PyMem_Free(coss);
		}
		return(NULL);
	}
	else{
	  double s, c;
		angle= PyFloat_AsDouble(angles);
		if( PyErr_Occurred() ){
			return(NULL);
		}
		cxsincos( angle, base, &s, &c );
		return Py_BuildValue( "(dd)", s, c );
	}
}

extern void cephes_sincos(double, double*, double*, int );

static PyObject *sincos_cephes_sincos( PyObject *self, PyObject *args )
{ double angle, base= 360;
  PyObject *angles;
  int isList= 0, flg=0;
	
	if(!PyArg_ParseTuple(args, "Oi|d:cephes_sincos", &angles, &flg, &base )){
		return NULL;
	}

	if( PyList_Check(angles) ){
		if( !(angles= PyList_AsTuple(angles)) ){
			PyErr_SetString( FMError, "Unexpected failure converting angles list to tuple" );
			return(NULL);
		}
		isList= 1;
	}

	if( base && base != 360.0 ){
		base = 360.0/base;
	}

	if( PyTuple_Check(angles) ){
	  long i, N= PyTuple_Size(angles);
	  PyObject *sins= PyList_New(N);
	  PyObject *coss= PyList_New(N), *ret= NULL;
		if( sins && coss ){
			for( i= 0; i< N; i++ ){
			  double s, c;
				angle= PyFloat_AsDouble( PyTuple_GetItem(angles,i) );

				if( base && base != 1 ){
					angle*= base;
				}
				if( PyErr_Occurred() ){
					goto SC2t_ESCAPE;
				}
				cephes_sincos( angle, &s, &c, flg );
				PyList_SetItem( sins, i, PyFloat_FromDouble(s) );
				PyList_SetItem( coss, i, PyFloat_FromDouble(c) );
			}
			if( !isList ){
			  PyObject *r;
				if( (r= PyList_AsTuple(sins)) ){
					sins= r;
				}
				if( (r= PyList_AsTuple(coss)) ){
					coss= r;
				}
			}
			ret= Py_BuildValue( "(OO)", sins, coss );
		}
		else{
			PyErr_NoMemory();
		}
SC2t_ESCAPE:;
		Py_XDECREF(sins);
		Py_XDECREF(coss);
		return(ret);
	}
	else if( PyArray_Check(angles) ){
	  long i, N= PyArray_Size(angles);
	  double *PyArrayBuf= NULL, *sins= PyMem_New( double, N);
	  double *coss= PyMem_New( double, N);
	  PyObject *r1= NULL, *r2= NULL, *ret= NULL;
		if( sins && coss ){
		  npy_intp dims[2]= {0,1};
		  PyArrayObject *parray= (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) angles, PyArray_DOUBLE, 0,0 );
		  PyArrayIterObject *it= NULL;
			if( parray ){
				PyArrayBuf= (double*) PyArray_DATA(parray);
			}
			else{
				parray= (PyArrayObject*) angles;
				it= (PyArrayIterObject*) PyArray_IterNew(angles);
			}
			dims[0]= N;
			for( i= 0; i< N; i++ ){
				if( PyArrayBuf ){
					angle= PyArrayBuf[i];
				}
				else{
					angle= PyFloat_AsDouble( PyArray_DESCR(parray)->f->getitem( it->dataptr, angles ) );
				}

				if( base && base != 1 ){
					angle*= base;
				}
				if( PyErr_Occurred() ){
					if( it ){
						Py_DECREF(it);
					}
					else if( parray ){
						Py_DECREF(parray);
					}
					goto SC2a_ESCAPE;
				}
				cephes_sincos( angle, &sins[i], &coss[i], flg );
				if( it ){
					PyArray_ITER_NEXT(it);
				}
			}
// 			r1= PyArray_FromDimsAndData( 1, dims, PyArray_DOUBLE, (char*) sins );
// 			r2= PyArray_FromDimsAndData( 1, dims, PyArray_DOUBLE, (char*) coss );
			r1= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) sins );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r1, NPY_OWNDATA );
			r2= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) coss );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r2, NPY_OWNDATA );
			ret= Py_BuildValue( "(OO)", r1, r2 );
			Py_XDECREF(r1);
			Py_XDECREF(r2);
			if( it ){
				Py_DECREF(it);
			}
			else if( parray ){
				Py_DECREF(parray);
			}
			return(ret);
		}
		else{
			PyErr_NoMemory();
			goto SC2a_ESCAPE;
		}
SC2a_ESCAPE:;
		if( sins ){
			PyMem_Free(sins);
		}
		if( coss ){
			PyMem_Free(coss);
		}
		return(NULL);
	}
	else{
	  double s, c;
		if( base && base != 1 ){
			angle= PyFloat_AsDouble(angles) * base;
		}
		else{
			angle= PyFloat_AsDouble(angles);
		}
		if( PyErr_Occurred() ){
			return(NULL);
		}
		cephes_sincos( angle, &s, &c, flg );
		return Py_BuildValue( "(dd)", s, c );
	}
}

static PyObject *sincos_noop( PyObject *self, PyObject *args )
{ double angle, base= 0;
  PyObject *angles;
  int isList= 0;
	
	if(!PyArg_ParseTuple(args, "O|d:mips_sincos", &angles, &base )){
		return NULL;
	}

	if( PyList_Check(angles) ){
		if( !(angles= PyList_AsTuple(angles)) ){
			PyErr_SetString( FMError, "Unexpected failure converting angles list to tuple" );
			return(NULL);
		}
		isList= 1;
	}

	if( base && base != M_PI2 ){
		base = M_PI2/base;
	}

	if( PyTuple_Check(angles) ){
	  long i, N= PyTuple_Size(angles);
	  PyObject *sins= PyList_New(N);
	  PyObject *coss= PyList_New(N), *ret= NULL;
		if( sins && coss ){
			for( i= 0; i< N; i++ ){
			  double s, c;
				angle= PyFloat_AsDouble( PyTuple_GetItem(angles,i) );

				if( base && base != 1 ){
					angle*= base;
				}
				if( PyErr_Occurred() ){
					goto SC2t_ESCAPE;
				}
				PyList_SetItem( sins, i, PyFloat_FromDouble(s) );
				PyList_SetItem( coss, i, PyFloat_FromDouble(c) );
			}
			if( !isList ){
			  PyObject *r;
				if( (r= PyList_AsTuple(sins)) ){
					sins= r;
				}
				if( (r= PyList_AsTuple(coss)) ){
					coss= r;
				}
			}
			ret= Py_BuildValue( "(OO)", sins, coss );
		}
		else{
			PyErr_NoMemory();
		}
SC2t_ESCAPE:;
		Py_XDECREF(sins);
		Py_XDECREF(coss);
		return(ret);
	}
	else if( PyArray_Check(angles) ){
	  long i, N= PyArray_Size(angles);
	  double *PyArrayBuf= NULL, *sins= PyMem_New( double, N);
	  double *coss= PyMem_New( double, N);
	  PyObject *r1= NULL, *r2= NULL, *ret= NULL;
		if( sins && coss ){
		  npy_intp dims[2]= {0,1};
		  PyArrayObject *parray= (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) angles, PyArray_DOUBLE, 0,0 );
		  PyArrayIterObject *it= NULL;
			if( parray ){
				PyArrayBuf= (double*) PyArray_DATA(parray);
			}
			else{
				parray= (PyArrayObject*) angles;
				it= (PyArrayIterObject*) PyArray_IterNew(angles);
			}
			dims[0]= N;
			for( i= 0; i< N; i++ ){
				if( PyArrayBuf ){
					angle= PyArrayBuf[i];
				}
				else{
					angle= PyFloat_AsDouble( PyArray_DESCR(parray)->f->getitem( it->dataptr, angles ) );
				}

				if( base && base != 1 ){
					angle*= base;
				}
				if( PyErr_Occurred() ){
					if( it ){
						Py_DECREF(it);
					}
					else if( parray ){
						Py_DECREF(parray);
					}
					goto SC2a_ESCAPE;
				}
				if( it ){
					PyArray_ITER_NEXT(it);
				}
			}
			r1= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) sins );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r1, NPY_OWNDATA );
			r2= PyArray_SimpleNewFromData( 1, dims, PyArray_DOUBLE, (char*) coss );
			PyArray_ENABLEFLAGS( (PyArrayObject*)r2, NPY_OWNDATA );
			ret= Py_BuildValue( "(OO)", r1, r2 );
			Py_XDECREF(r1);
			Py_XDECREF(r2);
			if( it ){
				Py_DECREF(it);
			}
			else if( parray ){
				Py_DECREF(parray);
			}
			return(ret);
		}
		else{
			PyErr_NoMemory();
			goto SC2a_ESCAPE;
		}
SC2a_ESCAPE:;
		if( sins ){
			PyMem_Free(sins);
		}
		if( coss ){
			PyMem_Free(coss);
		}
		return(NULL);
	}
	else{
	  double s, c;
		if( base && base != 1 ){
			angle= PyFloat_AsDouble(angles) * base;
		}
		else{
			angle= PyFloat_AsDouble(angles);
		}
		if( PyErr_Occurred() ){
			return(NULL);
		}
		return Py_BuildValue( "(dd)", s, c );
	}
}


static PyMethodDef sincos_methods[] =
{
	{ "sincos", sincos_sincos2, METH_VARARGS, doc_sincos },
//	{ "cxsin", sincos_cxsin, METH_VARARGS, "cxsin(x[,base=2PI])" },
//	{ "cxcos", sincos_cxcos, METH_VARARGS, "cxcos(x[,base=2PI])" },
//	{ "cxsincos", sincos_cxsincos, METH_VARARGS, "cxsincos(x[,base=2PI])" },
	{ "mips_sincos", sincos_mipssincos, METH_VARARGS, "mips_sincos(x[,base=2PI])" },
	{ "cephes_sincos", sincos_cephes_sincos, METH_VARARGS,
	       "cephes_sincos(x,interpolate,[,base=360]); if interpolate=True, do an interpolation between the calculated values"
	       " at the surrounding int(x) degrees"
	},
	{ "sincos_noop", sincos_noop, METH_VARARGS, "sincos_noop(x[,base=2PI]); calculates nothing" },
	{ NULL, NULL }
};

#ifndef IS_PY3K
void initsincos()
#else
PyObject *PyInit_sincos()
#endif
{
	PyObject *mod, *dict;

	mod=Py_InitModule("sincos", sincos_methods);

	FMError= PyErr_NewException( "sincos.error", NULL, NULL );
	Py_XINCREF(FMError);
	PyModule_AddObject( mod, "error", FMError );
	if( PyErr_Occurred() ){
		PyErr_Print();
	}

	dict=PyModule_GetDict(mod);

	import_array();

	M_PI2 = 2.0 * M_PI;
#ifdef IS_PY3K
	return mod;
#endif
}
