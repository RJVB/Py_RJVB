#ifndef _RJVBFILTERS_H

#ifdef __cplusplus
extern "C" {
#endif

	extern unsigned long fourconv3_nan_handling( double *data, size_t NN, int nan_handling );
	extern double *convolve( double *Data, int N, double *Mask, int Nm, int nan_handling );

#	define savgol_flp	double
	extern int savgol(savgol_flp *c, int np, int nl, int nr, int ld, int m);

#	ifndef xfree
#		define xfree(x)	if((x)){ free((x)); (x)=NULL; }
#	endif

#	ifndef MAXINT
#		define MAXINT INT_MAX
#	endif
#	ifndef MAXSHORT
#		define MAXSHORT SHRT_MAX
#	endif

#ifndef MAX
#	define MAX(a,b)                (((a)>(b))?(a):(b))
#endif
#ifndef MIN
#	define MIN(a,b)                (((a)<(b))?(a):(b))
#endif

#if defined(__GNUC__)
/* IMIN: version of MIN for integers that won't evaluate its arguments more than once (as a macro would).
\ With gcc, we can do that with an inline function, removing the need for static cache variables.
\ There are probably other compilers that honour the inline keyword....
*/
#	undef IMIN
	inline static int IMIN(int m, int n)
	{
		return( MIN(m,n) );
	}
#else
#	ifndef IMIN
		static int mi1,mi2;
#		define IMIN(m,n)	(mi1=(m),mi2=(n),(mi1<mi2)? mi1 : mi2)
#	endif
#endif

extern double *SplineResample( int use_PWLInt, double *OrgX, int orgXN, double *OrgY, int orgYN, double *resampledX, int resampledXN,
						 int pad, double **retOX, double **retOY, double **retCoeffs );

extern double EulerArrays( double *vals, int valsN, double *t, int tN, double ival, double **returnAll, int nan_resets );

extern const char docSplineResample[], docEulerSum[];

#ifdef __cplusplus
}
#endif

#	define _RJVBFILTERS_H
#endif
