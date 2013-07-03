/*
 * lowlevel code for filtering routines exported to Python in rjvbFilt.c
 \ (c) 2005-2010 R.J.V. Bertin
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

#include <Python.h>

#if defined(__GNUC__) && !defined(_GNU_SOURCE)
#	define _GNU_SOURCE
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
#	include <valarray>
#	ifdef __VEC__
#		include <altivec.h>
#	endif
#	include <macstl/valarray.h>
#	ifndef isnan
#		define isnan(X)	std::isnan(X)
#	endif
#endif

#include "pragmas.h"
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

extern PyObject *FMError;

unsigned long fourconv3_nan_handling( double *data, size_t NN, int nan_handling )
{ unsigned long nNaN=0;
	if( nan_handling ){
	  size_t i, j, sL= 0, eL, sR= NN, eR;
	  /* sL: 0 or index to previous, non-NaN element before this block
	   \ eL: NN or index of first non-NaN element of a 'block'
	   \ sR: NN or index of last non-NaN element in a block.
	   \ eR: NN or index of next, non-NaN element after this block.
	   */
		for( i= 0; i< NN; ){
			if( isNaN(data[i]) ){
			  /* retrieve previous, non-NaN element: */
				sL= (i>0)? i-1 : 0;
				  /* Find next, non-NaN element: */
				j= i+1;
				while( j< NN && isNaN(data[j]) ){
					j++;
				}
				eL= j;
			}
			else{
				sL= i;
				eL= i;
			}
			if( eL< NN ){
/* 						j= i;	*/
				j= eL+1;
				  /* See if there is another NaN: */
				while( j< NN && !isNaN(data[j]) ){
					j++;
				}
				if( j< NN && isNaN(data[j]) ){
					sR= (j>0)? j-1 : 0; /* MAX(j-1,0); */
					j+= 1;
					while( j< NN && isNaN(data[j]) ){
						j++;
					}
					eR= j;
				}
				else{
					sR= NN;
					eR= NN;
				}
			}
			if( sL== 0 && sR>= NN ){
			  /* Nothing special to be done: there are no NaNs here. */
			}
			else{
				if( sL != eL ){
					if( !isNaN(data[sL]) ){
						switch( nan_handling ){
							case 2:{
							    double slope= (data[eL] - data[sL]) / (eL - sL);
								  /* we have a non-NaN preceding value: fill the gap of NaN(s) with a linear
								   \ gradient.
								   */
#ifdef DEBUG
								fprintf( StdErr, "'left' NaN hole from %lu-%lu; filling with gradient {", sL+1, eL-1 );
#endif
								for( j= sL+1; j< eL && j< NN; j++ ){
									data[j]= data[sL] + slope* (j-sL);
									nNaN++;
#ifdef DEBUG
									fprintf( StdErr, "%g,", data[j] );
#endif
								}
#ifdef DEBUG
								fprintf( StdErr, "}\n" );
#endif
								break;
							}
							default:
							case 1:{
							  size_t halfway= sL + (eL-sL)/2;
#ifdef DEBUG
								fprintf( StdErr, "'left' NaN hole from %lu-%lu; filling with step\n", sL+1, eL-1 );
#endif
								for( j= sL+1; j< eL && j< NN; j++ ){
									data[j]= (j<halfway)? data[sL] : data[eL];
									nNaN++;
								}
								break;
							}
						}
					}
					else{
						  /* Best guess we can do is to fill the gap with the first non-NaN value we have at hand */
						  /* 20050108: must pad from sL and not from sL+1 ! */
#ifdef DEBUG
						fprintf( StdErr, "'left' NaN hole from %lu-%lu; padding with %g (%d)\n", sL+0, eL-1, data[eL], __LINE__ );
						fprintf( StdErr, "\td[%d]=%s d[%d]=%s d[%d]=%s d[%d]=%s\n",
							sL, ad2str(data[sL],NULL,NULL), sL+1, ad2str(data[sL+1],NULL,NULL),
							eL-1, ad2str(data[eL-1],NULL,NULL),
							eL, ad2str(data[eL],NULL,NULL), eL+1, ad2str(data[eL+1],NULL,NULL)
						);
#endif
						if( sL==0 && eL==1 ){
							data[0]= data[eL];
							nNaN++;
						}
						else{
							for( j= sL+0; j< eL && j< NN; j++ ){
								data[j]= data[eL];
								nNaN++;
							}
						}
					}
				}
				if( sR != eR ){
					if( eR< NN && !isNaN(data[eR]) ){
						switch( nan_handling ){
							case 2:{
							  double slope= (data[eR] - data[sR]) / (eR - sR);
#ifdef DEBUG
								fprintf( StdErr, "'right' NaN hole from %lu-%lu; filling with gradient {", sR+1, eR-1 );
#endif
								for( j= sR+1; j< eR && j< NN; j++ ){
									data[j]= data[sR] + slope* (j-sR);
									nNaN++;
#ifdef DEBUG
									fprintf( StdErr, "%g,", data[j] );
#endif
								}
#ifdef DEBUG
								fprintf( StdErr, "}\n" );
#endif
								break;
							}
							case 1:
							default:{
							  size_t halfway= sR + (eR-sR)/2;
#ifdef DEBUG
								fprintf( StdErr, "'right' NaN hole from %lu-%lu; filling with step\n", sR+1, eR-1 );
#endif
								for( j= sR+1; j< eR && j< NN; j++ ){
									data[j]= (j<halfway)? data[sR] : data[eR];
									nNaN++;
								}
								break;
							}
						}
					}
					else{
						  /* Best guess we can do is to fill the gap with the first non-NaN value we have at hand */
#ifdef DEBUG
						fprintf( StdErr, "'right' NaN hole from %lu-%lu; padding with %g\n", sR+1, eR-1, data[sR] );
#endif
						for( j= sR+1; j< eR && j< NN; j++ ){
							data[j]= data[sR];
							nNaN++;
						}
					}
				}
			}
			i= sR+1;
		}
	}
	return( nNaN );
}

#ifdef __cplusplus
	static void _convolve( double *Data, size_t NN, double *Mask, double *Output, int Start, int End, int Nm )
	{ int nm= Nm/ 2, i, j;
	  size_t end = NN - nm;
	  stdext::refarray<double> vmask(Mask,Nm), vdata(Data, NN);
		for( i= Start; i< End; i++ ){
			if( i < nm ){
			  int k;
			  stdext::valarray<double> vd = vdata[stdext::slice(0, Nm,1)].shift(i-nm);
				j = 0;
				do{
					k = i+ j- nm;
					vd[j] = Data[0];
					j++;
				} while( k < 0 && j < Nm );
				Output[i]= (vmask * vd).sum();
			}
			else if( i > end ){
			  int k;
			  stdext::valarray<double> vd = vdata[stdext::slice(NN-Nm, Nm,1)].shift(end-i);
				j = End + nm - i;
				do{
					k = i+ j- nm;
					vd[j] = Data[End-1];
					j++;
				} while( j < Nm );
				Output[i]= (vmask * vd).sum();
			}
			else{
				Output[i]= (vmask * vdata[stdext::slice(i-nm, Nm,1)]).sum();
			}
#if 0
			else{

				for( j= 0; j< Nm; j++ ){
					k = i+ j- nm;
					if( k< 0 ){
						vd[j] = Data[0];
					}
					else if( k< End ){
						vd[j] = Data[k];
					}
					else{
						vd[j] = Data[End-1];
					}
				}
				Output[i]= (vmask * vd).sum();
			}
#endif
		}
	}
#else
static void _convolve( double *Data, double *Mask, double *Output, int Start, int End, int Nm )
{ int nm= Nm/ 2, i, j, k;
	for( i= Start; i< End; i++ ){
	 double accum= 0;
		for( j= 0; j< Nm; j++ ){
		  double v;
			k= i+ j- nm;
			if( k< 0 ){
				v= Data[0];
			}
			else if( k< End ){
				v= Data[k];
			}
			else{
				v= Data[End-1];
			}
			accum+= Mask[j]* v;
		}
		Output[i]= accum;
	}
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

double *convolve( double *Data, int N, double *Mask, int Nm, int nan_handling )
{ int padding, NN, i;
  double *output;
	  // we will padd the input data with half the mask's width, in order to avoid boundary artefacts
	padding = Nm/2 + 1;
	NN = N + 2 * padding;
	if( (output = (double*) PyMem_New(double, NN)) ){
	  double *data= (double*) malloc( NN * sizeof(double));
		if( data ){
			  // make a copy of the input data, with the required amount of padding
			  // with the initial and last observed values:
			for( i= 0; i < padding; i++ ){
				data[i]= Data[0];
			}
			memcpy( &data[i], Data, N*sizeof(double) );
			i += N;
			for( ; i < NN; i++ ){
				data[i]= Data[N-1];
			}
			/* If requested, treat the data (source, input) array for NaNs. Gaps with
			 \ NaNs are filled with a linear gradient between the surrounding values (nan_handling==2)
			 \ or with a 'half-way step' between these values (nan_handling==1) if possible, otherwise,
			 \ simple padding with the first or last non-NaN value is done.
			 \ These estimates are removed after the convolution.
			 */
			fourconv3_nan_handling( data, NN, nan_handling );
#ifdef __cplusplus
			_convolve( data, NN, Mask, output, 0, NN, Nm );
#else
			_convolve( data, Mask, output, 0, NN, Nm );
#endif
			memmove( output, &output[padding], N*sizeof(double) );
			  // replace the original NaN values where they ought to go:
			if( nan_handling ){
				for( i= 0; i < N; i++ ){
					if( isNaN(Data[i]) ){
						output[i]= Data[i];
					}
				}
			}
			free(data);
		}
		else{
			PyErr_NoMemory();
		}
	}
	else{
		PyErr_NoMemory();
	}
	return( output );
}

#ifdef __cplusplus
}
#endif

#ifdef MINDOUBLE
#	define TINY MINDOUBLE
#else
#	define TINY DBL_MIN
#endif

int ludcmp(savgol_flp **a, int n, int *indx, savgol_flp *d)
/*
 \ Given a matrix a[1..n][1..n], this routine replaces it by the LU decomposition of a rowwise
 \ permutation of itself. a and n are input. a is output, arranged as in equation (2.3.14) above;
 \ indx[1..n] is an output vector that records the row permutation effected by the partial
 \ pivoting; d is output as if 1 depending on whether the number of row interchanges was even
 \ or odd, respectively. This routine is used in combination with lubksb to solve linear equations
 \ or invert a matrix.
 */
{ int i, imax= 0, j, k;
  savgol_flp big, dum, sum, temp;
	  /* vv stores the implicit scaling of each row. 	*/
  savgol_flp *vv = (savgol_flp*) malloc( (n+1)*sizeof(savgol_flp) );

	if( !vv ){
		return(0);
	}
	*d= 1.0;
	for( i= 1; i<= n; i++ ){
		big= 0.0;
		for( j= 1; j<= n; j++ ){
			if( (temp= fabs(a[i][j])) > big){
				big= temp;
			}
		}
		if( big == 0.0){
			PyErr_SetString( FMError, "Singular matrix in routine ludcmp" );
			errno= EINVAL;
			xfree(vv);
			return(0);
		}
		vv[i]= 1.0/big;
	}
	for( j= 1; j<= n; j++ ){
		for( i= 1; i< j; i++ ){
			sum= a[i][j];
			for( k= 1; k< i; k++ ){
				sum -= a[i][k]*a[k][j];
			}
			a[i][j]= sum;
		}
		big= 0.0;
		for( i= j; i<= n; i++ ){
			sum= a[i][j];
			for( k= 1; k< j; k++){
				sum -= a[i][k]*a[k][j];
			}
			a[i][j]= sum;
			if( (dum= vv[i]*fabs(sum)) >= big ){
				big= dum;
				imax= i;
			}
		}
		if( j != imax ){
			for( k= 1; k<= n; k++ ){
				dum= a[imax][k];
				a[imax][k]= a[j][k];
				a[j][k]= dum;
			}
			*d = -(*d);
			vv[imax]= vv[j];
		}
		indx[j]= imax;
		if( a[j][j] == 0.0){
			a[j][j]= TINY;
		}
		if( j != n ){
			dum= 1.0/(a[j][j]);
			for( i= j+1; i<= n; i++) a[i][j] *= dum;
		}
	}
	xfree(vv);
	return(1);
}

void lubksb(savgol_flp **a, int n, int *indx, savgol_flp *b)
/*
 \ Solves the set of n linear equations A . X = B. Here a[1..n][1..n] is input, not as the matrix
 \ A but rather as its LU decomposition, determined by the routine ludcmp. indx[1..n] is input
 \ as the permutation vector returned by ludcmp. b[1..n] is input as the right-hand side vector
 \ B, and returns with the solution vector X. a, n, andindx are not modified by this routine
 \ and can be left in place for successive calls with dirent right-hand sides b. This routine takes
 \ into account the possibility that b will begin with many zero elements, so it is efficient for use
 \ in matrix inversion.
 */
{ int i, ii= 0, ip, j;
  savgol_flp sum;
	for( i= 1; i<= n; i++ ){
	  /* When ii is set to a positive value, it will become the
	   \ index of the first nonvanishing element of b. We now
	   \ do the forward substitution. The
	   \ only new wrinkle is to unscramble the permutation as we go.
	   */
		ip= indx[i];
		sum= b[ip];
		b[ip]= b[i];
		if( ii){
			for( j= ii; j<= i-1; j++){
				sum -= a[i][j]*b[j];
			}
		}
		else if( sum){
		  /* A nonzero element was encountered, so from now on we
		   \ will have to do the sums in the loop above.
		   */
			ii= i;
		}
		b[i]= sum;
	}
	for( i= n; i>= 1; i-- ){
	  /* Now we do the backsubstitution.	*/
		sum= b[i];
		for( j= i+1; j<= n; j++){
			sum -= a[i][j]*b[j];
		}
		  /* Store a component of the solution vector X.	*/
		b[i]= sum/ a[i][i];
	}
}


void xfree_sgf_matrix( savgol_flp **a, int h, int v )
{ int i;
	if( a ){
		for( i= 0; i<= v; i++ ){
			xfree( a[i] );
		}
		xfree( a );
	}
}

savgol_flp **calloc_dmatrix( int h, int v)
{ int i;
  savgol_flp **m;

	  /* 20010901: RJVB: allocate 1 element more per row/column. Adaptation of NR
	   \ code dealing with matrices is tricky business...
	   */
	if( !(m = (savgol_flp **) calloc((unsigned) v+1,sizeof(savgol_flp*))) ){
		PyErr_SetString( FMError, "allocation failure 1 in calloc_dmatrix" );
		return( NULL );
	}
	for( i = 0; i <= v; i++ ){
		if( !(m[i] = (savgol_flp *) calloc((unsigned) h+ 1, sizeof(savgol_flp))) ){
			PyErr_SetString( FMError, "allocation failure 2 in calloc_dmatrix" );
			for( --i; i>= 0; i-- ){
				xfree( m[i] );
			}
			xfree(m);
			return(NULL);
		}
	}
	return( m );
}

#ifdef __cplusplus
extern "C" {
#endif

/* savgol():
 \ Returns in c[1..np], in wrap-around order (N.B.!) consistent with the argument respns in
 \ routine convlv, a set of Savitzky-Golay filter coefficients. nl is the number of leftward (past)
 \ data points used, while nr is the number of rightward (future) data points, making the total
 \ number of data points used nl +nr +1. ld is the order of the derivative desired (e.g., ld = 0
 \ for smoothed function). m is the order of the smoothing polynomial, also equal to the highest
 \ conserved moment; usual values are m = 2or m = 4.
 */
int savgol(savgol_flp *c, int np, int nl, int nr, int ld, int m)
{ void lubksb(savgol_flp **a, int n, int *indx, savgol_flp *b);
//   int ludcmp(savgol_flp **a, int n, int *indx, savgol_flp *d);
  int imj, ipj, j, k, kk, mm;
  savgol_flp d, fac, sum,**a;
  int *indx = (int*) malloc( (m+2) * sizeof(int) );
  savgol_flp *b = (savgol_flp*) malloc( (m+2) * sizeof(savgol_flp) );

	if( !indx || !b ){
		PyErr_NoMemory();
		return(0);
	}
	if( np < nl+nr+1 || nl < 0 || nr < 0 || ld > m || nl+nr < m){
		fprintf( StdErr, "bad args in savgol\n");
		errno= EINVAL;
		xfree(indx); xfree(b);
		return(0);
	}
	if( !(a= calloc_dmatrix(m+2, m+2)) || !indx || !b ){
		xfree(indx); xfree(b);
		return(0);
	}
	for( ipj= 0; ipj<= (m << 1); ipj++ ){
	  /* Set up the normal equations of the desired least-squares fit.	*/
		sum= (ipj)? 0.0 : 1.0;
		for( k= 1; k<= nr; k++){
			sum += (savgol_flp) pow((double) k, (double) ipj);
		}
		for( k= 1; k<= nl; k++){
			sum += (savgol_flp) pow((double)-k, (double) ipj);
		}
		mm= IMIN(ipj, 2*m-ipj);
		for( imj = -mm; imj<= mm; imj+= 2){
#ifdef DEBUG
		  int a1= 1+(ipj+imj)/ 2, a2= 1+(ipj-imj)/ 2;
			if( a1< m+2 && a2< m+2 ){
				a[a1][a2]= sum;
			}
			else{
				fprintf( stderr, "Range error in savgol.%d: a1=%d or a2=%d >= m+2=%d\n",
					__LINE__, a1, a2, m
				);
			}
#else
			a[1+(ipj+imj)/ 2][1+(ipj-imj)/ 2]= sum;
#endif
		}
	}
	if( !ludcmp(a, m+1, (int*) indx, &d) ){
		PyErr_SetString( FMError, "failure in ludcmp()" );
		xfree_sgf_matrix(a, m+2, m+2);
		xfree(indx); xfree(b);
		return(0);
	}
	for( j= 1; j<= m+1; j++){
		b[j]= 0.0;
	}
	b[ld+1]= 1.0;
	lubksb(a, m+1, indx, b);
	for( kk= 1; kk<= np; kk++){
	  /* Zero the output array (it may be bigger than number of coefficients).	*/
		c[kk]= 0.0;
	}
	for( k = -nl; k<= nr; k++ ){
	  /* Each Savitzky-Golay coefficient is the dot product of powers of
	    \an integer with the inverse matrix row.
	    */
		sum= b[1];
		fac= 1.0;
		for( mm= 1; mm<= m; mm++){
			sum += b[mm+1]*(fac *= k);
		}
		kk= ((np-k) % np)+1;	/* Store in wrap-around order.	*/
#if DEBUG
		if( kk> np ){
			fprintf( stderr, "Range error in savgol.%d: kk=%d > np=%d\n",
				__LINE__, kk, np
			);
		}
		else
#endif
		c[kk]= sum;
	}
	xfree_sgf_matrix(a, m+2, m+2);
	xfree(indx); xfree(b);
	return(1);
}

unsigned long savgol2D_dim( int fw, unsigned long *diag )
{ unsigned long N, d;
	// the "standard size" of the 1D kernel corresponding to an <fw> halfwidth;
	// the 2D kernel will be an N*N square
	N = fw * 2 + 3;
	// the diagonal of an fw*fw square (this is the size of the actual kernel we'll need to calculate)
	if( diag ){
		d = (unsigned long) (sqrt( 2.0 * fw * fw ) + 0.5);
		*diag = d * 2 + 3;
	}
	return N;
}

#define COEFF2D(c,i,j)	(c)[(i)+N1d*(j)]

int savgol2D( savgol_flp *c, unsigned long N, int fw, int deriv, int fo )
{ unsigned long N1d, N1d_2, diagN, i, j, k;
  savgol_flp *coeffs;

	if( fw < 0 || fw > (MAXINT-1)/2 ){
		return 0;
	}
	if( fo < 0 || fo > 2*fw ){
		return 0;
	}
	if( deriv < -fo || deriv > fo ){
		return 0;
	}
	N1d = savgol2D_dim( fw, &diagN );
	if( N != N1d * N1d ){
		return 0;
	}
	errno = 0;
	if( (coeffs = (savgol_flp*) malloc( (diagN+1) * sizeof(savgol_flp) )) ){
		if( !(fw== 0 && fo== 0 && deriv==0) ){
			if( savgol( &(coeffs)[-1], diagN, fw, fw, deriv, fo ) ){
					fprintf( StdErr, "coeffs[%lu,%d,%d,%d]={%g", diagN, fw, deriv, fo, coeffs[0] );
					for( i = 1; i < diagN; i++ ){
						fprintf( StdErr, ",%g", coeffs[i] );
					}
					fputs( "}\n", StdErr );
//					// testing:
//					coeffs[0] = 0;
//					for( i = 1; i <= diagN; i++ ){
//						coeffs[i] = i;
//					}
//					fprintf( StdErr, "coeffs={%g", coeffs[0] );
//					for( i = 1; i < diagN; i++ ){
//						fprintf( StdErr, ",%g", coeffs[i] );
//					}
//					fputs( "}\n", StdErr );
//				for( j = 0 ; j < N ; j++ ){
//					c[j] = 0.0/0.0;
//				}
				i = j = N1d_2 = N1d/2;
				for( j = 0 ; j <= N1d_2 ; j++ ){
//						i = 0;
//						fprintf( StdErr, "%lu:%lu|", j, i );
					for( i = 0; i <= N1d_2; i++ ){
					  long aa = N1d_2 - i, bb = N1d_2 - j,
						cc = N1d_2 + i, dd = N1d_2 + j;
						k = (unsigned long) (sqrt(i*i + j*j) + 0.5);
//							fprintf( StdErr, " [%ld,%ld],%lu", i, j, k );
						if( k <= diagN ){
							COEFF2D(c, aa, dd) = coeffs[k];
							COEFF2D(c, cc, bb) = coeffs[diagN-k];
							if( aa != N1d_2 && bb != N1d_2 && cc != N1d_2 && dd != N1d_2 ){
								COEFF2D(c, aa, bb) = coeffs[diagN-k];
								COEFF2D(c, cc, dd) = coeffs[k];
							}
						}
						else{
							COEFF2D(c, aa, dd) = 0;
							COEFF2D(c, cc, bb) = 0;
							COEFF2D(c, aa, bb) = 0;
							COEFF2D(c, cc, dd) = 0;
							if( aa != N1d_2 && bb != N1d_2 && cc != N1d_2 && dd != N1d_2 ){
								COEFF2D(c, aa, bb) = 0;
								COEFF2D(c, cc, dd) = 0;
							}
						}
					}
//						fprintf( StdErr, " |%lu:%lu\n", j, i-1 );
				}
				COEFF2D(c,N1d_2,N1d_2) = coeffs[0];
				free(coeffs);
				return 1;
			}
		}
	}
	return 0;
}

#ifdef __cplusplus
}
#endif

// ### spline functions
void eliminate_NaNs( double *x, double *y, int n )
  /* 20050523: for large n (1000000 or possibly even smaller), we can run into crashes. Avoid using alloca() c.s. */
{ /* _XGALLOCA(yy, double, n+1, yylen); */
  int i, N= 0;
  double *yy= NULL;
	  /* 20050515: we're working with Pascal-style arrays, so we need to go to <=n !! */
	for( i=1; i<= n; i++ ){
	  int j, J;
	  /* NaN handling. We're going to suppose that there are none in the x,
	   \ as the preparing routines ought to have eliminated those.
	   */
		if( NaN(y[i]) ){
		  /* If the point currently under scrutiny is a NaN itself, do a linear
		   \ intrapolation between the two non-NaN values we will find:
		   */
			if( !yy ){
				if( !(yy= (double*) malloc( (n+1) * sizeof(double) )) ){
					fprintf( StdErr, " (allocation problem: no NaN elimination! %s) ", strerror(errno) );
					return;
				}
				  /* rather than doing copying of all non-NaN elements, make yy a copy of y at once,
				   \ and then only touch the appropriate (NaN) elements.
				   */
				memcpy( &yy[1], &y[1], n*sizeof(double) );
			}
			for( j=i+1; j<= n && NaN(y[j]); j++ );
			for( J=i-1; J>= 1 && NaN(y[J]); J-- );
			if( J> 0 && j<= n ){
				yy[i]= y[J]+ (x[i] - x[J])* ( (y[j]-y[J])/ (x[j]-x[J]) );
				N+= 1;
			}
			else{
				yy[i]= y[i];
			}
		}
/* 		else{	*/
/* 			yy[i]= y[i];	*/
/* 		}	*/
	}
	if( N ){
		  /* 20050515: copy the appropriate portion: */
		memcpy( &y[1], &yy[1], n*sizeof(double) );
		xfree(yy);
	}
}

void spline( double *x, double *y, int n, double yp1, double ypn, double *y2)
/*
 \ Given arrays x[1..n] and y[1..n] containing a tabulated function, i.e., y i = f(xi), with
 \ x1<x2< ...<xN , and given values yp1 and ypn for the first derivative of the interpolating
 \ function at points 1 and n, respectively, this routine returns an array y2[1..n] that contains
 \ the second derivatives of the interpolating function at the tabulated points xi. If yp1 and/or
 \ ypn are equal to 1e30 or larger, the routine is signaled to set the corresponding boundary
 \ condition for a natural spline, with zero second derivative on that boundary.
 \ NOTE: arrays are 1 based!!
 */
{ int i,k;
  double p,qn,sig,un;
  double *u= NULL;

	if( n >= 2 ){
		u= (double*) calloc( n+1, sizeof(double) );
		if( !u ){
			return;
		}
	}
	else{
		return;
	}

	eliminate_NaNs( x, y, n );
	if( yp1 > 0.99e30 ){
	  /* The lower boundary condition is set either to be "natural"
	   \ or else to have a specified first derivative.
	   */
		y2[1]= u[1]= 0.0;
	}
	else{
	  int j;
		for( j=2; j< n && ( NaN(x[j]) || NaN(y[j]) ); j++ );
		y2[1]= -0.5;
		u[1]= (3.0/(x[j]-x[1]))*((y[j]-y[1])/(x[j]-x[1])-yp1);
	}
	for( i=2; i< n; i++ ){
	  int j, J;
	  double xii, yii, yJ, yj;
	  /* This is the decomposition loop of the tridiagonal al-
	   \ gorithm. y2 and u are used for tem-
	   \ porary storage of the decomposed
	   \ factors.
	   */
		j= i+1, J= i-1;
		  /* Update y[i] here! This is not the original data in SplineSet anyway, so we can. */
		xii= x[i], yii= y[i];
		yj= y[j];
		yJ= y[J];
		sig= (xii-x[J])/(x[j]-x[J]);
		p= sig*y2[J]+2.0;
		y2[i]= (sig-1.0)/p;
		u[i]= (yj-yii)/(x[j]-xii) - (yii-yJ)/(xii-x[J]);
		u[i]= (6.0*u[i]/(x[j]-x[J])-sig*u[J])/p;
	}
	if( ypn > 0.99e30 ){
	  /* The upper boundary condition is set either to be "natural"	*/
		qn= un= 0.0;
	}
	else{
	  int J;
	  /* or else to have a specified first derivative.	*/
		for( J=n-1; J> 1 && ( NaN(x[J]) || NaN(y[J]) ); J-- );
		qn= 0.5;
		un=(3.0/(x[n]-x[J]))*(ypn-(y[n]-y[J])/(x[n]-x[J]));
	}
	y2[n]= (un-qn*u[n-1])/(qn*y2[n-1]+1.0);
	  /* This is the backsubstitution loop of the tridiagonal algorithm.	*/
	for( k= n-1; k>= 1; k--){
		y2[k]= y2[k]*y2[k+1]+u[k];
	}
	xfree( u );
}

/* RJVB: as spline() above, but now do a piecewise linear interpolation: */
void pwlint_coeffs( double *x, double *y, int n, double *coeff)
{ int i;
	eliminate_NaNs( x, y, n );
	for( i= 1; i< n; i++ ){
	  int ii= i, j= i+1;
		if( j<= n ){
			coeff[i]= (y[j] - y[ii]) / (x[j] - x[ii]);
		}
		else{
			set_NaN( coeff[i] );
		}
	}
}

void pwlint(double *xa, double *ya, double *coeff, int n, double x, double *y)
{ int klo,khi,k;
  static int pklo = 2, pklh = 2;
	if( !coeff || n == 1 ){
		set_NaN(*y);
	}
	else{
		if( pklo < 1 || pklo > n ){
			pklo = 2;
		}
		if( pklh < 1 || pklh > n ){
			pklh = 2;
		}
		if( xa[pklo] <= x && xa[pklh] > x ){
			klo = pklo, khi = pklh;
		}
		else if( n > 2 && xa[pklo+1] <= x && xa[pklh+1] > x ){
			klo = pklo+1, khi = pklh+1;
			pklo = klo, pklh = khi;
		}
		else if( xa[pklo-1] <= x && xa[pklh-1] > x ){
			klo = pklo-1, khi = pklh-1;
			pklo = klo, pklh = khi;
		}
		else{
			klo= 1;
			khi= n;
			while( khi-klo > 1 ){
				k= (khi+klo) >> 1;
				if( xa[k] > x ){
					khi= k;
				}
				else{
					klo= k;
				}
			}
			pklo = klo, pklh = khi;
		}
		  /* klo and khi are now non-NaN values bracketing the input value of x.	*/
		{ double xlo= xa[klo], h= xa[khi]-xlo;
			if( h == 0.0 ){
				fprintf( StdErr, "pwlint was passed a value (%g) not in the X values array\n", x );
				set_NaN(*y);
			}
			else{
				  /*  The xa's must be distinct.	*/
				*y= (x- xlo)* coeff[klo]+ ya[klo];
			}
		}
	}
}

static int PWLInt= False;

/*
 \ It is important to understand that the program spline is called only once to
 \ process an entire tabulated function in arrays xi and yi . Once this has been done,
 \ values of the interpolated function for any value of x are obtained by calls (as many
 \ as desired) to a separate routine splint (for "spline interpolation"):
 */
void splint(double *xa, double *ya, double *y2a, int n, double x, double *y)
 /*
  \ Given the arrays xa[1..n] and ya[1..n], which tabulate a function (with the xai's in order),
  \ and given the array y2a[1..n], which is the output from spline above, and given a value of
  \ x, this routine returns a cubic-spline interpolated value y.
  */
{ int klo,khi, kl, kh, k;
  double h,b,a;

	if( PWLInt< 0 ){
		pwlint(xa, ya, y2a, n, x, y);
	}
	else{
	  static int pklo = 2, pklh = 2, delta = 1;

		if( !y2a || n == 1 ){
			set_NaN(*y);
			return;
		}

		if( xa[pklo] <= x && xa[pklh] > x ){
			klo = pklo, khi = pklh;
		}
		else if( pklo+delta <= n && xa[pklo+delta] <= x && pklh+delta <= n && xa[pklh+delta] > x ){
			klo = pklo+delta, khi = pklh+delta;
			pklo = klo, pklh = khi;
		}
		else if( pklo-delta > 0 && xa[pklo-delta] <= x && pklh-delta > 0 && xa[pklh-delta] > x ){
			klo = pklo-delta, khi = pklh-delta;
			pklo = klo, pklh = khi;
		}
		else{
			  /*
			   \ We will find the right place in the table by means of
			   \ bisection. This is optimal if sequential calls to this
			   \ routine are at random values of x. If sequential calls
			   \ are in order, and closely spaced, one would do better
			   \ to store previous values of klo and khi and test if
			   \ they remain appropriate on the next call.
			   */
			klo= 1;
			khi= n;
			while( khi-klo > 1 ){
				k= (khi+klo) >> 1;
				if( xa[k] > x ){
					khi= k;
				}
				else{
					klo= k;
				}
			}
			  /* klo and khi now bracket the input value of x.	*/
			kl= klo, kh= khi;
			{ int d = abs(klo-pklo);
				if( d && d != delta ){
					if( d < (n >>1) ){
						delta = d;
					}
				}
			}
			pklo = klo, pklh = khi;
		}
		  /* spline() will have filtered out all NaNs by the linear interpolation described above. */
		{ double xhi, xlo;
			h= (xhi= xa[khi])- (xlo= xa[klo]);
			if( h == 0.0 ){
				fprintf( StdErr, "splint was passed a value (%g) not in the X values array\n", x );
				set_NaN(*y);
			}
			else{
				  /* 	The xa's must be distinct.	*/
				a= (xhi-x)/h;
				b= (x-xlo)/h;
				  /* Cubic spline polynomial is now evaluated.	*/
				*y= a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[kl]+(b*b*b-b)*y2a[kh])*(h*h)/6.0;
			}
		}
	}
}

int ascanf_arg_error=0, ascanf_verbose= 0;

#define ASCANF_ARRAY_ELEM(a,i)		(a)[(i)]
#define ASCANF_ARRAY_ELEM_SET(a,i,v)	((a)[(i)]=(v))

#ifdef __cplusplus
extern "C" {
#endif

const char docSplineResample[] =
	"Spline_Resample(orgX, orgY, resampledX[,pad=1[,returnX=0,returnY=0,returnCoeffs=0]]]]: resample the data in orgY using a cubic spline.\n"
	"PWLint_Resample(orgX, orgY, resampledX[,pad=1[,returnX=0,returnY=0,returnCoeffs=0]]]]: idem, but using pair-wise linear interpolation.\n"
	" orgX and orgY describe the independent (X) and dependent (Y) values to be used in the calculation of the spline coefficients\n"
	" (orgX is supposed to be monotonically in/decreasing)\n"
	" resampledX specifies the values at which the spline must be evaluated (i.e. at which to resample)\n"
	" The optional <pad> argument activates a simple form of padding, which, given orgX={0,1,3} orgY={0,1,3} adds 2\n"
	" external points such that resampling is done against {-1,0,1,3,5},{-1,0,1,3,5}. This reduces boundary artefacts\n"
	" to some extent.\n"
	" orgX, orgY and resampledX must be sequences (lists, tuples, numpy arrays).\n"
	" The function returns the resampled data in a numpy array.\n"
	" [retOX,retOY,coeffs]: whether to return, respectively, the orginal X and Y data (padded, NaNs replaced)\n"
	" and the spline/interpolation coefficients. In this case, a tuple containing the various arrays is returned.\n";

// workhorse function for Spline_Resample and PWLInt_Resample. <use_PWLInt> determines whether to use pair-wise linear
// interpolation, or a cubic spline for interpolation; the other arguments correspond to the arguments in the docstring above.
// the function returns an array with the resampled values, which will be of length <resampledXN>; NULL in case of an error.
// Memory of the return arrays will have to be deallocated after use (or returned to the Python interpreter as a numpy
// array with the NPY_OWNDATA flag set.

double *SplineResample( int use_PWLInt, double *OrgX, int orgXN, double *OrgY, int orgYN, double *resampledX, int resampledXN,
	int pad, double **retOX, double **retOY, double **retCoeffs )
{ double *orgX, *orgY, *coeffs = NULL, *resampledY = NULL;
  int i;

	if( !OrgX || !OrgY || !resampledX ){
		return( NULL );
	}
	if( pad || retOX || retOY ){
		if( !(orgX= (double*) PyMem_New(double, ((pad)? orgXN+2 : orgXN) ))
			|| !(orgY= (double*) PyMem_New(double, ((pad)? orgYN+2 : orgYN) ))
		){
			PyErr_NoMemory();
			if( orgX ){
				PyMem_Free(orgX);
			}
			if( orgY ){
				PyMem_Free(orgY);
			}
			return( NULL );
		}
	}
	else{
		orgX = OrgX;
		orgY = OrgY;
	}
	if( !(coeffs = (double*) PyMem_New(double, ((pad)? orgXN+2 : orgXN) ))
		|| !(resampledY = (double*) PyMem_New(double, resampledXN))
	){
		PyErr_NoMemory();
		if( orgX ){
			PyMem_Free(orgX);
		}
		if( orgY ){
			PyMem_Free(orgY);
		}
		if( coeffs ){
			PyMem_Free(coeffs);
		}
		if( resampledY ){
			PyMem_Free(resampledY);
		}
		return( NULL );
	}
	if( orgX != OrgX ){
	  int j;
		if( pad ){
			orgX[0] = 2* ASCANF_ARRAY_ELEM(OrgX,0) - ASCANF_ARRAY_ELEM(OrgX,1);
			j= 1;
		}
		else{
			j= 0;
		}
		for( i= 0; i< orgXN; i++, j++ ){
			orgX[j] = ASCANF_ARRAY_ELEM(OrgX,i);
		}
		if( pad ){
			orgX[j] = 2* ASCANF_ARRAY_ELEM(OrgX,i-1) - ASCANF_ARRAY_ELEM(OrgX,i-2);
			if( pad && pragma_unlikely((ascanf_verbose> 1)) ){
				fprintf( StdErr, " OrgX[%d+2]=(%g", orgXN, orgX[0] );
				for( i= 1; i< orgXN+2; i++ ){
					fprintf( StdErr, ",%g", orgX[i] );
				}
				fprintf( StdErr, ") " );
				fflush( StdErr );
			}
		}
	}
	if( orgY != OrgY ){
	  int j;
		if( pad ){
			orgY[0] = 2* ASCANF_ARRAY_ELEM(OrgY,0) - ASCANF_ARRAY_ELEM(OrgY,1);
			j= 1;
		}
		else{
			j= 0;
		}
		for( i= 0; i< orgYN; i++, j++ ){
			orgY[j] = ASCANF_ARRAY_ELEM(OrgY,i);
		}
		if( pad ){
			orgY[j] = 2* ASCANF_ARRAY_ELEM(OrgY,i-1) - ASCANF_ARRAY_ELEM(OrgY,i-2);
		}
		if( pad && pragma_unlikely((ascanf_verbose> 1)) ){
			fprintf( StdErr, " OrgY[%d+2]=(%g", orgYN, orgY[0] );
			for( i= 1; i< orgYN+2; i++ ){
				fprintf( StdErr, ",%g", orgY[i] );
			}
			fprintf( StdErr, ") " );
			fflush( StdErr );
		}
	}
	if( use_PWLInt ){
		pwlint_coeffs( &orgX[-1], &orgY[-1], (pad)? orgYN+2 : orgYN, &coeffs[-1] );
	}
	else{
	  int first= 0, Last= (pad)? orgYN+1 : orgYN-1, tried= 0, last= Last;
	  double xp;
		while( ( isNaN(orgY[first]) || isNaN(orgY[first+1])
			|| isNaN(orgX[first]) || isNaN(orgX[first+1]) )
			&& first< last
		){
			first+= 1;
			tried+= 1;
		}
		while( ( isNaN(orgY[last]) || isNaN(orgY[last-1])
			|| isNaN(orgX[last]) || isNaN(orgX[last-1]) )
			&& last> first+2
		){
			last-= 1;
			tried+= 1;
		}
		if( first+1 != last ){
			xp = (orgY[first+1]-orgY[last]) / (orgX[first+1]-orgX[last]);
		}
		else{
			xp = (orgY[first]-orgY[last]) / (orgX[first]-orgX[last]);
		}
#if 0
		fprintf( StdErr, "spline(&%p[%d][-1], &%p[%d][%d], %d-%d+1=%d, %g, %g, &%p[%d][%d])\n",
			orgX, orgXN, orgY, orgYN, first-1, last, first, last-first+1,
				xp,
				(orgY[last]-orgY[last-1]) / (orgX[last]-orgX[last-1]),
				coeffs, orgXN, first-1
		);
#endif
		if( first >= Last || first == last ){
		  /* 20090922: sometimes, one just doesn't find a usable range of usable values... */
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (unsupported non-NaN element range %d-%d of %d) ", first, last, Last );
			}
		}
		else{
			if( tried ){
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (using non-NaN entries %d..%d) ", first, last );
				}
/* 					memset( coeffs, 0, orgXN*sizeof(double) );	*/
			}
			spline( &orgX[-1], &(orgY[first-1]), last-first+1,
				xp,
				(orgY[last]-orgY[last-1]) / (orgX[last]-orgX[last-1]),
				&coeffs[first-1]
			);
		}
		for( tried= 0; tried< first; tried++ ){
			set_NaN( coeffs[tried] );
		}
		for( tried= last+1; tried< orgYN; tried++ ){
			set_NaN( coeffs[tried] );
		}
	}

	if( resampledX ){
	  int porgYN= (pad)? orgYN+2 : orgYN;
		if( use_PWLInt ){
			for( i= 0; i< resampledXN; i++ ){
				pwlint( &orgX[-1], &orgY[-1], &coeffs[-1], porgYN,
					(double) resampledX[i], &resampledY[i] );
#if DEBUG == 2
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "pwlint(%d==%g) = %g\n", i, resampledX[i], resampledY[i] );
				}
#endif
			}
		}
		else{
			for( i= 0; i< resampledXN; i++ ){
				splint( &orgX[-1], &orgY[-1], &coeffs[-1], porgYN,
					(double) resampledX[i], &resampledY[i] );
#if DEBUG == 2
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "spline(%d==%g) = %g\n", i, resampledX[i], resampledY[i] );
				}
#endif
			}
		}
	}

	if( retOX ){
		*retOX = orgX;
	}
	else if( orgX != OrgX ){
		PyMem_Free(orgX);
	}
	if( retOY ){
		*retOY = orgY;
	}
	else if( orgY != OrgY ){
		PyMem_Free(orgY);
	}
	if( retCoeffs ){
		*retCoeffs = coeffs;
	}
	else{
		PyMem_Free(coeffs);
		coeffs = NULL;
	}
	return( resampledY );
}

const char docEulerSum[] = "EulerSum(values, t[,initialValue=0, returnAll=False, nanResets=False]):\n"
	" calculate the cumulative sum (cumsum) of values[i] * delta(t[i]).\n"
	" When returnAll=True, the running sum is returned instead in an array (the cumulative sum is of course in the last element).\n"
	" NB: it is assumed that the process was already being sampled before t[0], so a non-zero values[0]\n"
	" will be added to the cumsum as values[0] * average-delta-t! \n"
	" When nanResets is True, the integrator restarts at (is reset to) the (default) initial-value whenever\n"
	" values[i] and/or t[i] is a NaN; the NaNs themselves propagate to the running sum array if it was requested.\n";

double EulerArrays( double *vals, int valsN, double *t, int tN, double ival, double **returnAll, int nan_resets )
{ double *sum= NULL, result;

	ascanf_arg_error = 0;
	set_NaN(result);

	if( !(vals && t && valsN == tN) ){
		PyErr_SetString( FMError, "arrays must be valid and of equal size" );
		ascanf_arg_error = 1;
		return( result );
	}

	if( (returnAll && !(sum= (double*) PyMem_New(double, valsN ))) ){
		PyErr_NoMemory();
		ascanf_arg_error = 1;
	}
	else{
	  int ok= 0, nnstart= 0, i;
	  double av_dt= 0, prevT, T;
		if( sum ){
			for( i= 0; i< valsN; i++ ){
				set_NaN( sum[i] );
			}
		}
		while( !ok && nnstart< valsN ){
		  double v= ASCANF_ARRAY_ELEM(vals,nnstart);
			T= ASCANF_ARRAY_ELEM(t,nnstart);
			if( NaN(v) || NaN(T) ){
				nnstart+= 1;
			}
			else{
				ok= 1;
			}
		}
		if( ASCANF_ARRAY_ELEM(vals,nnstart) ){
		  int dtN = 0;
			for( i= nnstart+1, prevT= ASCANF_ARRAY_ELEM(t,nnstart); i< tN; i++ ){
				T= ASCANF_ARRAY_ELEM(t,i);
				if( !NaN(T) ){
					av_dt+= T-prevT;
					prevT= T;
					dtN += 1;
				}
			}
//			av_dt/= tN - nnstart - 1;
			av_dt /= (double) dtN;
			  /* we try to do something useful with a non-zero vals[0], even if t[0]==0
			   \ we'll assume that the sampled process has been going on for a while at the same (average) sampling rate.
			   */
/* 				prevT= -av_dt;	*/
			  /* 20070126: T - prevT should never be negative... */
			prevT= ASCANF_ARRAY_ELEM(t,nnstart) - av_dt;
			i= nnstart;
		}
		else{
			prevT= ASCANF_ARRAY_ELEM(t,nnstart+1);
			i= nnstart+1;
		}
		if( ascanf_verbose ){
			fprintf( StdErr, "dt=%g, first included element: %d, initial value= %g+%g=%g\n",
				av_dt, i, ival, ASCANF_ARRAY_ELEM(vals,i), ival+ASCANF_ARRAY_ELEM(vals,i)
			);
		}
		if( sum && i ){
			ASCANF_ARRAY_ELEM_SET(sum,i-1,ival);
		}
		{ double n_ival;
			set_NaN(n_ival);
			for( result= ival; i< valsN; i++ ){
			  double v= ASCANF_ARRAY_ELEM(vals,i);
				T= ASCANF_ARRAY_ELEM(t,i);
				if( NaN(v) || NaN(T) ){
					if( nan_resets ){
						n_ival= ival;
						set_NaN(result);
					}
				}
				else{
					if( nan_resets && !NaN(n_ival) ){
						result= n_ival;
						set_NaN(n_ival);
					}
					else{
						result+= v * (T - prevT);
					}
					prevT= T;
				}
				if( sum ){
					ASCANF_ARRAY_ELEM_SET(sum,i,result);
				}
			}
		}
	}
	if( returnAll ){
		*returnAll = sum;
	}
	return( result );
}

#ifdef __cplusplus
}
#endif
