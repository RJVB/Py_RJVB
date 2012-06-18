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

#include <stdio.h>

#ifdef __MACH__
#	include <mach/mach.h>
#	include <mach/mach_time.h>
#	include <mach/mach_init.h>
#	include <sys/sysctl.h>
#endif
#ifdef __CYGWIN__
#	undef _WINDOWS
#	undef WIN32
#	undef MS_WINDOWS
#	undef _MSC_VER
#endif
#if ! defined(_WINDOWS) && !defined(WIN32) && !defined(MS_WINDOWS) && !defined(_MSC_VER)
#	include <time.h>
#	include <sys/time.h>
#	include <unistd.h>
#else
#	define MS_WINDOWS
#endif
#include <errno.h>

#include "PythonHeader.h"
#include "Py_InitModule.h"

#include "HRTime.h"

static PyObject *HRTimeError;

#define RETURN_NONE return (Py_INCREF(Py_None), Py_None);

#if defined(__MACH__)

#warning "Using mach_time_base and mach_absolute_time!"
#include <time.h>

static mach_timebase_info_data_t sTimebaseInfo;
static double calibrator= 0;

static void init_HRTime()
{
	if( !calibrator ){
		mach_timebase_info(&sTimebaseInfo);
		  /* go from absolute time units to seconds (the timebase is calibrated in nanoseconds): */
		calibrator= 1e-9 * sTimebaseInfo.numer / (double) sTimebaseInfo.denom;
	}
}

    /*DOC*/ static char doc_HRTime[] =
    /*DOC*/    "HRTime.HRTime() -> double\n"
    /*DOC*/    "seconds since initialisation\n"
    /*DOC*/    "Uses mach_absolute_time().\n"
    /*DOC*/ ;

static PyObject *HRTime_HRTime( PyObject *self, PyObject *args )
{
	
// 	if(!PyArg_ParseTuple(args, "" )){
// 		return NULL;
// 	}
	return PyFloat_FromDouble( mach_absolute_time() * calibrator );
}

static double ticTime;
static PyObject *HRTime_tic( PyObject *self, PyObject *args )
{
	
// 	if(!PyArg_ParseTuple(args, "" )){
// 		return NULL;
// 	}
// 	return PyFloat_FromDouble( (ticTime= mach_absolute_time() * calibrator) );
	ticTime= mach_absolute_time() * calibrator;

	RETURN_NONE;
}

static PyObject *HRTime_toc( PyObject *self, PyObject *args )
{
	
// 	if(!PyArg_ParseTuple(args, "" )){
// 		return NULL;
// 	}
	return PyFloat_FromDouble( mach_absolute_time() * calibrator - ticTime );
}


#elif defined(linux)

#include <time.h>

#	ifdef CPUCLOCK_CYCLES_PER_SEC

	typedef unsigned long long tsc_time;

	typedef struct tsc_timers{
		tsc_time t1, t2;
	} tsc_timers;

	static __inline__ tsc_time read_tsc()
	{ tsc_time ret;

		__asm__ __volatile__("rdtsc": "=A" (ret)); 
		/* no input, nothing else clobbered */
		return ret;
	}

#	define tsc_get_time(t)  ((*(tsc_time*)t)=read_tsc())
#	define tsc_time_to_sec(t) (((double) (*t)) / CPUCLOCK_CYCLES_PER_SEC)

#	endif

static inline double gettime()
#ifdef CPUCLOCK_CYCLES_PER_SEC
{ tsc_time t;
	tsc_get_time(&t);
	return tsc_time_to_sec( &t );
}
#elif defined(CLOCK_MONOTONIC)
{ struct timespec hrt;
	clock_gettime( CLOCK_MONOTONIC, &hrt );
	return hrt.tv_sec + hrt.tv_nsec * 1e-9;
}
#elif defined(CLOCK_REALTIME)
{ struct timespec hrt;
	clock_gettime( CLOCK_REALTIME, &hrt );
	return hrt.tv_sec + hrt.tv_nsec * 1e-9;
}
#else
	  /* Use gettimeofday():	*/
{ struct timezone tzp;
  struct timeval ES_tv;

	gettimeofday( &ES_tv, &tzp );
	return ES_tv.tv_sec + ES_tv.tv_usec* 1e-6;
}
#endif

static void init_HRTime()
{
	{ struct timespec res;
		fprintf( stderr, "clock_getres(CLOCK_REALTIME)=%d", clock_getres(CLOCK_REALTIME, &res) );
		fprintf( stderr, " tv_sec=%lu tv_nsec=%ld\n", res.tv_sec, res.tv_nsec );
		fprintf( stderr, "clock_getres(CLOCK_MONOTONIC)=%d", clock_getres(CLOCK_MONOTONIC, &res) );
		fprintf( stderr, " tv_sec=%lu tv_nsec=%ld\n", res.tv_sec, res.tv_nsec );
	}
}

    /*DOC*/ static char doc_HRTime[] =
    /*DOC*/    "HRTime.HRTime() -> double\n"
    /*DOC*/    "seconds since initialisation\n"
    /*DOC*/    "Uses mach_absolute_time().\n"
    /*DOC*/ ;

static PyObject *HRTime_HRTime( PyObject *self, PyObject *args )
{
	
// 	if(!PyArg_ParseTuple(args, "" )){
// 		return NULL;
// 	}
	return PyFloat_FromDouble( gettime() );
}

static double ticTime;
static PyObject *HRTime_tic( PyObject *self, PyObject *args )
{
	
// 	if(!PyArg_ParseTuple(args, "" )){
// 		return NULL;
// 	}
	ticTime= gettime();

	RETURN_NONE;
}

static PyObject *HRTime_toc( PyObject *self, PyObject *args )
{
	
// 	if(!PyArg_ParseTuple(args, "" )){
// 		return NULL;
// 	}
	return PyFloat_FromDouble( gettime() - ticTime );
}

#elif defined(_WINDOWS) || defined(WIN32) || defined(MS_WINDOWS) || defined(_MSC_VER)

#include <windows.h>

static LARGE_INTEGER lpFrequency;
static double calibrator= 0;

    /*DOC*/ static char doc_HRTime[] =
    /*DOC*/    "HRTime.HRTime() -> double\n"
    /*DOC*/    "seconds since initialisation\n"
    /*DOC*/    "Uses QueryPerformanceCounter().\n"
    /*DOC*/ ;

static void init_HRTime()
{
	if( !calibrator ){
		if( !QueryPerformanceFrequency(&lpFrequency) ){
			calibrator= 0;
		}
		else{
			calibrator= 1.0 / ((double) lpFrequency.QuadPart);
		}
	}
}


static PyObject *HRTime_HRTime( PyObject *self, PyObject *args )
{ LARGE_INTEGER count;
	
	QueryPerformanceCounter(&count);
	return PyFloat_FromDouble( count.QuadPart * calibrator );
}

static double ticTime;
static PyObject *HRTime_tic( PyObject *self, PyObject *args )
{ LARGE_INTEGER count;
	QueryPerformanceCounter(&count);
	ticTime= count.QuadPart * calibrator;
	RETURN_NONE;
}

static PyObject *HRTime_toc( PyObject *self, PyObject *args )
{ LARGE_INTEGER count;
	QueryPerformanceCounter(&count);
	return PyFloat_FromDouble( count.QuadPart * calibrator - ticTime );
}

struct timezone{
	int tz_minuteswest;
	int tz_dsttime;
};

struct timeval0{
	long tv_sec;
	long tv_usec;
};

int gettimeofday( struct timeval *tp, struct timezone *tzp )
{ SYSTEMTIME st;
	GetSystemTime(&st);
	tp->tv_sec = st.wHour * 3600 + st.wMinute * 60 + st.wSecond;
	tp->tv_usec = st.wMilliseconds * 1000;
	return( 0 );
}

#else


	  /* Use gettimeofday():	*/
#define gettime(time)	{ struct timezone tzp; \
	  struct timeval ES_tv; \
		gettimeofday( &ES_tv, &tzp ); \
		time= ES_tv.tv_sec + ES_tv.tv_usec* 1e-6; \
	}

    /*DOC*/ static char doc_HRTime[] =
    /*DOC*/    "HRTime.HRTime() -> double\n"
    /*DOC*/    "seconds since initialisation\n"
    /*DOC*/    "Uses gettimeofday().\n"
    /*DOC*/ ;

static void init_HRTime()
{
}

static PyObject *HRTime_HRTime( PyObject *self, PyObject *args )
{ double time;
	
// 	if(!PyArg_ParseTuple(args, "" )){
// 		return NULL;
// 	}
	gettime(time);
	return PyFloat_FromDouble( time );
}

static double ticTime;
static PyObject *HRTime_tic( PyObject *self, PyObject *args )
{
	
// 	if(!PyArg_ParseTuple(args, "" )){
// 		return NULL;
// 	}
	gettime(ticTime);

	RETURN_NONE;
}

static PyObject *HRTime_toc( PyObject *self, PyObject *args )
{
	
// 	if(!PyArg_ParseTuple(args, "" )){
// 		return NULL;
// 	}
	{ double time;
		gettime(time);
		return PyFloat_FromDouble( time - ticTime );
	}
}

#endif

    /*DOC*/ static char doc_nanosleep[] =
    /*DOC*/    "HRTime.nanosleep(seconds[,nanos]) -> double\n"
    /*DOC*/    "go to sleep for the specified time, with theoretical nanosecond resolution.\n"
    /*DOC*/    "returns the time actually slept\n"
    /*DOC*/ ;

static PyObject *HRTime_nanosleep( PyObject *self, PyObject *args )
{
	double secs=0;
	int nanos= 0;
	if(!PyArg_ParseTuple(args, "d|i:nanosleep", &secs, &nanos ))	/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;	
	
#ifdef MS_WINDOWS
	{ unsigned long slp = (unsigned long) (secs * 1000 + nanos*1e-6);
		Sleep(slp);
		return PyFloat_FromDouble((double)slp);
	}
#else
	
	if(secs > 0 || nanos>0){
	  static struct timespec req, rem;
		req.tv_sec= (time_t) floor(secs);
		req.tv_nsec= (int32_t) ( (secs - req.tv_sec) * 1e9 ) + nanos;
		rem.tv_sec= rem.tv_nsec= 0;
		
/* 		fprintf( stderr, "nanosleep(%g -> %d,%d)\n", secs, req.tv_sec, req.tv_nsec );	*/
		errno= 0;
		if( !nanosleep(&req,&rem) ){
			rem= req;
		}
		else{
			if( !errno ){
				rem.tv_sec= req.tv_sec - rem.tv_sec;
				rem.tv_nsec= req.tv_nsec - rem.tv_nsec;
			}
			else{
				PyErr_SetString( HRTimeError, strerror(errno) );
			}
		}
		return PyFloat_FromDouble( rem.tv_sec + rem.tv_nsec * 1e-9 );
	}
	else{
		return PyFloat_FromDouble( 0 );
	}
#endif
}

    /*DOC*/ static char doc_usleep[] =
    /*DOC*/    "HRTime.usleep(int microseconds) -> double\n"
    /*DOC*/	"returns whatever the usleep systemcall returns!\n"
    /*DOC*/ ;

/* usleep(micros=0) */
static PyObject *HRTime_usleep( PyObject *self, PyObject *args )
{
	long micros=0;
	int ret= 0;
	
	if(!PyArg_ParseTuple(args, "l:usleep", &micros ))	/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;

	if(micros){
#ifdef MS_WINDOWS
		Sleep((unsigned long)(micros/1000.0));
		ret = 0;
#else
		ret= usleep((unsigned long)micros);
#endif
	}

	return PyInt_FromLong( (long) ret );
}

    /*DOC*/ static char doc_timeofday[] =
    /*DOC*/    "HRTime.timeofday() -> double\n"
    /*DOC*/    "present time in seconds, obtained with gettimeofday()\n"
    /*DOC*/ ;

/* timeofday()
 * return the time of day in seconds
 */
static PyObject *HRTime_timeofday( PyObject *self, PyObject *args )
{
	double seconds=0;
	static struct timeval tv;
	static struct timezone tz;
	int result;
	
	if(!PyArg_ParseTuple(args, ":timeofday" ))	/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;

	/* get present time in microseconds */
	result=gettimeofday(&tv, &tz);
	if(result)
	{
		if(errno==EPERM)
		{
			/* Permission error. Shouldn't happen with gettimeofday */
			PyErr_SetString( HRTimeError, "Permission error. Cannot gettimeofday" );
			return NULL;			
		}
		else if(errno==EINVAL)
		{
			/* invalid value. Shouldn't happen with gettimeofday */
			PyErr_SetString( HRTimeError, "Invalid value. Cannot gettimeofday" );
			return NULL;			
		}
		else if(errno==EFAULT)
		{
			/* fatal internal error! */
			PyErr_SetString( HRTimeError, "Fatal internal error!" );
			return NULL;			
		}
	
		/* error occured. errno set appropriately */
		PyErr_SetString( HRTimeError, strerror(errno) );
		return NULL;
	}
	
	seconds=tv.tv_sec+tv.tv_usec * 1e-6;
	
	/* success. return the value */
	return PyFloat_FromDouble(seconds);
}

/*DOC*/ static char doc_tic[] =
/*DOC*/    "HRTime.tic() -> None\n"
/*DOC*/    "mark start of a timing interval"
/*DOC*/ ;

/*DOC*/ static char doc_toc[] =
/*DOC*/    "HRTime.toc() -> double\n"
/*DOC*/    "returns the number of seconds elapsed since the last HRTime.tic() invocation"
/*DOC*/ ;

/* 
 * the modules (HRTime's) object refs
 * -----------------------------------
 */
static PyMethodDef HRTime_methods[] =
{
	{ "HRTime", HRTime_HRTime, METH_VARARGS, doc_HRTime },
	{ "tic", HRTime_tic, METH_VARARGS, doc_tic },
	{ "toc", HRTime_toc, METH_VARARGS, doc_toc },
	{ "nanosleep", HRTime_nanosleep, METH_VARARGS, doc_nanosleep },
	{ "usleep", HRTime_usleep, METH_VARARGS, doc_usleep },
	{ "timeofday", HRTime_timeofday, METH_VARARGS, doc_timeofday },
	{ NULL, NULL }
};

/* function to help us insert values into the module dictionary */
static void insint(PyObject *d, char *name, int value)
{
        PyObject *v = PyInt_FromLong((long) value);
        if (v == NULL) {
                /* Don't bother reporting this error */
                PyErr_Clear();
        }
        else {
                PyDict_SetItemString(d, name, v);
                Py_DECREF(v);
        }
}

#ifdef IS_PY3K
PyObject *PyInit_HRTime(void)
#else
void initHRTime(void)
#endif
{
	PyObject *mod, *dict;

	mod=Py_InitModule("HRTime", HRTime_methods);
	dict=PyModule_GetDict(mod);

	/* exceptions */
	HRTimeError=PyErr_NewException("HRTime.HRTimeError", NULL, NULL);
	PyDict_SetItemString(dict,"HRTimeError", HRTimeError);

	init_HRTime();
#ifdef IS_PY3K
	return mod;
#endif
}
