/*
 * Python soft real-time scheduling system calls
 */
 
/*
rtsched
=======
Realtime scheduling support for python
    
Copyright (C) 2002  Crispin Wellington <crispin@aeonline.net>

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

/* 200502: interface to the Mach Thread API under Mac OS X
 \ (c) R.J.V. Bertin
*/
 
#include <stdio.h>

#ifdef __MACH__
#	include <mach/mach.h>
#	include <mach/mach_time.h>
#	include <mach/mach_init.h>
#	include <mach/thread_policy.h>
#	include <sys/sysctl.h>
#	include <pthread.h>
#endif
#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
//#include <Python.h>
#include "../PythonHeader.h"
#include "../Py_InitModule.h"

/* our exceptions */
static PyObject *SchedError;

#ifdef __MACH__

static mach_timebase_info_data_t sTimebaseInfo;
static double calibrator= 0;

static PyObject *rtsched_thread_policy_set( PyObject *self, PyObject *args )
{
	thread_act_t thread=0;
	thread_policy_flavor_t policy=THREAD_STANDARD_POLICY;
	union{
		struct thread_standard_policy standard;
		struct thread_time_constraint_policy ttc;
		struct thread_precedence_policy precedence;
	} poldat;
	unsigned long period= 0, computation= 0, constraint= 0;
	long importance= 0;
	mach_msg_type_number_t count;
	
	if(!PyArg_ParseTuple(args, "|iillll:thread_policy_set", &thread, &policy, &period, &computation, &constraint, &importance )){
		/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;
	}
	if( !thread ){
		thread= mach_thread_self();
	}
	if( !calibrator ){
		mach_timebase_info(&sTimebaseInfo);
		  /* go from microseconds to absolute time units (the timebase is calibrated in nanoseconds): */
		calibrator= 1e3 * sTimebaseInfo.denom / sTimebaseInfo.numer;
	}
	switch( policy ){
		default:
		case THREAD_STANDARD_POLICY:
			count= THREAD_STANDARD_POLICY_COUNT;
			break;
		case THREAD_TIME_CONSTRAINT_POLICY:
			poldat.ttc.period= period * calibrator;
			poldat.ttc.computation= computation * calibrator;
			poldat.ttc.constraint= constraint * calibrator;
			poldat.ttc.preemptible= 1;
			count= THREAD_TIME_CONSTRAINT_POLICY_COUNT;
			break;
		case THREAD_PRECEDENCE_POLICY:
			poldat.precedence.importance= importance;
			count= THREAD_PRECEDENCE_POLICY_COUNT;
			break;
	}
	if( thread_policy_set( thread, policy, (thread_policy_t)&poldat, count ) != KERN_SUCCESS ){
	  char errmsg[512];
		switch( policy ){
			case THREAD_TIME_CONSTRAINT_POLICY:
				snprintf( errmsg, sizeof(errmsg), "thread_policy_set(period=%lu,computation=%lu,constraint=%lu) failed",
					poldat.ttc.period, poldat.ttc.computation, poldat.ttc.constraint
				);
				break;
			case THREAD_PRECEDENCE_POLICY:
				snprintf( errmsg, sizeof(errmsg), "thread_policy_set(importance=%ld) failed",
					poldat.precedence.importance
				);
				break;
		}
		PyErr_SetString( SchedError, errmsg );
		return( NULL );
	}
	Py_INCREF(Py_None);
	return( Py_None );
}

static PyObject *rtsched_thread_policy_get( PyObject *self, PyObject *args )
{
	thread_act_t thread=0;
	thread_policy_flavor_t policy=THREAD_STANDARD_POLICY;
	union{
		struct thread_standard_policy standard;
		struct thread_time_constraint_policy ttc;
		struct thread_precedence_policy precedence;
	} poldat;
	unsigned long period= 0, computation= 0, constraint= 0;
	long importance= 0;
	mach_msg_type_number_t count;
	boolean_t get_default= TRUE;
	
	if(!PyArg_ParseTuple(args, "|ii:thread_policy_get", &thread, &policy )){
		/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;
	}
	if( !thread ){
		thread= mach_thread_self();
	}
	if( !calibrator ){
		mach_timebase_info(&sTimebaseInfo);
		/* go from microseconds to absolute time units (the timebase is calibrated in nanoseconds): */
		calibrator= 1e3 * sTimebaseInfo.denom / sTimebaseInfo.numer;
	}
	switch( policy ){
		default:
		case THREAD_STANDARD_POLICY:
			count= THREAD_STANDARD_POLICY_COUNT;
			break;
		case THREAD_TIME_CONSTRAINT_POLICY:
			count= THREAD_TIME_CONSTRAINT_POLICY_COUNT;
			break;
		case THREAD_PRECEDENCE_POLICY:
			count= THREAD_PRECEDENCE_POLICY_COUNT;
			break;
	}
	if( thread_policy_get( thread, policy, (thread_policy_t)&poldat, &count, &get_default ) != KERN_SUCCESS ){
		char errmsg[512];
		snprintf( errmsg, sizeof(errmsg), "thread_policy_get(thread=%lu) failed",
			thread
		);
		PyErr_SetString( SchedError, errmsg );
		return( NULL );
	}
	switch( policy ){
		default:
		case THREAD_STANDARD_POLICY:
			return Py_BuildValue("i", policy);
			break;
		case THREAD_TIME_CONSTRAINT_POLICY:
			period= poldat.ttc.period/calibrator;
			computation= poldat.ttc.computation/calibrator;
			constraint= poldat.ttc.constraint/calibrator;
			return Py_BuildValue("illl", policy, (unsigned long) period, (unsigned long) computation, (unsigned long) constraint);
			break;
		case THREAD_PRECEDENCE_POLICY:
			importance= poldat.precedence.importance;
			return Py_BuildValue("il", policy, poldat.precedence.importance);
			break;
	}
	
}

#else /* !__MACH__ */

#ifdef THIS_IS_FOR_MSWINDOWS
static int set_priority_RT()
{ HANDLE us = GetCurrentThread();
	int current = GetThreadPriority(us);
	SetThreadPriority( us, THREAD_PRIORITY_TIME_CRITICAL );
	return current;
}

static int set_priority(int priority)
{
	return SetThreadPriority( GetCurrentThread(), priority );
}
#endif

/* setscheduler(pid=0, policy=SCHED_OTHER, priority=0) */
static PyObject *rtsched_setscheduler( PyObject *self, PyObject *args )
{
	int pid=getpid(), policy=SCHED_OTHER, priority=0;
	struct sched_param param;
	int result;
	
	if(!PyArg_ParseTuple(args, "|iii:setscheduler", &pid, &policy, &priority ))	/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;
		
	param.sched_priority=priority;
	
	result=sched_setscheduler(pid, policy, &param);
	if(result)
	{
		if(errno==ESRCH)
		{
			/* no such pid */
			PyErr_SetString( SchedError, "No such process id" );
			return NULL;
		}
		else if(errno==EPERM)
		{
			/* insufficient privaleges. Need root */
			PyErr_SetString( SchedError, "Insufficient privileges. You need to be root" );
			return NULL;	
		}
		else if(errno==EINVAL)
		{
			/* unrecognised policy or priority */
			PyErr_SetString( SchedError, "Invalid policy or priority" );
			return NULL;	
		}

		/* error occured. errno set appropriately */
		PyErr_SetString( SchedError, strerror(errno) );
		return NULL;	
	}
	
	/* success. return None */
	Py_INCREF(Py_None);
	return Py_None;
}

/* policy,priority=getdcheduler(pid=0) */
static PyObject *rtsched_getscheduler( PyObject *self, PyObject *args )
{
	int pid=getpid(), policy=0, priority=0, result;
	static struct sched_param param;
	
	if(!PyArg_ParseTuple(args, "|i:getscheduler", &pid ))	/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;
	
	policy=sched_getscheduler(pid);
	if(policy==-1)
	{
		/* error occured. errno set appropriately */
		PyErr_SetString( SchedError, strerror(errno) );
		return NULL;
	}
	
	result=sched_getparam( pid, &param );
	if(result)
	{
		if(errno==ESRCH)
		{
			/* no such pid */
			PyErr_SetString( SchedError, "No such process id" );
			return NULL;		
		}
		else if(errno==EPERM)
		{
			/* insufficient privaleges. Need root */
			PyErr_SetString( SchedError, "Insufficient privileges. You need to be root" );
			return NULL;	
		}
		else if(errno==EINVAL)
		{
			/* unrecognised policy or priority */
			PyErr_SetString( SchedError, "Invalid policy or priority" );
			return NULL;	
		}
		
		/* error occured. errno set appropriately */
		PyErr_SetString( SchedError, strerror(errno) );
		return NULL;
	}
	priority=param.sched_priority;
	
	/* success. return a tuple of the values */
	return Py_BuildValue("ii",policy,priority);
}

/* yield() */
static PyObject *rtsched_yield( PyObject *self, PyObject *args )
{
	if(!PyArg_ParseTuple(args, ":yield" ))	/* | means optional params follow : means end of units. then function name for error messages */
					 return NULL;
					 
					 if(sched_yield())
					 {
						 /* error occured. errno set appropriately */
						 PyErr_SetString( SchedError, strerror(errno) );
						 return NULL;
					 }
					 
					 /* success. return None */
					 Py_INCREF(Py_None);
					 return Py_None;		
}

#endif /* __MACH__ */

/* nanosleep(nanos=0, secs=0) */
static PyObject *rtsched_nanosleep( PyObject *self, PyObject *args )
{
	int nanos=0, secs=0;
	static struct timespec req, rem;
	
	if(!PyArg_ParseTuple(args, "|ii:nanosleep", &nanos, &secs ))	/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;	
	
	if(nanos || secs)
	{
		req.tv_sec=secs;
		req.tv_nsec=nanos;
		
		nanosleep(&req,&rem);
	}
	
	/* success. return None */
	Py_INCREF(Py_None);
	return Py_None;	
}

/* usleep(micros=0) */
static PyObject *rtsched_usleep( PyObject *self, PyObject *args )
{
	long micros=0;
	
	if(!PyArg_ParseTuple(args, "|l:usleep", &micros ))	/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;

	if(micros)
		usleep((unsigned long)micros);
	
	/* success. return None */
	Py_INCREF(Py_None);
	return Py_None;		
}

/* msleep(millis=0) */
static PyObject *rtsched_msleep( PyObject *self, PyObject *args )
{
	long millis=0;
	
	if(!PyArg_ParseTuple(args, "|l:msleep", &millis ))	/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;

	if(millis)
		usleep((unsigned long)millis*1000UL);
	
	/* success. return None */
	Py_INCREF(Py_None);
	return Py_None;		
}

/* utime()
 * return the time of day in microseconds
 */
static PyObject *rtsched_utime( PyObject *self, PyObject *args )
{
	double micros=0;
	static struct timeval tv;
	static struct timezone tz;
	int result;
	
	if(!PyArg_ParseTuple(args, ":utime" ))	/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;

	/* get present time in microseconds */
	result=gettimeofday(&tv, &tz);
	if(result)
	{
		if(errno==EPERM)
		{
			/* Permission error. Shouldn't happen with gettimeofday */
			PyErr_SetString( SchedError, "Permission error. Cannot gettimeofday" );
			return NULL;			
		}
		else if(errno==EINVAL)
		{
			/* invalid value. Shouldn't happen with gettimeofday */
			PyErr_SetString( SchedError, "Invalid value. Cannot gettimeofday" );
			return NULL;			
		}
		else if(errno==EFAULT)
		{
			/* fatal internal error! */
			PyErr_SetString( SchedError, "Fatal internal error!" );
			return NULL;			
		}
	
		/* error occured. errno set appropriately */
		PyErr_SetString( SchedError, strerror(errno) );
		return NULL;
	}
	
	/* make the value into a long! We may overflow. TODO: work out WTF to do! */
	micros=1000000.0*tv.tv_sec+tv.tv_usec;
	
	/* success. return the value */
	return Py_BuildValue("d",micros);		
}

/* mtime()
 * return the time of day in milliseconds
 */
static PyObject *rtsched_mtime( PyObject *self, PyObject *args )
{
	double millis=0;
	static struct timeval tv;
	static struct timezone tz;
	int result;
	
	if(!PyArg_ParseTuple(args, ":mtime" ))	/* | means optional params follow : means end of units. then function name for error messages */
		return NULL;

	/* get present time in microseconds */
	result=gettimeofday(&tv, &tz);
	if(result)
	{
		if(errno==EPERM)
		{
			/* Permission error. Shouldn't happen with gettimeofday */
			PyErr_SetString( SchedError, "Permission error. Cannot gettimeofday" );
			return NULL;			
		}
		else if(errno==EINVAL)
		{
			/* invalid value. Shouldn't happen with gettimeofday */
			PyErr_SetString( SchedError, "Invalid value. Cannot gettimeofday" );
			return NULL;			
		}
		else if(errno==EFAULT)
		{
			/* fatal internal error! */
			PyErr_SetString( SchedError, "Fatal internal error!" );
			return NULL;			
		}
	
		/* error occured. errno set appropriately */
		PyErr_SetString( SchedError, strerror(errno) );
		return NULL;
	}
	
	/* make the value into a long! We may overflow. TODO: work out WTF to do! */
	millis=1000.0 * tv.tv_sec + tv.tv_usec / 1000.0;
	
	/* success. return the value */
	return Py_BuildValue("d",millis);		
}

/* 
 * the modules (rtsched's) object refs
 * -----------------------------------
 */
static PyMethodDef rtsched_methods[] =
{
#ifdef __MACH__
	{ "thread_policy_set", rtsched_thread_policy_set, METH_VARARGS },
	{ "thread_policy_get", rtsched_thread_policy_get, METH_VARARGS },
#else
	{ "setscheduler", rtsched_setscheduler, METH_VARARGS },
	{ "getscheduler", rtsched_getscheduler, METH_VARARGS },
	{ "syield", rtsched_yield, METH_VARARGS },
#endif
	{ "nanosleep", rtsched_nanosleep, METH_VARARGS },
	{ "usleep", rtsched_usleep, METH_VARARGS },
	{ "msleep", rtsched_msleep, METH_VARARGS },
	{ "utime", rtsched_utime, METH_VARARGS },
	{ "mtime", rtsched_mtime, METH_VARARGS },
	{ NULL, NULL }
};

#define DEC_CONSTN(module,x) PyModule_AddIntConstant(module, #x, (int) x);

/* function to help us insert values into the module dictionary */
static void insint(PyObject *d, char *name, int value)
{
        PyObject *v = PyLong_FromLong((long) value);
        if (v == NULL) {
                /* Don't bother reporting this error */
                PyErr_Clear();
        }
        else {
                PyDict_SetItemString(d, name, v);
                Py_DECREF(v);
        }
}

#ifdef DL_EXPORT
DL_EXPORT(void)
#else
void
#endif
initrtsched(void)
{
	PyObject *mod, *dict;

	mod=Py_InitModule("rtsched", rtsched_methods);
	dict=PyModule_GetDict(mod);
        
	/* exceptions */
	SchedError=PyErr_NewException("rtsched.SchedError", NULL, NULL);
	PyDict_SetItemString(dict,"SchedError", SchedError);

	/* our module values - insert into the module disctionary */
#ifdef __MACH__
	insint(dict, "THREAD_STANDARD_POLICY", THREAD_STANDARD_POLICY);
	insint(dict, "THREAD_TIME_CONSTRAINT_POLICY", THREAD_TIME_CONSTRAINT_POLICY);
	insint(dict, "THREAD_PRECEDENCE_POLICY", THREAD_PRECEDENCE_POLICY);
	insint(dict, "THREAD_TIME_CONSTRAINT_POLICY_COUNT", THREAD_TIME_CONSTRAINT_POLICY_COUNT);
#else	
	insint(dict, "SCHED_FIFO", SCHED_FIFO);
	insint(dict, "SCHED_RR", SCHED_RR);
	insint(dict, "SCHED_OTHER", SCHED_OTHER);
#endif
	
}
