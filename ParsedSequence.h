#ifndef _PARSED_SEQUENCE_H

typedef enum { ContArray=1, Array=2, Tuple=3 } PSTypes;

typedef struct ParsedSequences{
	PSTypes type;
	double *array;
	npy_intp N, dealloc_array;
	PyArrayObject *ContArray;
} ParsedSequences;

static PyObject *ParseSequence( PyObject *var, ParsedSequences *pseq, PyObject *FMError )
{ PyObject *ret;
	if( PyArray_Check(var) ){
		PyArrayObject* xd= (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) var, PyArray_DOUBLE, 0, 0 );
		PyArrayIterObject *it;
		pseq->N = PyArray_Size(var);
		if( xd ){
			pseq->array = (double*) PyArray_DATA(xd);
			pseq->dealloc_array = False;
			pseq->type = ContArray;
			// for decref'ing after finishing with the data operations:
			pseq->ContArray = xd;
			ret = var;
		}
		else{
			int i, ok= True;
			PyArrayObject *parray= (PyArrayObject*) var;
			pseq->ContArray = NULL;
			if( !(pseq->array = PyMem_New( double, pseq->N )) ){
				PyErr_NoMemory();
				return(NULL);
			}
			if( !(it= (PyArrayIterObject*) PyArray_IterNew(var)) ){
				PyMem_Free(pseq->array);
				return(NULL);
			}
			PyErr_Clear();
			for( i= 0; ok && i< pseq->N; i++ ){
				if( it->index < it->size ){
					PyObject *elem= PyArray_DESCR(parray)->f->getitem( it->dataptr, var);
					if( PyInt_Check(elem) || PyLong_Check(elem) || PyFloat_Check(elem) ){
						pseq->array[i] = PyFloat_AsDouble(elem);
						PyArray_ITER_NEXT(it);
					}
					else{
						PyErr_SetString( FMError, "type clash: only arrays with scalar, numeric elements are supported" );
						ok = False;
					}
				}
			}
			Py_DECREF(it);
			if( ok ){
				pseq->dealloc_array = True;
				pseq->type = Array;
				ret = var;
			}
			else{
				PyMem_Free(pseq->array);
				pseq->array = NULL;
				ret = NULL;
			}
		}
	}
	else if( PyList_Check(var) ){
		if( !(var= PyList_AsTuple(var)) ){
 			PyErr_SetString( FMError, "unexpected failure converting list to tuple" );
			return(NULL);
		}
		else{
			goto handleTuple;
		}
	}
	else if( PyTuple_Check(var) ){
handleTuple:;
		{ int i, ok= True;
			pseq->N = PyTuple_Size(var);
			if( !(pseq->array = PyMem_New( double, pseq->N )) ){
				PyErr_NoMemory();
				return(NULL);
			}
			for( i= 0; ok && i< pseq->N; i++ ){
				PyObject *el= PyTuple_GetItem(var, i);
				if( (el && (PyInt_Check(el) || PyLong_Check(el) || PyFloat_Check(el))) ){
					pseq->array[i] = PyFloat_AsDouble(el);
				}
				else{
					PyErr_SetString( FMError, "type clash: only tuples with scalar, numeric elements are supported" );
					ok = False;
				}
			}
			if( ok ){
				pseq->dealloc_array = True;
				pseq->type = Tuple;
				ret = var;
			}
			else{
				PyMem_Free(pseq->array);
				pseq->array = NULL;
				ret = NULL;
			}
		}
	}
	else{
		PyErr_SetString( FMError, "sequence must be a numpy.ndarray, tuple or list" );
		ret = NULL;
	}
	return(ret);
}


#define _PARSED_SEQUENCE_H
#endif
