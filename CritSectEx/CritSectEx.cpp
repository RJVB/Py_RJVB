/*!
	@file CritSectEx.cpp
 */

#include "CritSectEx.h"

#ifdef CRITSECTGCC

#endif //CRITSECTGCC

#if defined(WIN32) || defined(_MSC_VER) || defined(CRITSECTGCC)

DWORD CritSectEx::s_dwProcessors = 0;

void CritSectEx::AllocateKernelSemaphore()
{
	if (!m_hSemaphore)
	{
		HANDLE hSemaphore = CreateSemaphore(NULL, 0, 0x7FFFFFFF, NULL);
		cseAssertEx(!hSemaphore, __FILE__, __LINE__);
		if (InterlockedCompareExchangePointer( (PVOID*) &m_hSemaphore, hSemaphore, NULL))
			VERIFY(CloseHandle(hSemaphore)); // we're late
	}
}

bool CritSectEx::PerfLock(DWORD dwThreadID, DWORD dwTimeout)
{
#ifdef DEBUG
	if( m_bIsLocked ){
		fprintf( stderr, "Thread %lu attempting to lock mutex of thread %lu\n",
			   dwThreadID, m_nLocker
		);
	}
#endif
	// Attempt spin-lock
	for (DWORD dwSpin = 0; dwSpin < m_dwSpinMax; dwSpin++)
	{
		if (PerfLockImmediate(dwThreadID)){
			return true;
		}

		YieldProcessor();
	}

	// Ensure we have the kernel event created
	AllocateKernelSemaphore();

	bool bval = PerfLockKernel(dwThreadID, dwTimeout);
	WaiterMinus();

	return bval;
}

inline bool CritSectEx::PerfLockKernel(DWORD dwThreadID, DWORD dwTimeout)
{
	bool bWaiter = false;

	for (DWORD dwTicks = GetTickCount(), m_bTimedOut = false; ; )
	{
		if (!bWaiter)
			WaiterPlus();

		if (PerfLockImmediate(dwThreadID)){
			return true;
		}
		else if( m_bTimedOut ){
			// RJVB 20111128:
			return false;
		}

		DWORD dwWait;
		if (INFINITE == dwTimeout)
			dwWait = INFINITE;
		else
		{
			dwWait = GetTickCount() - dwTicks; // how much time elapsed
			if (dwTimeout <= dwWait)
				return false;
			dwWait = dwTimeout - dwWait;
		}

		ASSERT(m_hSemaphore);
		switch (WaitForSingleObject(m_hSemaphore, dwWait))
		{
			case WAIT_OBJECT_0:
				bWaiter = false;
				break;
			case WAIT_TIMEOUT:
				m_bTimedOut = true;
				bWaiter = true;
				break;
			default:
				cseAssertEx(true, __FILE__, __LINE__);
				return false;
		}
	}

	// unreachable
}

void CritSectEx::SetSpinMax(DWORD dwSpinMax)
{
#if defined(WIN32) || defined(_MSC_VER)
	if (!s_dwProcessors)
	{
		SYSTEM_INFO stSI;
		GetSystemInfo(&stSI);
		s_dwProcessors = stSI.dwNumberOfProcessors;
	}
	if (s_dwProcessors > 1)
#endif
		m_dwSpinMax = dwSpinMax;
}

#endif // WIN32 || _MSC_VER || CRITSECTGCC

#pragma mark ---- C interface glue ----

CSEHandle *CreateCSEHandle(unsigned int dwSpinMax)
{
	return new CSEHandle((DWORD)dwSpinMax);
}

CSEHandle *DeleteCSEHandle(CSEHandle *cseH)
{
	if( cseH ){
		delete cseH;
	}
	return NULL;
}

unsigned char LockCSEHandle(CSEHandle *cseH)
{ bool unlock = false;
	if( cseH && cseH->cse ){
		cseH->IsLocked = cseH->cse->Lock(unlock);
	}
	return unlock;
}

const char *CSEHandleInfo(CSEHandle *cseH)
{
	if( cseH ){
		return cseH->CSEHandleInfo();
	}
	else{
		return NULL;
	}
}

unsigned char LockCSEHandleWithTimeout(CSEHandle *cseH, unsigned int dwTimeout)
{ bool unlock = false;
	if( cseH && cseH->cse ){
		cseH->IsLocked = cseH->cse->Lock(unlock, (DWORD) dwTimeout);
	}
	return unlock;
}

unsigned char UnlockCSEHandle(CSEHandle *cseH, unsigned char unlockFlag)
{ bool locked = false;
	if( cseH && cseH->cse ){
		cseH->cse->Unlock(unlockFlag);
		cseH->IsLocked = locked = cseH->cse->IsLocked();
	}
	return locked;
}

CSEScopedLock *ObtainCSEScopedLock(CSEHandle *cseH)
{
	return (cseH && cseH->cse)? new CSEScopedLock(cseH) : NULL;
}

CSEScopedLock *ObtainCSEScopedLockWithTimeout(CSEHandle *cseH, unsigned int dwTimeout)
{
	return (cseH && cseH->cse)? new CSEScopedLock(cseH->cse, (DWORD) dwTimeout) : new CSEScopedLock((DWORD) dwTimeout);
}

unsigned char LockCSEScopedLock(CSEScopedLock *scopeL)
{
	return ( scopeL && scopeL->scope )? scopeL->IsLocked = scopeL->scope->Lock() : false;
}

unsigned char LockCSEScopedLockWithTimeout(CSEScopedLock *scopeL, unsigned int dwTimeout)
{
	return ( scopeL && scopeL->scope )? scopeL->IsLocked = scopeL->scope->Lock((DWORD)dwTimeout) : false;
}

unsigned char UnlockCSEScopedLock(CSEScopedLock *scopeL)
{
	if( scopeL && scopeL->scope ){
		scopeL->scope->Unlock();
		return scopeL->IsLocked = scopeL->scope->IsLocked();
	}
	else{
		return false;
	}
}

CSEScopedLock *ReleaseCSEScopedLock(CSEScopedLock *scopeL)
{
	if( scopeL ){
		delete scopeL;
	}
	return NULL;
}

unsigned char IsCSEHandleLocked(CSEHandle *cseH)
{
	return (cseH && cseH->cse)? cseH->cse->IsLocked() : false;
}

unsigned char DidCSEHandleTimeout(CSEHandle *cseH)
{
	return (cseH && cseH->cse)? cseH->cse->TimedOut() : false;
}

unsigned char IsCSEScopeLocked(CSEScopedLock *scopeL)
{
	return (scopeL && scopeL->scope)? scopeL->scope->IsLocked() : false;
}

unsigned char DidCSEScopeTimeout(CSEScopedLock *scopeL)
{
	return (scopeL && scopeL->scope)? scopeL->scope->TimedOut() : false;
}


