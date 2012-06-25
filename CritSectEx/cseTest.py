import sys,os
try:
	from multiprocessing import Process as Thread
	from multiprocessing import current_process as currentThread
	from multiprocessing import Value as shValue
	useMP=True
except:
	from threading import Thread,currentThread
	useMP=False
	class shValue:
		def __init__(self,t,v):
			self.value = v
		def __repr__(self):
			return '<shValue(%s)>' %(self.value)

from MSEmul import *
from HRTime import *

tStart = 0

class bgThread(Thread):
	def __init__(self,evt=None,sleeper=None,sem=None):
		Thread.__init__(self)
		self.evt = None
		self.sleepTime = None
		if evt:
			self.evt = evt
		elif sleeper:
			self.sleepTime = shValue('d',sleeper)
		elif sem:
			self.sem=sem
		self.ok = shValue('b',True)
		self.threadHandle = GetCurrentThread()

	def run(self):
		global tStart
		if self.evt:
			print >>sys.stderr, '##%lx bgThread2Nudge starting to wait for nudge event %s at t=%g' %(currentThread().ident, self.evt, HRTime() - tStart)
			sys.stderr.flush()
			ret = WAIT_TIMEOUT
			while ret != WAIT_OBJECT_0 and self.ok.value:
				YieldProcessor()
				if ret == WAIT_ABANDONED:
					print >>sys.stderr, 'WAIT_ABANDONED errno=%d ok=%d' %(GetLastError(), self.ok.value)
				ret = WaitForSingleObject( self.evt, 100 )
			tEnd = HRTime()
			if self.ok.value:
				print >>sys.stderr, '##%lx WaitForSingleObject(nudgeEvent,INFINITE) = %lu at t=%g; sleep(1ms) and then send return nudge' %(currentThread().ident, ret, tEnd - tStart)
				#sys.stderr.flush()
				usleep(1000)
				print >>sys.stderr, '##%lx t=%g SetEvent(nudgeEvent) = %d' %(currentThread().ident, HRTime() - tStart, SetEvent(self.evt))
			else:
				print >>sys.stderr, '##%lx (bgThread2Nudge,evt) should exit, t=%g' %(currentThread().ident, HRTime() - tStart)
		elif self.sleepTime:
			print >>sys.stderr, '##%lx bgThread2Nudge starting to sleep nudge %ss at t=%g' %(currentThread().ident, self.sleepTime.value, HRTime() - tStart)
			usleep( int(self.sleepTime.value * 1e6) )
			print >>sys.stderr, '##%lx (bgThread2Nudge,sleeper) will exit, t=%g' %(currentThread().ident, HRTime() - tStart)
		elif self.sem:
			hh = OpenSemaphore( DELETE|SYNCHRONIZE|SEMAPHORE_MODIFY_STATE, 0, "cseSem")
			print >>sys.stderr, '##%lx bgThread2Nudge starting to wait for semaphore %s release at t=%g' %(currentThread().ident, self.sem.asString().c_str(), HRTime() - tStart)
			sys.stderr.flush()
			ret = WaitForSingleObject( hh, INFINITE )
			tEnd = HRTime()
			print >>sys.stderr, '##%lx WaitForSingleObject( hh, INFINITE ) = %s at t=%g; sleep(1ms) and ReleaseSemaphore(hh,1,None)' %(currentThread().ident, ret, tEnd - tStart)
			Sleep(500);
			print >>sys.stderr, '##%lx t=%g ReleaseSemaphore(hh,1,None) = %s' %(currentThread().ident, HRTime() - tStart, ReleaseSemaphore(hh, 1, None))
			CloseHandle(hh)
			print >>sys.stderr, '##%lx (bgThread2Nudge,sem) will exit, t=%g' %(currentThread().ident, HRTime() - tStart)
		sys.stderr.flush()


if not useMP:
	bgThread2Nudge = bgThread(sleeper=5.0)
	tStart = HRTime()
	print >>sys.stderr, '\n>%lx t=%g will start %s and WaitForSingleObject on it (should take %ss)' %(currentThread().ident, HRTime() - tStart, bgThread2Nudge, bgThread2Nudge.sleepTime.value)
	bgThread2Nudge.start()
	print >>sys.stderr, 'started'
	usleep(1000)
	print >>sys.stderr, 'slept 1m'
	ret = WaitForSingleObject(bgThread2Nudge.threadHandle,10000)
	tEnd = HRTime()
	print >>sys.stderr, '>%lx WaitForSingleObject(%s, 10000) = %s at t=%g' %(currentThread().ident, bgThread2Nudge.threadHandle, ret, tEnd - tStart)

nudgeEvent = CreateEvent(None, False, False, None)
bgThread2Nudge = bgThread(evt=nudgeEvent)

tStart = HRTime()
bgThread2Nudge.start()
usleep(1000000);
print >>sys.stderr, '\n>%lx t=%g SetEvent(nudgeEvent(%s)) = %d; sleep(1ms) and then wait for return nudge' %(currentThread().ident, HRTime() - tStart, nudgeEvent, SetEvent(nudgeEvent))
sys.stderr.flush()
usleep(1000);
ret = WAIT_TIMEOUT
while ret != WAIT_OBJECT_0:
	YieldProcessor()
	if ret == WAIT_ABANDONED:
		print >>sys.stderr, 'WAIT_ABANDONED errno=%d' %(GetLastError())
	ret = WaitForSingleObject( nudgeEvent, 100 )
tEnd = HRTime()
print >>sys.stderr, '>%lx WaitForSingleObject( nudgeEvent, INFINITE ) = %lu at t=%g' %(currentThread().ident, ret, tEnd - tStart)
sys.stderr.flush()

bgThread2Nudge.ok.value = False
ret = bgThread2Nudge.join(5); tEnd = HRTime();
print >>sys.stderr, '>%lx bgThread2Nudge.join(5)=%s at t=%g' %(currentThread().ident, ret, tEnd - tStart)
sys.stderr.flush()
CloseHandle(nudgeEvent)

print >>sys.stderr,'----------------'
nudgeEvent = CreateSemaphore( None, 0, 0x7FFFFFFF, "cseSem" )
bgThread2Nudge = bgThread(sem=nudgeEvent)
tStart = HRTime()
bgThread2Nudge.start()
Sleep(1000)
print >>sys.stderr, '>%lx t=%g ReleaseSemaphore(nudgeEvent,1,None) = %s; sleep(1ms) and then wait for return nudge' %(currentThread().ident, HRTime() - tStart, ReleaseSemaphore(nudgeEvent, 1, None) );
Sleep(1);
ret = WaitForSingleObject( nudgeEvent, INFINITE )
tEnd = HRTime()
print >>sys.stderr, '>%lx WaitForSingleObject( nudgeEvent, INFINITE ) = %s at t=%g' %(currentThread().ident, ret, tEnd - tStart)
ret = bgThread2Nudge.join(5)
tEnd = HRTime();
print >>sys.stderr, '>%lx bgThread2Nudge.join(5)=%s at t=%g' %(currentThread().ident, ret, tEnd - tStart)
CloseHandle(nudgeEvent)
