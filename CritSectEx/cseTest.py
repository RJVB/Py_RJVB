import sys,os
try:
	from multiprocessing import Process0 as Thread
	from multiprocessing import current_process as currentThread
except:
	from threading import Thread,currentThread
from MSEmul import *
from HRTime import *

tStart = 0

class bgThread(Thread):
	def __init__(self,evt):
		Thread.__init__(self)
		self.evt = evt
		self.ok = True

	def run(self):
		global tStart
		print >>sys.stderr, '##%lx bgThread2Nudge starting to wait for nudge event %s at t=%g' %(currentThread().ident, self.evt, HRTime() - tStart)
		sys.stderr.flush()
		ret = WAIT_TIMEOUT
		while ret != WAIT_OBJECT_0 and self.ok:
			YieldProcessor()
			if ret == WAIT_ABANDONED:
				print >>sys.stderr, 'WAIT_ABANDONED errno=%d ok=%d' %(GetLastError(), self.ok)
			ret = WaitForSingleObject( self.evt, 100 )
		tEnd = HRTime()
		print >>sys.stderr, '##%lx WaitForSingleObject(nudgeEvent,INFINITE) = %lu at t=%g; sleep(1ms) and then send return nudge' %(currentThread().ident, ret, tEnd - tStart)
		#sys.stderr.flush()
		usleep(1000)
		print >>sys.stderr, '##%lx t=%g SetEvent(nudgeEvent) = %d' %(currentThread().ident, HRTime() - tStart, SetEvent(self.evt))
		sys.stderr.flush()

nudgeEvent = CreateEvent(None, False, False, None)
bgThread2Nudge = bgThread(nudgeEvent)

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
bgThread2Nudge.ok = False
ret = bgThread2Nudge.join(5); tEnd = HRTime();
print >>sys.stderr, '>%lx bgThread2Nudge.join(5)=%s at t=%g' %(currentThread().ident, ret, tEnd - tStart)
sys.stderr.flush()
