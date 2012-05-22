from sincos import *
import HRTime
from numpy import *
from time import clock
from math import pi
import rtsched

realtime = True;

loops = 100000/5

if realtime:
	try:
		rtpars = rtsched.thread_policy_get(0, rtsched.THREAD_TIME_CONSTRAINT_POLICY )
		print "THREAD_TIME_CONSTRAINT_POLICY: ", rtpars

		def rt():
			rtsched.thread_policy_set( 0, rtsched.THREAD_TIME_CONSTRAINT_POLICY, 5e3, 2.5e3, 3e3 )
	except:
		def rt():
			pass
else:
	def rt():
		pass

X = arange(-10*loops, 10*loops) / float(loops)
X5 = arange(-10*loops*5, 10*loops*5) / float(loops*5)

rt()
start=clock()
HRTime.tic()
for x in X:
	(s,c) = sincos_noop(x)
overhead = HRTime.toc()
overhead_CPU = clock() - start

print 'Python calling/argument parsing overhead: %d calls in %g:%g seconds\n' %(loops, overhead_CPU, overhead)

rt()
start=clock()
HRTime.tic()
for x in X:
	s=sin(x)
	c=cos(x)
sin_cos_time = HRTime.toc()
sin_cos_CPU = clock() - start

print 'sin(x), cos(x) : %d calls in %g:%g seconds, corrected %g:%g sec\n' %(loops, sin_cos_CPU, sin_cos_time, sin_cos_CPU-overhead_CPU, sin_cos_time-overhead)

rt()
start=clock()
HRTime.tic()
for x in X5:
	(s,c) = sincos(x)
sincos_time = HRTime.toc()
sincos_CPU = clock() - start
sincos_time /= 5
sincos_CPU /= 5

print 'sincos(x) : %d calls in %g:%g seconds, corrected %g:%g sec\n' %(loops, sincos_CPU, sincos_time, sincos_CPU-overhead_CPU, sincos_time-overhead)

rt()
start=clock()
HRTime.tic()
for x in X:
	(s,c) = mips_sincos(x)
mips_sincos_time = HRTime.toc()
mips_sincos_CPU = clock() - start

print 'mips_sincos(x) : %d calls in %g:%g seconds, corrected %g:%g sec\n' %(loops, mips_sincos_CPU, mips_sincos_time, mips_sincos_CPU-overhead_CPU, mips_sincos_time-overhead)

X *= 360.0/(2*pi)
rt()
start=clock()
HRTime.tic()
for x in X:
	(s,c) = cephes_sincos(x,0)
cephes_sincos_time = HRTime.toc()
cephes_sincos_CPU = clock() - start

print 'cephes_sincos(x,0) : %d calls in %g:%g seconds, corrected %g:%g sec\n' %(loops, cephes_sincos_CPU, cephes_sincos_time, cephes_sincos_CPU-overhead_CPU, cephes_sincos_time-overhead)

rt()
start=clock()
HRTime.tic()
for x in X:
	(s,c) = cephes_sincos(x,1)
cephes_sincos_time = HRTime.toc()
cephes_sincos_CPU = clock() - start

print 'cephes_sincos(x,1) : %d calls in %g:%g seconds, corrected %g:%g sec\n' %(loops, cephes_sincos_CPU, cephes_sincos_time, cephes_sincos_CPU-overhead_CPU, cephes_sincos_time-overhead)

