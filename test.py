import HRTime
from math import *
from sincos import *
import rtsched
import numpy
from numpy import *


N=100000

realtime = False;

try:
	rtpars = rtsched.thread_policy_get(0, rtsched.THREAD_TIME_CONSTRAINT_POLICY )
	print "THREAD_TIME_CONSTRAINT_POLICY: ", rtpars

	def rt():
		if realtime:
			rtsched.thread_policy_set( 0, rtsched.THREAD_TIME_CONSTRAINT_POLICY, 5e3, 2.5e3, 3e3 )
except:
	def rt():
		pass

rt()
HRTime.tic()
for i in range(N):
	s=numpy.sin(i)
	c=numpy.cos(i)
print HRTime.toc()

rt()
HRTime.tic()
for i in range(N):
	(s,c)=sincos(i)
print HRTime.toc()

angles= array( [0,45,90,135,180,225,270,315,360] ) * 2 * pi / 360
print sincos( array( [0,45,90,135,180,225,270,315,360] ), 360 )
print sincos( angles )

rt()
HRTime.tic()
for i in range(N):
	s=numpy.sin(angles)
	c=numpy.cos(angles)
print HRTime.toc()
print (s,c)

rt()
HRTime.tic()
for i in range(N):
	(s,c)=sincos(angles)
print HRTime.toc()
print (s,c)

