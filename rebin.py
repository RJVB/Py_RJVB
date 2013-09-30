# from http://www.scipy.org/Cookbook/Rebinning
import numpy as n
import scipy.interpolate

def xvals(dims):
	'''Returns a volume with the appropriate x index value in each
	element.'''
	evList = ['n.fromfunction( xi, (%d' % dims[0] ] + \
			 [', %d' % dims[i+1] for i in range( len(dims) - 1 )] + \
			 [') )']
	return eval( ''.join(evList) )

def nvals(n, dims):
	'''Returns xvals-like volumes, indexing over any dimension n.
	Will probably crash and burn if n >= len(dims).'''
	evList = ''
	for i in range( len(dims) - 1):
		if i < n:
			evList = evList + '%d' % dims[i]
		else:
			evList = evList + '%d' % dims[i + 1]
		if i < len(dims) - 2:
			evList = evList + ', '
	evList = 'xvals( (%d,' % dims[n] + evList + ')' + ')'
	xs = eval( evList )

	evList = ''
	for i in range( len(dims) ):
		if i < n:
			ind = i + 1
		elif i == n:
			ind = 0
		else:
			ind = i
		evList = evList + '%d' % ind
		if i < len(dims) - 1:
			evList = evList + ', '
	evList = 'xs.transpose(' + evList + ')'
	return eval( evList )

def congrid(a, nud, method='neighbour', centre=False, minusone=False):
	'''Arbitrary resampling of source array to new dimension sizes.

	Example:
	rebinned = rebin.congrid( raw, (2,2,2), \
		 method='cubic', minusone=False, centre=False)

	Parameters:
	arg0: input array
	arg1: tuple of resulting dimensions

	method:
	neighbour - closest value from original data
	the ''kinds'' supported by scipy.interpolate.interp1d

	centre:
	True - interpolation points are at the centres of the bins
	False - points are at the front edge of the bin

	minusone:
	For example- inarray.shape = (i,j) & new dimensions = (x,y)
	False - inarray is resampled by factors of (i/x) * (j/y)
	True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
	This prevents extrapolation one element beyond bounds of input
	array.

	Currently only supports maintaining the same number of dimensions.
	Also doesn''t work for 1-D arrays unless promoted to shape (x,1).

	Based loosely on IDL''s congrid routine, which apparently originally
	came from a VAX/VMS routine of the same name.

	I''m not completely sure of the validity of using parallel 1-D
	interpolations repeated along each axis in succession, but the
	results are visually compelling.
	'''

	if not a.dtype.type in [n.typeDict['Float32'], n.typeDict['Float64']]:
		print "Converting to float"
		a = a.astype('Float32')

	if minusone:
		m1 = 1.
	else:
		m1 = 0.

	if centre:
		ofs = 0.5
	else:
		ofs = 0.

	old = n.asarray( a.shape )
	ndims = len( old )
	if len( nud ) != ndims:
		print "Congrid: dimensions error. This routine currently only support rebinning to the same number of dimensions."
		return None

	nudr = n.asarray( nud ).astype('Float32')

	dimlist = []

	if method == 'neighbour':
		for i in range( ndims ):
			base = nvals(i, nudr)
			dimlist.append( (old[i] - m1) / (nudr[i] - m1) \
							* (base + ofs) - ofs )

		cd = n.array( dimlist )
		cdr = cd.round().astype( 'UInt16' )
		nua = a[list( cdr )]
		return nua

	elif method in ['nearest','linear','cubic','spline']:
		# calculate new dims
		for i in range( ndims ):
			base = n.arange( nudr[i] )
			dimlist.append( (old[i] - m1) / (nudr[i] - m1) \
							* (base + ofs) - ofs )

		# specify old dims
		olddims = [n.arange(i).astype('Float32') for i in list( a.shape )]

		# first interpolation - for ndims = any
		mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
		nua = mint( dimlist[-1] )

		trorder = [ndims - 1] + range( ndims - 1 )
		for i in range( ndims - 2, -1, -1 ):
			nua = nua.transpose( trorder )

			mint = scipy.interpolate.interp1d( olddims[i], nua, kind=method )
			nua = mint( dimlist[i] )

		if ndims > 1:
			# need one more transpose to return to original dimensions
			nua = nua.transpose( trorder )

		return nua

	else:
		print "Congrid error: Unrecognized interpolation type.\n", \
			  "This routine currently only supports \'nearest\',\'linear\',\'cubic\', and \'spline\'."
		return None
