# Generated automatically from Makefile.pre by makesetup.
# Generated automatically from Makefile.pre.in by sedscript.
# Universal Unix Makefile for Python extensions
# =============================================

# Short Instructions
# ------------------

# 1. Build and install Python (1.5 or newer).
# 2. "make -f Makefile.pre.in boot"
# 3. "make"
# You should now have a shared library.

# Long Instructions
# -----------------

# Build *and install* the basic Python 1.5 distribution.  See the
# Python README for instructions.  (This version of Makefile.pre.in
# only withs with Python 1.5, alpha 3 or newer.)

# Create a file Setup.in for your extension.  This file follows the
# format of the Modules/Setup.dist file; see the instructions there.
# For a simple module called "spam" on file "spammodule.c", it can
# contain a single line:
#   spam spammodule.c
# You can build as many modules as you want in the same directory --
# just have a separate line for each of them in the Setup.in file.

# If you want to build your extension as a shared library, insert a
# line containing just the string
#   *shared*
# at the top of your Setup.in file.

# Note that the build process copies Setup.in to Setup, and then works
# with Setup.  It doesn't overwrite Setup when Setup.in is changed, so
# while you're in the process of debugging your Setup.in file, you may
# want to edit Setup instead, and copy it back to Setup.in later.
# (All this is done so you can distribute your extension easily and
# someone else can select the modules they actually want to build by
# commenting out lines in the Setup file, without editing the
# original.  Editing Setup is also used to specify nonstandard
# locations for include or library files.)

# Copy this file (Misc/Makefile.pre.in) to the directory containing
# your extension.

# Run "make -f Makefile.pre.in boot".  This creates Makefile
# (producing Makefile.pre and sedscript as intermediate files) and
# config.c, incorporating the values for sys.prefix, sys.exec_prefix
# and sys.version from the installed Python binary.  For this to work,
# the python binary must be on your path.  If this fails, try
#   make -f Makefile.pre.in Makefile VERSION=1.5 installdir=<prefix>
# where <prefix> is the prefix used to install Python for installdir
# (and possibly similar for exec_installdir=<exec_prefix>).

# Note: "make boot" implies "make clobber" -- it assumes that when you
# bootstrap you may have changed platforms so it removes all previous
# output files.

# If you are building your extension as a shared library (your
# Setup.in file starts with *shared*), run "make" or "make sharedmods"
# to build the shared library files.  If you are building a statically
# linked Python binary (the only solution of your platform doesn't
# support shared libraries, and sometimes handy if you want to
# distribute or install the resulting Python binary), run "make
# python".

# Note: Each time you edit Makefile.pre.in or Setup, you must run
# "make Makefile" before running "make".

# Hint: if you want to use VPATH, you can start in an empty
# subdirectory and say (e.g.):
#   make -f ../Makefile.pre.in boot srcdir=.. VPATH=..


# === Bootstrap variables (edited through "make boot") ===

# The prefix used by "make inclinstall libainstall" of core python
installdir=	/Library/Frameworks/Python2.4.framework/Versions/2.4

# The exec_prefix used by the same
exec_installdir=/Library/Frameworks/Python2.4.framework/Versions/2.4

# Source directory and VPATH in case you want to use VPATH.
# (You will have to edit these two lines yourself -- there is no
# automatic support as the Makefile is not generated by
# config.status.)
srcdir=		.
VPATH=		.

# === Variables that you may want to customize (rarely) ===

# (Static) build target
TARGET=		python

# Installed python binary (used only by boot target)
PYTHON=		python

# Add more -I and -D options here
CFLAGS=		$(BASECFLAGS) $(OPT) -I$(INCLUDEPY) -I$(EXECINCLUDEPY) # $(DEFS)

# These two variables can be set in Setup to merge extensions.
# See example[23].
BASELIB=	
BASESETUP=	

# === Variables set by makesetup ===

MODOBJS=	
MODLIBS=	$(LOCALMODLIBS) $(BASEMODLIBS)

# === Definitions added by makesetup ===

LOCALMODLIBS=
BASEMODLIBS=         
SHAREDMODS= ./HRTime$(SO)
GLHACK=-Dclear=__GLclear
PYTHONPATH=$(COREPYTHONPATH)
COREPYTHONPATH=$(DESTPATH)$(SITEPATH)$(TESTPATH)$(MACHDEPPATH)$(EXTRAMACHDEPPATH)$(TKPATH)
TKPATH=:lib-tk
EXTRAMACHDEPPATH=
MACHDEPPATH=:plat-$(MACHDEP)
TESTPATH=
SITEPATH=
DESTPATH=
MACHDESTLIB=$(BINLIBDEST)
DESTLIB=$(LIBDEST)

# === Variables from configure (through sedscript) ===

VERSION=	2.4
CC=		gcc-4.0.0
LINKCC=		$(PURIFY) $(CC)
SGI_ABI=	
BASECFLAGS= -fno-strict-aliasing -fno-common -dynamic
OPT=		-DNDEBUG -mcpu=G5 -mtune=G5 -mpowerpc -mpowerpc-gfxopt -mpowerpc-gpopt -mpowerpc64 -mhard-float -mpowerpc-gfxopt -mnew-mnemonics -mhard-float -mnew-mnemonics -mstring -misel=yes -fomit-frame-pointer -fdollars-in-identifiers -O3 -fexpensive-optimizations -frerun-cse-after-loop -fschedule-insns -fschedule-insns2 -finline-functions -funroll-loops -fno-strict-aliasing -mfused-madd -ftree-vectorize -ftracer -faltivec
LDFLAGS=	
LDLAST=				
DEFS=		@DEFS@
LIBS=		-ldl 
LIBM=		
LIBC=		
RANLIB=		ranlib
MACHDEP=	darwin
SO=		.so
LDSHARED=	$(CC) $(LDFLAGS) -bundle -undefined dynamic_lookup /Library/Frameworks/Python2.4.framework/Versions/$(VERSION)/$(PYTHONFRAMEWORK)
CCSHARED=	
LINKFORSHARED=	-u _PyMac_Error $(PYTHONFRAMEWORKDIR)/Versions/$(VERSION)/$(PYTHONFRAMEWORK)
PYTHONFRAMEWORK= Python2.4
CXX=		g++-4.0.0

# Install prefix for architecture-independent files
prefix=		/Library/Frameworks/Python2.4.framework/Versions/2.4

# Install prefix for architecture-dependent files
exec_prefix=	${prefix}

# Uncomment the following two lines for AIX
#LINKCC= 	$(LIBPL)/makexp_aix $(LIBPL)/python.exp "" $(LIBRARY); $(PURIFY) $(CC)
#LDSHARED=	$(LIBPL)/ld_so_aix $(CC) -bI:$(LIBPL)/python.exp

# === Fixed definitions ===

# Shell used by make (some versions default to the login shell, which is bad)
SHELL=		/bin/sh

# Expanded directories
BINDIR=		$(exec_installdir)/bin
LIBDIR=		$(exec_prefix)/lib
MANDIR=		$(installdir)/man
INCLUDEDIR=	$(installdir)/include
SCRIPTDIR=	$(prefix)/lib

# Detailed destination directories
BINLIBDEST=	$(LIBDIR)/python$(VERSION)
LIBDEST=	$(SCRIPTDIR)/python$(VERSION)
INCLUDEPY=	$(INCLUDEDIR)/python$(VERSION)
EXECINCLUDEPY=	$(exec_installdir)/include/python$(VERSION)
LIBP=		$(exec_installdir)/lib/python$(VERSION)
DESTSHARED=	$(BINLIBDEST)/site-packages

LIBPL=		$(LIBP)/config

PYTHONLIBS=	$(LIBPL)/libpython$(VERSION).a

MAKESETUP=	$(LIBPL)/makesetup
MAKEFILE=	$(LIBPL)/Makefile
CONFIGC=	$(LIBPL)/config.c
CONFIGCIN=	$(LIBPL)/config.c.in
SETUP=		$(LIBPL)/Setup.config $(LIBPL)/Setup.local $(LIBPL)/Setup

SYSLIBS=	$(LIBM) $(LIBC)

ADDOBJS=	$(LIBPL)/python.o config.o

# Portable install script (configure doesn't always guess right)
INSTALL=	$(LIBPL)/install-sh -c
# Shared libraries must be installed with executable mode on some systems;
# rather than figuring out exactly which, we always give them executable mode.
# Also, making them read-only seems to be a good idea...
INSTALL_SHARED=	${INSTALL} -m 555

# === Fixed rules ===

# Default target.  This builds shared libraries only
default:	sharedmods

# Build everything
all:		static sharedmods

# Build shared libraries from our extension modules
sharedmods:	$(SHAREDMODS)

# Build a static Python binary containing our extension modules
static:		$(TARGET)
$(TARGET):	$(ADDOBJS) lib.a $(PYTHONLIBS) Makefile $(BASELIB)
		$(LINKCC) $(LDFLAGS) $(LINKFORSHARED) $(ADDOBJS) lib.a $(PYTHONLIBS) $(LINKPATH) $(BASELIB) $(MODLIBS) $(LIBS) $(SYSLIBS) -o $(TARGET) $(LDLAST)

install:	sharedmods
		if test ! -d $(DESTSHARED) ; then \
			mkdir $(DESTSHARED) ; else true ; fi
		-for i in X $(SHAREDMODS); do \
			if test $$i != X; \
			then $(INSTALL_SHARED) $$i $(DESTSHARED)/$$i; \
			fi; \
		done

# Build the library containing our extension modules
lib.a:		$(MODOBJS)
		-rm -f lib.a
		ar cr lib.a $(MODOBJS)
		-$(RANLIB) lib.a 

# This runs makesetup *twice* to use the BASESETUP definition from Setup
config.c Makefile:	Makefile.pre Setup $(BASESETUP) $(MAKESETUP)
		$(MAKESETUP) \
		 -m Makefile.pre -c $(CONFIGCIN) Setup -n $(BASESETUP) $(SETUP)
		$(MAKE) -f Makefile do-it-again

# Internal target to run makesetup for the second time
do-it-again:
		$(MAKESETUP) \
		 -m Makefile.pre -c $(CONFIGCIN) Setup -n $(BASESETUP) $(SETUP)

# Make config.o from the config.c created by makesetup
config.o:	config.c
		$(CC) $(CFLAGS) -c config.c

# Setup is copied from Setup.in *only* if it doesn't yet exist
Setup:
		cp $(srcdir)/Setup.in Setup

# Make the intermediate Makefile.pre from Makefile.pre.in
Makefile.pre: Makefile.pre.in sedscript
		sed -f sedscript $(srcdir)/Makefile.pre.in >Makefile.pre

# Shortcuts to make the sed arguments on one line
P=prefix
E=exec_prefix
H=Generated automatically from Makefile.pre.in by sedscript.
L=LINKFORSHARED

# Make the sed script used to create Makefile.pre from Makefile.pre.in
sedscript:	$(MAKEFILE)
	sed -n \
	 -e '1s/.*/1i\\/p' \
	 -e '2s%.*%# $H%p' \
	 -e '/^VERSION=/s/^VERSION=[ 	]*\(.*\)/s%@VERSION[@]%\1%/p' \
	 -e '/^CC=/s/^CC=[ 	]*\(.*\)/s%@CC[@]%\1%/p' \
	 -e '/^CXX=/s/^CXX=[ 	]*\(.*\)/s%@CXX[@]%\1%/p' \
	 -e '/^LINKCC=/s/^LINKCC=[ 	]*\(.*\)/s%@LINKCC[@]%\1%/p' \
	 -e '/^BASECFLAGS=/s/^BASECFLAGS=[ 	]*\(.*\)/s%@BASECFLAGS[@]%\1%/p' \
	 -e '/^OPT=/s/^OPT=[ 	]*\(.*\)/s%@OPT[@]%\1%/p' \
	 -e '/^LDFLAGS=/s/^LDFLAGS=[ 	]*\(.*\)/s%@LDFLAGS[@]%\1%/p' \
	 -e '/^LDLAST=/s/^LDLAST=[      ]*\(.*\)/s%@LDLAST[@]%\1%/p' \
	 -e '/^DEFS=/s/^DEFS=[ 	]*\(.*\)/s%@DEFS[@]%\1%/p' \
	 -e '/^LIBS=/s/^LIBS=[ 	]*\(.*\)/s%@LIBS[@]%\1%/p' \
	 -e '/^LIBM=/s/^LIBM=[ 	]*\(.*\)/s%@LIBM[@]%\1%/p' \
	 -e '/^LIBC=/s/^LIBC=[ 	]*\(.*\)/s%@LIBC[@]%\1%/p' \
	 -e '/^RANLIB=/s/^RANLIB=[ 	]*\(.*\)/s%@RANLIB[@]%\1%/p' \
	 -e '/^MACHDEP=/s/^MACHDEP=[ 	]*\(.*\)/s%@MACHDEP[@]%\1%/p' \
	 -e '/^SO=/s/^SO=[ 	]*\(.*\)/s%@SO[@]%\1%/p' \
	 -e '/^LDSHARED=/s/^LDSHARED=[ 	]*\(.*\)/s%@LDSHARED[@]%\1%/p' \
	 -e '/^CCSHARED=/s/^CCSHARED=[ 	]*\(.*\)/s%@CCSHARED[@]%\1%/p' \
	 -e '/^SGI_ABI=/s/^SGI_ABI=[ 	]*\(.*\)/s%@SGI_ABI[@]%\1%/p' \
	 -e '/^$L=/s/^$L=[ 	]*\(.*\)/s%@$L[@]%\1%/p' \
	 -e '/^$P=/s/^$P=\(.*\)/s%^$P=.*%$P=\1%/p' \
	 -e '/^$E=/s/^$E=\(.*\)/s%^$E=.*%$E=\1%/p' \
	 -e '/^PYTHONFRAMEWORK=/s/^PYTHONFRAMEWORK=[ 	]*\(.*\)/s%@PYTHONFRAMEWORK[@]%\1%/p' \
	 $(MAKEFILE) >sedscript
	echo "/^installdir=/s%=.*%=	$(installdir)%" >>sedscript
	echo "/^exec_installdir=/s%=.*%=$(exec_installdir)%" >>sedscript
	echo "/^srcdir=/s%=.*%=		$(srcdir)%" >>sedscript
	echo "/^VPATH=/s%=.*%=		$(VPATH)%" >>sedscript
	echo "/^LINKPATH=/s%=.*%=	$(LINKPATH)%" >>sedscript
	echo "/^BASELIB=/s%=.*%=	$(BASELIB)%" >>sedscript
	echo "/^BASESETUP=/s%=.*%=	$(BASESETUP)%" >>sedscript

# Bootstrap target
boot:	clobber
	VERSION=`$(PYTHON) -c "import sys; print sys.version[:3]"`; \
	installdir=`$(PYTHON) -c "import sys; print sys.prefix"`; \
	exec_installdir=`$(PYTHON) -c "import sys; print sys.exec_prefix"`; \
	$(MAKE) -f $(srcdir)/Makefile.pre.in VPATH=$(VPATH) srcdir=$(srcdir) \
		VERSION=$$VERSION \
		installdir=$$installdir \
		exec_installdir=$$exec_installdir \
		Makefile

# Handy target to remove intermediate files and backups
clean:
		-rm -f *.o *~

# Handy target to remove everything that is easily regenerated
clobber:	clean
		-rm -f *.a tags TAGS config.c Makefile.pre $(TARGET) sedscript
		-rm -f *.so *.sl so_locations


# Handy target to remove everything you don't want to distribute
distclean:	clobber
		-rm -f Makefile Setup

# Rules appended by makedepend

./HRTime.o: $(srcdir)/./HRTime.c; $(CC) $(CCSHARED) $(CFLAGS) $(CPPFLAGS)  -c $(srcdir)/./HRTime.c -o ./HRTime.o
./HRTime$(SO):  ./HRTime.o; $(LDSHARED)  ./HRTime.o   -o ./HRTime$(SO)
