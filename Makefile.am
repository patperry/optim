# Process this file with automake to produce Makefile.in
#
ACLOCAL_AMFLAGS = -I m4 --install

AM_CPPFLAGS = \
	-Isrc -Ilib/libf2c

noinst_LIBRARIES = \
	liboptim.a \
	libf2c.a \
	libminpack.a

liboptim_a_SOURCES = \
	src/bfgs.c \
	src/bfgs.h \
	src/linesearch.c \
	src/linesearch.h

libf2c_a_SOURCES = \
	lib/libf2c/f2c.h \
	lib/libf2c/s_cmp.c \
	lib/libf2c/s_copy.c

libminpack_a_SOURCES = \
	lib/minpack/dcsrch.c \
	lib/minpack/dcstep.c

libminpack_a_CPPFLAGS = \
	-I./lib/libf2c \
	$(AM_CPPFLAGS)

