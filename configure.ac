AC_INIT([optim], [0.1], [patperry@gmail.com])

AC_CONFIG_SRCDIR([src/bfgs.h])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])

AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# Checks for programs
AC_PROG_CC
AC_PROG_RANLIB
m4_ifdef([AM_PROG_AR], [AM_PROG_AR]) ## requires automake >= 1.12
AM_PROG_CC_C_O

# Checks for libraries
AC_CHECK_LIBM

# Checks for header files
AC_HEADER_ASSERT

# Checks for typedefs, structures, and compiler characteristics
AC_C_INLINE

# Checks for library functions
# Checks for system services

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
