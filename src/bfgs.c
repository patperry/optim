#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "bfgs.h"

static void linesearch(struct bfgs *opt, double step0, double f0, double g0)
{
	size_t n = opt->dim;
	opt->ls_it = 0;

	memcpy(opt->x, opt->x0, n * sizeof(opt->x[0]));
	blas_daxpy(n, step0, opt->search, 1, opt->x, 1);

	linesearch_start(&opt->ls, step0, f0, g0, &opt->ctrl.ls);
}

static void update(struct bfgs *opt, double f, const double *grad)
{
	size_t i, n = opt->dim;
	double *H = opt->inv_hess;
	double *s = opt->step;
	double *y = opt->dg;

	/* update step */
	memcpy(s, opt->search, n * sizeof(s[0]));
	blas_dscal(n, linesearch_step(&opt->ls), s, 1);

	/* update df */
	opt->df = f - opt->f0;

	/* update dg */
	memcpy(y, grad, n * sizeof(y[0]));
	blas_daxpy(n, -1.0, opt->grad0, 1, y, 1);

	double s_y = blas_ddot(n, s, 1, y, 1);

	/* NOTE: could use damped update instead (Nocedal and Wright, p. 537) */
	assert(s_y > 0);

	/* initialize inv hessian on first step (Nocedal and Wright, p. 143) */
	if (opt->first_step) {	/*  */
		double y_y = blas_ddot(n, y, 1, y, 1);
		assert(y_y > 0);
		double scale = s_y / y_y;

		memset(H, 0, n * n * sizeof(H[0]));
		for (i = 0; i < n; i++) {
			H[i * n + i] = scale;
		}
		opt->first_step = 0;
	}

	/* compute H_y */
	double *H_y = opt->H_dg;
	blas_gemv(n, n, BLAS_NOTRANS, 1.0, H, n, y, 1, 0.0, H_y, 1);

	double y_H_y = blas_ddot(n, H_y, 1, y, 1);
	double scale1 = (1.0 + (y_H_y / s_y)) / s_y;
	double rho = 1.0 / s_y;

	/* update inverse hessian */
	blas_dger(n, n, scale1, s, 1, s, 1, H, n);
	blas_dger(n, n, -rho, H_y, 1, s, 1, H, n);
	blas_dger(n, n, -rho, s, 1, H_y, 1, H, n);

	/* update search direction */
	blas_dgemv(BLAS_NOTRANS, n, n, -1.0, opt->inv_hess, n, grad, 1,
		   0.0, opt->search, 1);
	assert(isfinite(blas_dnrm2(n, opt->search, 1)));

	/* update initial position, value, and grad */
	memcpy(opt->x0, opt->x, n * sizeof(opt->x0[0]));
	opt->f0 = f;
	memcpy(opt->grad0, grad, n * sizeof(opt->grad0[0]));
}

void bfgs_init(struct bfgs *opt, size_t n, const struct bfgs_ctrl *ctrl)
{
	assert(opt);
	assert(ctrl);
	assert(bfgs_ctrl_valid(ctrl));

	opt->ctrl = *ctrl;
	opt->inv_hess = xmalloc(n * n * sizeof(opt->inv_hess[0]));
	opt->search = xmalloc(n * sizeof(opt->search[0]));
	opt->grad0 = xmalloc(n * sizeof(opt->grad0[0]));
	opt->x0 = xmalloc(n * sizeof(opt->x0[0]));
	opt->x = xmalloc(n * sizeof(opt->x[0]));
	opt->step = xmalloc(n * sizeof(opt->step[0]));
	opt->dg = xmalloc(n * sizeof(opt->dg[0]));
	opt->H_dg = xmalloc(n * sizeof(opt->H_dg[0]));
}

void bfgs_deinit(struct bfgs *opt)
{
	free(&opt->H_dg);
	free(opt->dg);
	free(opt->step);
	free(opt->x);
	free(opt->x0);
	free(opt->grad0);
	free(opt->search);
	free(opt->inv_hess);
}

enum bfgs_task bfgs_start(struct bfgs *opt, const double *x0,
			  double f0, const double *grad0)
{
	assert(opt);
	assert(x0);
	assert(isfinite(f0));
	assert(grad0 || !bfgs_dim(opt));

	size_t n = opt->dim;
	double step0 = 1.0;

	opt->first_step = 1;
	memcpy(opt->x0, x0, n * sizeof(opt->x0[0]));
	opt->f0 = f0;
	memcpy(opt->grad0, grad0, n * sizeof(opt->grad0[0]));

	double scale = blas_dnrm2(n, grad0, 1);
	assert(scale == 0 || n > 0);
	assert(!isnan(scale));

	if (!isfinite(scale)) {
		opt->task = BFGS_OVFLW_GRAD;
	} else if (scale > 0) {
		opt->task = BFGS_STEP;

		memcpy(opt->search, grad0, n * sizeof(opt->search[0]));
		blas_dscal(n, -1.0 / scale, opt->search, 1);
		double g0 = -scale;
		linesearch(opt, step0, f0, g0);
	} else {
		opt->task = BFGS_CONV;
	}
	return opt->task;
}

static int converged(const struct bfgs *opt)
{
	size_t i, n = bfgs_dim(opt);
	double gtol = opt->ctrl.gtol;

	/*fprintf(stderr, "|df| = %.22f; |grad| = %.22f; |step| = %.22f\n",
	       fabs(opt->df /MAX(1, opt->f0)),
	      blas_dnrm2(n, opt->grad0, 1), blas_dnrm2(n, opt->step, 1)); */

	for (i = 0; i < n; i++) {
		double x = opt->x0[i];
		double g = opt->grad0[i];

		if (!(fabs(g) < gtol * MAX(1.0, fabs(x))))
			return 0;
	}
	return 1;
}

enum bfgs_task bfgs_advance(struct bfgs *opt, double f,
			    const double *grad)
{
	size_t n = opt->dim;

	assert(isfinite(f));
	assert(isfinite(blas_dnrm2(n, grad, 1)));
	assert(opt->task == BFGS_STEP);

	double g = blas_ddot(n, grad, 1, opt->search, 1);
	enum linesearch_task lstask = linesearch_advance(&opt->ls, f, g);
	int ok = linesearch_sdec(&opt->ls) && linesearch_curv(&opt->ls);

	switch (lstask) {
	case LINESEARCH_CONV:
		break;

	case LINESEARCH_STEP:
		opt->ls_it++;

		if (opt->ls_it < opt->ctrl.ls_maxit) {
			memcpy(opt->x, opt->x0, n * sizeof(opt->x[0]));
			blas_daxpy(n, linesearch_step(&opt->ls), opt->search, 1,
			           opt->x, 1);

			assert(opt->task == BFGS_STEP);
			goto out;
		} else if (ok) {
			break;
		} else {
			opt->task = BFGS_ERR_LNSRCH;	/* maximum number of iterations */
		}
	default:
		if (ok) {
			break;
		} else {
			opt->task = BFGS_ERR_LNSRCH;
			goto out;
		}
	}

	update(opt, f, grad);

	/* test for convergence */
	if (converged(opt)) {
		opt->task = BFGS_CONV;
	} else {
		assert(opt->task == BFGS_STEP);

		double step0 = 1.0;
		double f0 = f;
		double g0 = blas_ddot(n, grad, 1, opt->search, 1);
		assert(g0 < 0);

		linesearch(opt, step0, f0, g0);
	}
out:
	return opt->task;
}

const char *bfgs_errmsg(enum bfgs_task task)
{
	switch (task) {
	case BFGS_STEP:
		return "optimization in progress";
	case BFGS_ERR_LNSRCH:
		return "linesearch failed";
	case BFGS_OVFLW_GRAD:
		return "overflow computing norm of gradient";
	case BFGS_CONV:
		return NULL;
	}
	assert(0);
	return NULL;
}
