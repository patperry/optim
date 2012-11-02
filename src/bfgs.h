#ifndef OPTIM_BFGS_H
#define OPTIM_BFGS_H

#include "linesearch.h"
#include <stddef.h>


#define BFGS_GTOL0	(1e-5)
#define BFGS_LSMAX0	(20)
#define BFGS_LSCTRL0	LINESEARCH_CTRL0

#define BFGS_CTRL0 { \
		BFGS_GTOL0, \
		BFGS_LSMAX0, \
		BFGS_LSCTRL0 \
	}

struct bfgs_ctrl {
	double gtol;
	size_t ls_maxit;
	struct linesearch_ctrl ls;
};

enum bfgs_task {
	BFGS_CONV = 0,
	BFGS_STEP = 1,
	BFGS_ERR_LNSRCH = -1,	// linesearch failed to converge
	BFGS_OVFLW_GRAD = -2,	// overflow in computing norm of gradient
};

struct bfgs {
	/* control/status */
	struct bfgs_ctrl ctrl;
	enum bfgs_task task;
	int first_step;

	/* current values */
	double f0;
	double *grad0;
	double *x0;

	/* next step */
	double *x;
	double *search;
	double *step;

	/* linsearch workspace */
	struct linesearch ls;
	size_t ls_it;

	/* inverse hessian estimate */
	double *inv_hess;

	/* auxiliary variables */
	double df;
	double *dg;
	double *H_dg;

	/* problem dimensions */
	size_t dim;
};

void bfgs_init(struct bfgs *opt, size_t dim, const struct bfgs_ctrl *ctrl);
void bfgs_deinit(struct bfgs *opt);

static inline size_t bfgs_dim(const struct bfgs *opt);

enum bfgs_task bfgs_start(struct bfgs *opt, const double *x0,
			  double f0, const double *grad0);
static inline const double *bfgs_next(const struct bfgs *opt);
enum bfgs_task bfgs_advance(struct bfgs *opt, double f,
			    const double *grad);
const char *bfgs_errmsg(enum bfgs_task task);

/* current values */
static inline const double *bfgs_cur(const struct bfgs *opt);
static inline double bfgs_value(const struct bfgs *opt);
static inline const double *bfgs_grad(const struct bfgs *opt);
static inline const double *bfgs_inv_hess(const struct bfgs *opt);

/* control parameters */
static inline int bfgs_ctrl_valid(const struct bfgs_ctrl *ctrl);

/* inline function definitions */
int bfgs_ctrl_valid(const struct bfgs_ctrl *ctrl)
{
	assert(ctrl);

	if (!(ctrl->gtol > 0)) {
		return 0;
	} else if (!(ctrl->ls_maxit > 0)) {
		return 0;
	} else {
		return linesearch_ctrl_valid(&ctrl->ls);
	}
}

size_t bfgs_dim(const struct bfgs *opt)
{
	return opt->dim;
}

const double *bfgs_cur(const struct bfgs *opt)
{
	assert(opt);
	return opt->x0;
}

const double *bfgs_next(const struct bfgs *opt)
{
	assert(opt);
	assert(opt->task == BFGS_STEP);
	return opt->x;
}

double bfgs_value(const struct bfgs *opt)
{
	assert(opt);
	return opt->f0;
}

const double *bfgs_grad(const struct bfgs *opt)
{
	assert(opt);
	return opt->grad0;
}

const double *bfgs_inv_hess(const struct bfgs *opt)
{
	assert(opt);
	if (opt->first_step) {
		return NULL;
	} else {
		return opt->inv_hess;
	}
}

#endif /* OPTIM_BFGS_H */
