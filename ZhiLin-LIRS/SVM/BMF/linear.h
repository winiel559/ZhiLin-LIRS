#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	int *y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */  
};

enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, 
L2R_L1LOSS_SVC_PEGASOS, L2R_L1LOSS_SVC_PEGASOS_INNER, L2R_L2LOSS_SVC_DUAL_AVG,L2R_L1LOSS_SVC_DUAL_AVG, L2R_L1LOSS_SVC_DUAL_SUBSAMPLE}; /* solver_type */

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double inner_eps;	/* inner stopping criteria */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
	int max_iter;
	int inner_max_iter;
	int is_perm;
	int showprimal;
	int data_size_vs_acc;
	struct problem *prob_t;
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class (label[n]) */
	double bias;
};

struct model* train(const struct problem *prob, const struct parameter *param);
void cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, int *target);

int predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
int predict(const struct model *model_, const struct feature_node *x);
int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);
const char *check_parameter(const struct problem *prob, const struct parameter *param);
void set_print_string_function(void (*print_func) (const char*));


void solve_l2r_l1l2_svc(
	const problem *prob, double *w, 
	double *alpha, double eps, 
	double Cp, double Cn, int solver_type,
	double *_PGmax, double *_PGmin, int max_iter, bool* solved);
void pegasos_update_subgradient(const problem *prob, int n, double *w, double lambda, double eta);
void pegasos_solve_subprob(const problem *prob, int n, double *a, double *w, double *wnorms, double lambda, int t);

class Solver_MCSVM_CS
{
	public:
		Solver_MCSVM_CS(const problem *prob, int nr_class, double *C, double eps=0.1, int max_iter=100000);
		~Solver_MCSVM_CS();
		void Solve(double *w, double *alpha, double *_stopping, bool *solved);
	private:
		void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new);
		bool be_shrunk(int i, int m, int yi, double alpha_i, double minG);
		double *B, *C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};
double calulate_loss( const problem *prob, int solver_type, double *w);

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

