#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>
#include "linear.h"
#include "binary.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
	"	0 -- L2-regularized logistic regression\n"
	"	1 -- L2-regularized L2-loss support vector classification (dual)\n"	
	"	2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	4 -- multi-class support vector classification by Crammer and Singer\n"
	"	5 -- L1-regularized L2-loss support vector classification\n"
	"	6 -- L1-regularized logistic regression\n"
	"-c cost : set the parameter C (default 1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n" 
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
	"		where f is the primal function and pos/neg are # of\n" 
	"		positive/negative data (default 0.01)\n"
	"	-s 1, 3, and 4\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_inf <= eps*min(pos,neg)/l*|f'(w0)|_inf,\n"
	"		where f is the primal function (default 0.01)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-p perm: if perm>0, do permutation else no (default 1)\n"
	"-n shrinking: if shrinking>0, do permutation else no (default 1)\n"
	"-x blocksize:  (default 20000) \n"
	"-t testfile: testfile\n"
	"-m iter: maximum iteration\n"
	"-o : show primal objective value\n"
	"-q : quiet mode (no outputs)\n"
	"-b : read compress input\n"
	"-E n: divide data into equal n part\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void read_problem_t(const char *filename);
void do_cross_validation();

struct feature_node *x_space;
struct feature_node *x_space_t;
struct parameter param;
struct problem prob;
struct problem prob_t;
struct model* model_;
int flag_cross_validation;
int nr_fold;
double bias;
int binary_mode;
struct binary_problem binprob;
struct binary_problem binprob_t;
const char* test_file_name;

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	time_t start_t;
	double total_time = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	parse_command_line(argc, argv, input_file_name, model_file_name);
	start_t = time(NULL);
	if(binary_mode) 
		binprob = read_binary(input_file_name, bias);
	else 
		read_problem(input_file_name);
	total_time = difftime(time(NULL), start_t);

	if(test_file_name) 
	{
		if(binary_mode)
		{
			binprob_t = read_binary(test_file_name, bias);
			param.prob_t = binprob_t.prob;
		}
		else 
		{
			read_problem_t(test_file_name);
			param.prob_t = &prob_t;
		}
	}

	printf("iter 0 time %lf runtime 0 loadtime %lf cputime 0 obj 0 roc 100 acc 0\n", total_time, total_time);
	error_msg = check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}

	if(flag_cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		if(binary_mode)
			model_=train(binprob.prob, &param);
		else 
			model_=train(&prob, &param);
	//	save_model(model_file_name, model_);
		destroy_model(model_);
	}
	destroy_param(&param);
	if(binary_mode) 
		destroy_binary_problem(&binprob);
	else 
	{
		free(prob.y);
		free(prob.x);
		free(x_space);
	}

	if(test_file_name )
	{
		if(binary_mode) 
			destroy_binary_problem(&binprob_t);
		else 
		{
			free(prob_t.y);
			free(prob_t.x);
			free(x_space_t);
		}
	}
	free(line);

	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	problem *prob_ = (binary_mode)? binprob.prob: &prob;
	int l = prob_->l;
	int *target = Malloc(int, l);

	cross_validation(prob_,&param,nr_fold,target);

	for(i=0;i<l;i++)
		if(target[i] == prob_->y[i])
			++total_correct;
	printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/l);

	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.max_iter = 10;
	param.perm = 1;
	param.blocksize = 20000;
	param.prob_t = NULL;
	param.shrinking = 1;
	param.showprimal = 0;
	param.equal_part = 0;
	flag_cross_validation = 0;
	binary_mode = 0;
	bias = -1;
	test_file_name = NULL;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'c':
				param.C = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'p':
				param.perm = atoi(argv[i]);
				break;

			case 'x':
				param.perm = atoi(argv[i]);
				break;

			case 't':
				test_file_name = argv[i];
				break;

			case 'h':
				param.shrinking = atoi(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;

			case 'm':
				param.max_iter = atoi(argv[i]);
				break;

			case 'o':
				param.showprimal = 1;
				i--;
				break;

			case 'q':
				liblinear_print_string = &print_null;
				i--;
				break;

			case 'b':
				binary_mode = 1;
				i--;
				break;
			case 'E':
				param.equal_part = atoi(argv[i]);
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}

	if(param.eps == INF)
	{
		if(param.solver_type == L2R_LR || param.solver_type == L2R_L2LOSS_SVC)
			param.eps = 0.01;
		else if(param.solver_type == L2R_L2LOSS_SVC_DUAL || param.solver_type == L2R_L1LOSS_SVC_DUAL || param.solver_type == MCSVM_CS)
			param.eps = 0.1;
		else if(param.solver_type == L1R_L2LOSS_SVC || param.solver_type == L1R_LR)
			param.eps = 0.01;
	}
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++;
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(int,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t");
		prob.y[i] = (int) strtol(label,&endptr,10);
		if(endptr == label)
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n; 
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	fclose(fp);
}

void read_problem_t(const char *filename)
{
	int max_index, inst_max_index, i;
	long elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob_t.l = 0;
	elements = 0;
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++;
		prob_t.l++;
	}
	rewind(fp);

	prob_t.bias=bias;

	prob_t.y = Malloc(int,prob_t.l);
	prob_t.x = Malloc(struct feature_node *,prob_t.l);
	x_space_t = Malloc(struct feature_node,elements+prob_t.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob_t.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob_t.x[i] = &x_space_t[j];
		label = strtok(line," \t");
		prob_t.y[i] = (int) strtol(label,&endptr,10);
		if(endptr == label)
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space_t[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space_t[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space_t[j].index;

			errno = 0;
			x_space_t[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob_t.bias >= 0)
			x_space_t[j++].value = prob_t.bias;

		x_space_t[j++].index = -1;
	}

	if(prob_t.bias >= 0)
	{
		prob_t.n=max_index+1;
		for(i=1;i<prob_t.l;i++)
			(prob_t.x[i]-2)->index = prob_t.n; 
		x_space_t[j-2].index = prob_t.n;
	}
	else
		prob_t.n=max_index;

	fclose(fp);
}
