#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "linear.h"
#include "block.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void exit_with_help()
{
	printf(
	"Usage: blocktrain [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
	"	1 -- BLOCK L2-loss support vector machines (dual)\n"	
	"	3 -- BLOCK L1-loss support vector machines (dual)\n"
	"	4 -- multi-class support vector machines by Crammer and Singer\n"
	"	7 -- BLOCK-P-B Pegasos (single update for each block)\n"
	"	8 -- BLOCK-P-I Pegasos (multiple updates for each block)\n"
	"	9 -- BLOCK-A-2 L2-loss support vector machine (block average)\n"
	"	10 -- BLOCK-A-1 L1-loss support vector machine (block average)\n"
	"	11 -- BLOCK L1-Loss support vector machine (subsample)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 1, 3, and 4\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"-E inner_epsilon\n"
	"-M inner_max_iter (Use this to conduct BLOCK-L-N)\n"
	"-m max_iter (default 2000)\n"
	"-p turn on inter-block permutation (default off)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default 1)\n"
	"-wi weight : weights adjust the parameter C of different classes (see README for details)\n"
	"-v n : n-fold cross validation mode\n"
	"-t test_file: test_file in compression format\n"
	"-S nblocks : Single file mode, all data stored in memory, and set nBlocks\n"
	"-o : show the primal value\n"
	"-l : number of training instances\n"
	"-n : average number of non-zero features of one training instance\n"
	"-a : number of classes\n"
	);
	exit(1);
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void do_cross_validation();

struct parameter param;
BlockProblem bprob;
BinaryProblem prob_t;
struct model* model_;
int flag_cross_validation;
int nr_fold;
double bias;
int nBlocks; // >0 if SINGLE mode is on(i.e. -S nBlocks)


int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
	//read_problem(input_file_name);
	strcpy(bprob.input_file_name, input_file_name);

	printf("input_file_name %s\n",input_file_name);
	printf("model_file_name %s\n",model_file_name);
	//printf("nBlocks: %d\n", nBlocks);	-S is for single file mode
	printf("input file name: %s\n", bprob.input_file_name);	
		
	bprob.read_meta(input_file_name);
	
	bprob.setBias(bias);
	error_msg = check_parameter(NULL,&param);
	
	if(prob_t.l > 0) {
		prob_t.setBias(bprob.n, bias);
		param.prob_t = prob_t.get_problem();
	}

	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}

	model_=blocktrain(&bprob, &param);
	//save_model(model_file_name, model_);
	free_and_destroy_model(&model_);
	
	destroy_param(&param);

	return 0;
}

void do_cross_validation(){


	double cvrate = block_cross_validation(&bprob,&param,nr_fold);

	printf("Cross Validation Accuracy = %g%%\n",100.0*cvrate);
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;

	// default values
	param.solver_type = L2R_L1LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.inner_eps = 1e-1;
	param.max_iter = 2000;
	param.inner_max_iter = 200;
	param.nr_weight = 0;
	param.is_perm = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.prob_t = NULL;
	param.showprimal = 0;
	flag_cross_validation = 0;
	bias = -1;
	nBlocks = 0;

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

			case 'E':
				param.inner_eps = atof(argv[i]);
				break;

			case 'm':
				param.max_iter = atoi(argv[i]);
				break;

			case 'M':
				param.inner_max_iter = atoi(argv[i]);
				break;

			case 'p':
				param.is_perm = 1;
				i--;
				break;

			case 'o':
				param.showprimal = 1;
				i--;
				break;
				
			case 'B':
				bias = atof(argv[i]);
				break;

			case 'S':
				nBlocks = atoi(argv[i]);
				bprob.nBlocks = atoi(argv[i]);
				break;
			
			case 'r':
				bprob.random_assign = atoi(argv[i]);
				break;
/*
			case 'l':
				bprob.l = atoi(argv[i]);
				break;

			case 'n':
				bprob.n = atoi(argv[i]);
				break;
			
			case 'a':
				bprob.nr_class = atoi(argv[i]);
				break;*/
			
			case 't':
				prob_t.load_problem(argv[i], COMPRESSION);
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


