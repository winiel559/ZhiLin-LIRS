#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include "linear.h"


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


int max_nr_attr = 64;
struct parameter param;

double *wresize(double *w, int *n, int size) {
	int i;
	double *neww = w;
	if (size > *n) 
	{
		neww = (double*) realloc(w, sizeof(double)*size);
		for(i=*n;i<size;i++) neww[i] = 0;
		*n = size;
	}
	return neww;
}

void streamtrain(const char *train_file_name, const char *model_file_name){
	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	FILE *input = fopen(train_file_name, "r");
	
	double R = 0, si = 1;
	double *w = NULL;
	double C = param.C;
	double wnorm2 = 0;
	int n = 0, l = 0;
	struct feature_node *x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));

	printf("C = %g\n", C);

	time_t start_t = time(NULL);

	// Read an instance once a time!
	while(readline(input) != NULL) {
		int i = 0;
		int target_label, y;
		char *idx, *val, *label, *endptr;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		// Read an instance!
		label = strtok(line," \t");
		target_label = (int) strtol(label,&endptr,10);
		if(endptr == label)
			exit_input_error(l+1);
		y = target_label;
		
		while(1)
		{
			if(i>=max_nr_attr-2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,max_nr_attr*sizeof(struct feature_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(l+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(l+1);
			++i;
		}
		x[i].index = -1;

		l++;
		if(l == 1) {
		// Initialize 
			R = 0;
			si = 1;
			w = wresize(w, &n, inst_max_index);
			// w = yx and update |w|
			for(i=0;x[i].index!=-1;i++) 
				w[x[i].index-1] = y * x[i].value;
			wnorm2 = 0;
			for(i=0;i<n;i++) 
				wnorm2 += w[i]*w[i];
		} else  {
			double ywTx = 0, xnorm2 = 0, d;
			w = wresize(w, &n, inst_max_index);
			for(i=0;x[i].index!=-1;i++) {
				xnorm2 += x[i].value * x[i].value;
				ywTx += w[x[i].index-1] * x[i].value;
			}
			ywTx *= y;
			d = sqrt(wnorm2 - 2*ywTx + xnorm2 + si + 1.0/ C);
			if ( d >= R) {
				double eta = 0.5 * (1 - R/d); // 0.5 * (1 - R/d)
				double rdiff = 0.5 *(d - R);
				for(i=0;i<n;i++) 
					w[i] *= (1 - eta);
				for(i=0;x[i].index!=-1;i++) 
					w[x[i].index-1] += eta * y * x[i].value;
				wnorm2 = 0;
				for(i=0;i<n;i++) 
					wnorm2 += w[i]*w[i];

				R = R + rdiff;
				si = si*(1-eta)*(1-eta) + rdiff*rdiff;
			}
		}

		if(l % 10000 == 0) {
			putchar('.');
			fflush(stdout);
		}
	}
	
	puts("");
	printf("iter 1 time %g\n", difftime(time(NULL), start_t));

	// Save model
	{

		struct model *model_ = (struct model*)malloc(sizeof(model));
		int label_[2] = {+1,-1};
		model_->param= param;
		model_->nr_class=2;
		model_->nr_feature=n;
		model_->w=w;
		model_->label = label_;
		model_->bias=-1;
		save_model(model_file_name, model_);
		free(model_);
	}

}


void exit_with_help()
{
	printf(
	"Usage: streamtrain [options] training_set_file  [model_file]\n"
	"options:\n"
	"-c cost : set the parameter C (default 1)\n"
	);
	exit(1);
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = 1; // see setting below
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'c':
				param.C = atof(argv[i]);
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
}

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];

	parse_command_line(argc, argv, input_file_name, model_file_name);
	streamtrain(input_file_name, model_file_name);

	return 0;
}

