#include <math.h>
#include <time.h>
#include <algorithm>
#include <set>
#include <assert.h>
#include "zlib/zlib.h"
#include "block.h"

const char *data_format_table[] = {
	"SINGLE", "BINARY", "COMPRESSION", NULL
};

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Realloc(ptr, type, n) (type *)realloc((ptr), (n)*sizeof(type))
#define INF HUGE_VAL

using namespace std;

void myfread(void *ptr, size_t size, size_t nmemb, FILE * stream) 
{
	size_t ret = fread(ptr, size, nmemb, stream);
	if(ret != nmemb) {
		fprintf(stderr, "Read Error! Bye Bye %ld %ld\n", ret, size*nmemb);
		exit(-1);
	}
}

#define CHUNKSIZE 1073741824UL
int myuncompress(void *dest, size_t *destlen, const void *source, size_t sourcelen) {
	int ret;
	z_stream strm;

	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;
	strm.avail_in = 0;
	strm.next_in = Z_NULL;
	ret = inflateInit(&strm);
	if (ret != Z_OK) {
		(void)inflateEnd(&strm);
		return ret;
	}

	unsigned char *in = (unsigned char *)source;
	unsigned char *out = (unsigned char *)dest;
	unsigned long bytesread = 0, byteswritten = 0;

	/* decompress until deflate stream ends or end of file */
	do {
		strm.avail_in = (uInt) min(CHUNKSIZE, sourcelen - bytesread);
		//finish all input
		if (strm.avail_in == 0)
			break;
		strm.next_in = in + bytesread;
		bytesread += strm.avail_in;

		/* run inflate() on input until output buffer not full */
		do {
			strm.avail_out = (uInt)CHUNKSIZE;
			strm.next_out = out + byteswritten;
			ret = inflate(&strm, Z_NO_FLUSH);
			assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
			switch (ret) {
				case Z_NEED_DICT:
					ret = Z_DATA_ERROR;     /* and fall through */
				case Z_DATA_ERROR:
				case Z_MEM_ERROR:
					(void)inflateEnd(&strm);
					return ret;
			}
			byteswritten += CHUNKSIZE - strm.avail_out;
		} while (strm.avail_out == 0);

		/* done when inflate() says it's done */
	} while (ret != Z_STREAM_END);

	if(byteswritten != *destlen)
		fprintf(stderr,"buteswritten !- destlen!!!\n");
	*destlen = byteswritten;
	(void)inflateEnd(&strm);
	return 0;

}

// Body of class BinaryProblem
void BinaryProblem::setBias(int idx, double val, int datafmt) {
	if(bias >= 0) 
		fprintf(stderr, "Warning: the bias have been set to %lf\n.", bias);
	bias_idx = idx;
	bias = val;

	// Hack Here
	if(datafmt == SINGLE) {
		retprob = prob;
		if(bias >= 0 && prob.bias != bias) {
			prob.n = retprob.n = bias_idx;
			prob.bias = retprob.bias = bias;
			for(int i = 0; i < l; i++){
				struct feature_node *xi = prob.x[i];
				while(xi->index != -1) 
					xi++;
				xi->index = bias_idx;
				xi->value = bias;
			}
		}
	}
}

void BinaryProblem::load_problem(const char* filename, int datafmt) {
	FILE *fp = fopen(filename, "rb");
	load_header(fp);
	load_body(fp, datafmt);
	parseBinary();
	l = prob.l;
	n = prob.n;
	fclose(fp);
}

struct problem* BinaryProblem::get_problem() {
	retprob = prob;
	if(bias >= 0 && prob.bias != bias) {
		struct feature_node node;
		prob.n = retprob.n = bias_idx;
		prob.bias = retprob.bias = bias;
		node.index = bias_idx;
		node.value = bias;

		for(int i=1;i<retprob.l;i++) 
			*(retprob.x[i]-2) = node; 
		x_space[n_x_space-2] = node;
	} 
	return &retprob;
}

struct problem* BinaryProblem::get_subproblem(int start, int subl){
	get_problem();
	retprob.x = retprob.x + start;
	retprob.y = retprob.y + start;
	retprob.l = subl;
	return &retprob;
}

struct problem* BinaryProblem::get_accproblem(int subl){
	get_problem();
	retprob.x = retprob.x;
	retprob.y = retprob.y;
	retprob.l = subl;
	return &retprob;
}


void BinaryProblem::gen_subproblem(BinaryProblem& ret, vector<int> &mask){
	ret = *this;
	ret.l = (int) mask.size();
	ret.buflen = (sizeof(struct node*) + sizeof(int)) * ret.l;
	ret.buf = Malloc(unsigned char, ret.buflen);
	ret.prob.y = (int*)ret.buf;
	ret.prob.x = (struct feature_node**)(ret.buf + sizeof(int) * ret.l);
	for(int i = 0; i < ret.l; i++) {
		ret.prob.y[i] = prob.y[mask[i]];
		ret.prob.x[i] = prob.x[mask[i]];
	}
}

void BinaryProblem::load_header(FILE *fp) {
	myfread(&prob.l, sizeof(int), 1, fp);
	myfread(&prob.n, sizeof(int), 1, fp);
	myfread(&n_x_space, sizeof(unsigned long), 1, fp);
	myfread(&filelen, sizeof(unsigned long), 1, fp);
	prob.bias = -1;
	buflen = n_x_space * sizeof(struct feature_node) + prob.l * (sizeof(int)+sizeof(unsigned long));
}

void BinaryProblem::load_body(FILE *fp, int datafmt) {
	buf = Realloc(buf, unsigned char, buflen);
	if (buf == NULL) {
		fprintf(stderr,"Memory Error!\n");
	}
	if(datafmt == BINARY) {
		if (buflen != filelen) {
			fprintf(stderr,"l = %d n_x_space = %ld buflen%ld filelen = %ld\n",prob.l, n_x_space, buflen, filelen);
		}
		myfread(buf, sizeof(unsigned char), buflen, fp);
	} else if(datafmt == COMPRESSION) {
		unsigned char *compressedbuf;
		compressedbuf = Malloc(unsigned char, filelen);
		myfread(compressedbuf, sizeof(unsigned char), filelen, fp);
		int retcode = myuncompress(buf, &buflen, compressedbuf, filelen);
		if(retcode != Z_OK) {
			fprintf(stderr, "OK %d MEM %d BUF %d DATA %d g %d %p %ld\n", Z_OK, Z_MEM_ERROR, 
					Z_BUF_ERROR, Z_DATA_ERROR, retcode, buf, buflen);
		}
		free(compressedbuf);
	}
}

void BinaryProblem::parseBinary(){
	unsigned long offset = 0;
	x_space = (struct feature_node*) (buf + offset); 
	offset += sizeof(struct feature_node) * n_x_space;

	prob.y = (int*) (buf + offset); 
	offset += sizeof(int) * prob.l;

	prob.x = (struct feature_node**) (buf + offset); 
	for(int i = 0; i < prob.l; i++) 
		prob.x[i] = x_space + (unsigned long)prob.x[i];
}

// Body of Class BlockProblem
void BlockProblem::setBias(double b){
	bias = b;
	if(bias>=0) n+=1;
	prob_.setBias(n, b, datafmt);
}

void BlockProblem::read_single(const char* singlefile, int nsplits) {
	datafmt = SINGLE;
	nBlocks = nsplits;
	prob_.load_problem(singlefile, COMPRESSION);
	l = prob_.l;
	n = prob_.n;

	int *y = prob_.get_problem()->y;

	feature_node **x = prob_.get_problem()->x;
	vector<vector<int> > blocks(nBlocks);
	set<int> labelset;

	// Simulate the behavior of blockspliter
	// Make sure the random seed is the same of blocksplit.cpp
	srand(1);
	for(int i = 0; i <l; i++) {
		int bid = rand() % nBlocks;
		blocks[bid].push_back(i);
		labelset.insert(y[i]);
	}
	nr_class = (int)labelset.size();
	label.clear();
	if(nr_class == 2) {
		label.push_back(+1);
		label.push_back(-1);
	} else {
		for(set<int>::const_iterator it = labelset.begin(); it != labelset.end(); it++)
			label.push_back(*it);
	}

	start.resize(nBlocks,0);
	subl.resize(nBlocks,0);
	vector<int> newy(y, y+l);
	vector<feature_node*> newx(x, x+l);
	int k = 0;
	for(int i = 0; i < nBlocks; i++) {
		start[i] = k;
		subl[i] = (int) blocks[i].size();
		for(int j = 0; j < subl[i]; j++){
			y[k] = newy[blocks[i][j]];
			x[k] = newx[blocks[i][j]];
			k++;
		}
	}
}

void BlockProblem::read_meta(const char* dirname){
	char filename[1024], fmt[81];
	sprintf(filename,"%s/meta", dirname);
	FILE *meta = fopen(filename, "r");
	fscanf(meta, "%s", fmt);
	for(int i = 0; data_format_table[i]; i++)
		if (strcmp(data_format_table[i], fmt) == 0) 
			datafmt = i;
	if(datafmt == -1) {
		fprintf(stderr, "Unsupported data format\n");
		exit(-1);
	}
	fscanf(meta, "%d %d %d %d", &nBlocks, &l, &n, &nr_class);
	label.resize(nr_class, 0);
	for(int i = 0; i < nr_class; i++)
		fscanf(meta, "%d", &label[i]);

	binary_files.resize(nBlocks,"");
	start.resize(nBlocks,0);
	subl.resize(nBlocks,0);
	for(int i = 0; i < nBlocks; i++){
		fscanf(meta, "%d %d %s", &start[i], &subl[i], filename);
		binary_files[i] = string(dirname) + "/" + string(filename); 
	}
	fclose(meta);
}
struct problem* BlockProblem::get_acc_block(int id){
	int sublt = 0;
	switch(datafmt){
	case SINGLE:
		for(int i=0;i<=id;i++)
			sublt += subl[i];
		return prob_.get_accproblem(sublt);
		break;
	default:
		fprintf(stderr,"Only support single format");
		exit(-1);
	}

}


struct problem* BlockProblem::get_block(int id) {
	if (id >= nBlocks) {
		fprintf(stderr,"Wrong Block Id %d, only %d blocks\n", id, nBlocks);
		exit(-1);
		return NULL;
	}
	switch (datafmt) {
		case SINGLE:
			return prob_.get_subproblem(start[id], subl[id]);
			break;
		case BINARY:
			prob_.load_problem(binary_files[id].c_str(), BINARY);
			return prob_.get_problem();
			break;
		case COMPRESSION:
			prob_.load_problem(binary_files[id].c_str(), COMPRESSION);
			return prob_.get_problem();
			break;
		default:
			fprintf(stderr, "Unsupported data format\n");
			exit(-1);
			break;
	}
}

BlockProblem BlockProblem::genSubProblem(const vector<int>& blocklist) {
	BlockProblem subbprob;
	subbprob.nBlocks = (int)blocklist.size();
	subbprob.n = n;
	subbprob.nr_class = nr_class;
	subbprob.bias = bias;
	subbprob.datafmt = datafmt;
	subbprob.label = label;

	subbprob.l = 0;
	for(int i = 0; i < subbprob.nBlocks; i++) {
		subbprob.start.push_back(subbprob.l);
		subbprob.subl.push_back(subl[blocklist[i]]);
		subbprob.l+= subl[blocklist[i]];
	}

	if(datafmt == SINGLE) {
		vector<int>mask;
		for(int i = 0; i < subbprob.nBlocks; i++) 
			for(int j = 0; j < subl[blocklist[i]]; j++)
				mask.push_back(start[blocklist[i]]+j);
		prob_.gen_subproblem(subbprob.prob_, mask);

	} else {
		for(int i = 0; i < subbprob.nBlocks; i++) 
			subbprob.binary_files.push_back(binary_files[blocklist[i]]);
	}
	return subbprob;
}

class Comp{
	double *dec_val;
	public:
	Comp(double *ptr): dec_val(ptr){}
	bool operator()(int i, int j){
		return dec_val[i] > dec_val[j];
	}
};

// evaluate testing accuracy with w
double evaluate_testing(double *w, int n, const problem *prob_t, int positive_label)
{
	int correct=0;
	feature_node **tx = prob_t->x;
	int *ty = Malloc(int, prob_t->l); 
	double *dec_value = Malloc(double, prob_t->l);
	int *indices = Malloc(int, prob_t->l);
	for(int i =0; i < prob_t->l; i++){
		int idx;
		ty[i] = prob_t->y[i] == positive_label ? 1 : -1;
		dec_value[i]=0;
		indices[i] = i;
		for( const feature_node *lx=tx[i]; (idx=lx->index)!=-1; lx++){
			if(idx<= n)
				dec_value[i] += w[idx-1]*lx->value;
		}

		if (dec_value[i] > 0 && ty[i] == 1)
			++correct;
		else if (dec_value[i] <=0 && ty[i] == -1)
			++correct;
	}
	printf("accuracy %lf ", (double)correct/ prob_t->l);
	sort(indices, indices+prob_t->l, Comp(dec_value));
	int tp = 0,fp = 0;
	double roc = 0;
	for(int i = 0; i < prob_t->l; i++) {
		if(ty[indices[i]] == 1) tp++;
		else if(ty[indices[i]] == -1) {
			roc += tp;
			fp++;
		}
	}
	free(ty);
	free(dec_value);
	free(indices);

	return 1-roc/(tp*fp);
}

double block_testing(double *w, int n,  BlockProblem *bprob){
	int correct=0;
	int *ty = Malloc(int, bprob->l);
	double *dec_value = Malloc(double, bprob->l);
	int *indices = Malloc(int, bprob->l);

	for (int i=0; i < bprob->nBlocks; i++) {
		struct problem *subprob = bprob->get_block(i);
		int start = bprob->start[i], idx;
		for(int j = start; j < start + subprob->l; j++) {
			int lb = subprob->y[j-start];
			if(lb == bprob->label[0]) ty[j] = 1;
			else if (lb == bprob->label[1]) ty[j] = -1;
			else {
				fprintf(stderr,"The label is wrong %d %d %d\n", lb, bprob->label[0], bprob->label[1]);
				exit(-1);
			}
			indices[j] = j;
			dec_value[j]=0;

			for( const feature_node *lx=subprob->x[j-start]; (idx=lx->index)!=-1; lx++){
				if(idx<= n)
					dec_value[j] += w[idx-1]*lx->value;
			}
			if (dec_value[j] > 0 && ty[j] == 1)
				++correct;
			else if (dec_value[j] <= 0 && ty[j] == -1)
				++correct;
		}
	}

	sort(indices, indices+bprob->l, Comp(dec_value));
	int tp = 0,fp = 0;
	double roc = 0;
	for(int i = 0; i < bprob->l; i++) {
		if(ty[indices[i]] == 1) tp++;
		else if(ty[indices[i]] == -1) {
			roc += tp;
			fp++;
		}
	}
	free(dec_value);
	free(indices);
	free(ty);
	//return 1-roc/(tp*fp);
	return (double)correct;
	//return (double)correct/(double)prob_t->l;
}

double primal_value(BlockProblem *bprob, int solver_type, int *y, double C, double *w) {
	int nBlocks = bprob->nBlocks;
	int n = bprob->n;
	double v = 0;

	for(int i = 0; i < n; i++) 
		v += w[i]*w[i];
	v *= 0.5;
	for(int i = 0; i < nBlocks; i++) {
		struct problem *subprob = bprob->get_block(i);
		int start = bprob->start[i];
		for(int j = 0; j < subprob->l; j++) {
			double ywTx = 0;
			feature_node *xj = subprob->x[j];
			while(xj->index!= -1)
			{
				ywTx += w[xj->index-1]*(xj->value);
				xj++;
			}
			ywTx *= y[start+j];
			if(ywTx < 1) {
				if(solver_type == L2R_L1LOSS_SVC_PEGASOS || solver_type == L2R_L1LOSS_SVC_DUAL || solver_type == L2R_L1LOSS_SVC_PEGASOS_INNER)
					v += C * (1-ywTx);
				else if (solver_type == L2R_L2LOSS_SVC_DUAL) 
					v += C * (1-ywTx) * (1- ywTx);
			}
		}
	}
	return v;
}

void block_pegasos_as_inner(BlockProblem *bprob, const parameter *param, double *w, double Cp, double Cn) {
	int nBlocks = bprob->nBlocks;
	int l = bprob->l;
	int n = bprob->n;
	int solver_type = param->solver_type;
	int is_perm = 1; //param->is_perm;
	int max_iter=param->max_iter;
	if(solver_type != L2R_L1LOSS_SVC_PEGASOS_INNER) {
		fprintf(stderr, "Error: unknown solver_type or unsupported solver_type\n");
		return;
	}
	int *y = new int[l];
	double wnorm = 0.0; // w squared norm
	double a = 1; // coeff
	memset(w, 0, sizeof(double) * n);

	int iter = 0;
	int *perm = new int[nBlocks];
	for(int i = 0; i < nBlocks; i++)
		perm[i] = i;

	time_t start_t, startload_t;
	clock_t startcpu;
	double total_time = 0, total_load = 0, total_cpu = 0;
	double lambda = 1.0/ (Cp * l);
	int t = 0; // number of updates in pegasos
	printf("lambda %g\n", lambda);
	while (iter < max_iter) {
		startcpu = clock();
		start_t = time(NULL);
		startload_t = start_t;
		iter++;

		if(is_perm) 
		for (int i=0; i<nBlocks; i++) {
	//		perm[i] = rand() % nBlocks;
			int j = i+rand()%(nBlocks-i);
			swap(perm[i], perm[j]);

		}
		total_cpu += (double) (clock() - startcpu) / CLOCKS_PER_SEC;
		for (int i=0; i < nBlocks; i++) {
			startload_t = time(NULL);
			struct problem *subprob = bprob->get_block(perm[i]);
			total_load += difftime(time(NULL), startload_t);
			startcpu = clock();
			int start = bprob->start[perm[i]];
			if(iter == 1) {
				for(int j = start; j < start + subprob->l; j++) {
					int lb = subprob->y[j-start];
					if(lb == bprob->label[0]) y[j] = 1;
					else if (lb == bprob->label[1]) y[j] = -1;
					else {
						printf("id=%d, start = %d, l = %d datafmt = %d\n", i, start, subprob->l, bprob->datafmt);
						fprintf(stderr,"The label is wrong %d %d %d\n", lb, bprob->label[0], bprob->label[1]);
						exit(-1);
					}
				}
			}
			subprob->y = y+start;

			pegasos_solve_subprob(subprob, n, &a, w, &wnorm, lambda, t);
			t += subprob->l;
			total_cpu += (double) (clock() - startcpu) / CLOCKS_PER_SEC;
		}

		for(int i = 0; i < n; i++) w[i] *= a;
		a = 1.0;

		total_time += difftime(time(NULL), start_t);

		printf("iter %d time %.5lf runtime %.5lf loadtime %.5lf cputime %.5lf  ", 
				iter, total_time, total_time-total_load, total_load, total_cpu );
		if(param->showprimal) {
			double primal = primal_value(bprob, solver_type, y, Cp, w);
			printf("obj %.5lf primal %.5lf ", primal, primal);
		}

		if(param->prob_t != NULL) 
			printf("aoc %lf ", 100*evaluate_testing(w, n, param->prob_t,bprob->label[0]));
		printf("\n");
		fflush(stdout);

	}

	delete [] y;
	delete [] perm;
}


void block_pegasos(BlockProblem *bprob, const parameter *param, double *w, double Cp, double Cn) {
	int nBlocks = bprob->nBlocks;
	int l = bprob->l;
	int n = bprob->n;
	int solver_type = param->solver_type;
	int is_perm = 1; //param->is_perm;
	int max_iter=param->max_iter;
	if(solver_type != L2R_L1LOSS_SVC_PEGASOS) {
		fprintf(stderr, "Error: unknown solver_type or unsupported solver_type\n");
		return;
	}
	int *y = new int[l];
	memset(w, 0, sizeof(double) * n);

	int iter = 0;
	int *perm = new int[nBlocks];
	for(int i = 0; i < nBlocks; i++)
		perm[i] = i;

	time_t start_t, startload_t;
	clock_t startcpu;
	double total_time = 0, total_load = 0, total_cpu = 0;
	double lambda = 1.0/ (Cp * l);
	printf("lambda %g\n", lambda);
	while (iter < max_iter) {
		startcpu = clock();
		start_t = time(NULL);
		startload_t = start_t;
		iter++;

		if(is_perm) 
		for (int i=0; i<nBlocks; i++) {
			int j = i+rand()%(nBlocks-i);
			swap(perm[i], perm[j]);
			//perm[i] = rand() % nBlocks;
		}
		total_cpu += (double) (clock() - startcpu) / CLOCKS_PER_SEC;
		for (int i=0; i < nBlocks; i++) {
			startload_t = time(NULL);
			struct problem *subprob = bprob->get_block(perm[i]);
			total_load += difftime(time(NULL), startload_t);
			startcpu = clock();
			int start = bprob->start[perm[i]];
			if(iter == 1) {
				for(int j = start; j < start + subprob->l; j++) {
					int lb = subprob->y[j-start];
					if(lb == bprob->label[0]) y[j] = 1;
					else if (lb == bprob->label[1]) y[j] = -1;
					else {
						printf("id=%d, start = %d, l = %d datafmt = %d\n", i, start, subprob->l, bprob->datafmt);
						fprintf(stderr,"The label is wrong %d %d %d\n", lb, bprob->label[0], bprob->label[1]);
						exit(-1);
					}
				}
			}
			subprob->y = y+start;

			double eta = 1/(lambda * ((iter-1)*nBlocks+i+2));
			pegasos_update_subgradient(subprob, n, w, lambda, eta);
			total_cpu += (double) (clock() - startcpu) / CLOCKS_PER_SEC;
		}

		total_time += difftime(time(NULL), start_t);

		printf("iter %d time %.5lf runtime %.5lf loadtime %.5lf cputime %.5lf  ", 
				iter, total_time, total_time-total_load, total_load, total_cpu );
		if(param->showprimal) {
			double primal = primal_value(bprob, solver_type, y, Cp, w);
			printf("obj %.5lf primal %.5lf ", primal, primal);
		}

		if(param->prob_t != NULL) 
			printf("aoc %lf ", 100*evaluate_testing(w, n, param->prob_t,bprob->label[0]));
		printf("\n");
		fflush(stdout);

	}

	delete [] y;
	delete [] perm;
}

void block_solve_linear_c_svc(BlockProblem *bprob, const parameter *param, double *w, double Cp, double Cn) {
	int nBlocks = bprob->nBlocks;
	int l = bprob->l;
	int n = bprob->n;
	int solver_type = param->solver_type;
	int max_iter=param->max_iter;
	int inner_max_iter=param->inner_max_iter;
	int is_perm =param->is_perm;
	double eps=param->eps;
	double inner_eps = param->inner_eps;
	double PGmax=-INF, PGmin=INF;
	if(solver_type != L2R_L2LOSS_SVC_DUAL && solver_type != L2R_L1LOSS_SVC_DUAL) {
		fprintf(stderr, "Error: unknown solver_type or unsupported solver_type\n");
		return;
	}

	int *y = new int[l];
	double *alpha = new double[l];
	memset(alpha, 0, sizeof(double) * l);
	memset(w, 0, sizeof(double) * n);
	int iter = 0;
	int *perm = new int[nBlocks];
	for(int i = 0; i < nBlocks; i++)
		perm[i] = i;

	time_t start_t, startload_t;
	clock_t startcpu;
	double total_time = 0, total_load = 0, total_cpu = 0;
	while (iter < max_iter) {
		startcpu = clock();
		start_t = time(NULL);
		startload_t = start_t;
		PGmax = -INF;
		PGmin = INF;
		iter++;
		//	inner_eps /= (iter + 10);
		if(is_perm) 
		for (int i=0; i<nBlocks; i++) {
			int j = i+rand()%(nBlocks-i);
			swap(perm[i], perm[j]);
		}
		bool solved = true;

		total_cpu += (double) (clock() - startcpu) / CLOCKS_PER_SEC;
		for (int i=0; i < nBlocks; i++) {
			startload_t = time(NULL);
			struct problem *subprob = bprob->get_block(perm[i]);
			total_load += difftime(time(NULL), startload_t);
			startcpu = clock();
			int start = bprob->start[perm[i]];
			if(iter == 1) {
				for(int j = start; j < start + subprob->l; j++) {
					int lb = subprob->y[j-start];
					if(lb == bprob->label[0]) y[j] = 1;
					else if (lb == bprob->label[1]) y[j] = -1;
					else {
						printf("id=%d, start = %d, l = %d datafmt = %d\n", i, start, subprob->l, bprob->datafmt);
						fprintf(stderr,"The label is wrong %d %d %d\n", lb, bprob->label[0], bprob->label[1]);
						exit(-1);
					}
				}
			}
			subprob->y = y+start;

			//			fprintf(stderr,"%-3d", perm[i]);
			//			if(i%20==19) fprintf(stderr,"\n");
			double PGmax_ , PGmin_ ;
			solve_l2r_l1l2_svc(subprob,
					w, alpha+start, inner_eps, Cp, Cn, solver_type,
					&PGmax_, &PGmin_, inner_max_iter, &solved);
			PGmax = max(PGmax, PGmax_);
			PGmin = min(PGmin, PGmin_);
			total_cpu += (double) (clock() - startcpu) / CLOCKS_PER_SEC;
		}

		total_time += difftime(time(NULL), start_t);
		double v = 0;
		// calculate dual obj
		double diag_p = 0.5/Cp, diag_n = 0.5/Cn;
		if(solver_type == L2R_L1LOSS_SVC_DUAL)
			diag_p = diag_n = 0;

		int nSV = 0;
		for(int i=0; i<n; i++)
			v += w[i]*w[i];
		for(int i=0; i<l; i++)
		{
			if (y[i] == 1)
				v += alpha[i]*(alpha[i]*diag_p - 2); 
			else
				v += alpha[i]*(alpha[i]*diag_n - 2);
			if(alpha[i] > 0)
				++nSV;
		}

		printf("iter %d time %.5lf runtime %.5lf loadtime %.5lf cputime %.5lf obj %lf  PGmax %.5lf PGmin %.5lf gap %.5lf ", 
				iter, total_time, total_time-total_load, total_load, total_cpu,  v/2,  PGmax, PGmin, PGmax - PGmin);
		if(param->showprimal) {
			double primal = primal_value(bprob, solver_type, y, Cp, w);
			printf("primal %.5lf ", primal);
		}
		if(param->prob_t != NULL) 
			printf("aoc %lf ", 100*evaluate_testing(w, n, param->prob_t, bprob->label[0]));
		printf("\n");
		fflush(stdout);
		if(solved && PGmax - PGmin < eps)
			break;

	}

	delete [] y;
	delete [] perm;
	delete [] alpha;
}

void blocktrain_MCSVM_CS(BlockProblem *bprob, const parameter *param, double *w, int nr_class, double *weighted_C) {
	int nBlocks = bprob->nBlocks;
	int l = bprob->l;
	int n = bprob->n;
	vector<int> &label = bprob->label;

	int solver_type = param->solver_type;
	int max_iter = param->max_iter;
	int inner_max_iter = param->inner_max_iter;
	double eps=param->eps;
	double inner_eps = param->inner_eps;
	double stopping = -INF;
	if(solver_type != MCSVM_CS) {
		fprintf(stderr, "Error: unknown solver_type or unsupported solver_type\n");
		return;
	}
	int *y = new int[l];
	double *alpha = new double[l*nr_class];
	memset(alpha, 0, sizeof(double)*l*nr_class);
	memset(w, 0, sizeof(double)*n*nr_class);
	int iter = 0;
	int *perm = new int[nBlocks];
	for(int i = 0; i < nBlocks; i++)
		perm[i] = i;
	time_t start_t;
	double total_time = 0;
	while (iter < max_iter)	{
		start_t = time(NULL);
		stopping = -INF;
		iter++;
		for (int i=0; i<nBlocks; i++) {
			int j = i+rand()%(nBlocks-i);
			swap(perm[i], perm[j]);
		}
		bool solved = true;

		for (int i=0; i < nBlocks; i++) {
			struct problem *subprob = bprob->get_block(perm[i]);
			int start = bprob->start[perm[i]];
			if(iter == 1) {
				for(int j = start; j < start + subprob->l; j++) {
					int lb =  subprob->y[j - start], k;
					for(k = 0; k < nr_class; k++)
						if(lb == label[k]) {
							y[j] = k;
							break;
						}
					if(k == nr_class) {
						fprintf(stderr,"Wrong label\n");
						exit(-1);
					}
				}
			}

			subprob->y = y + start;

			//			fprintf(stderr,"%-3d", perm[i]);
			//			if(i%20==19) fprintf(stderr,"\n");
			double stopping_;

			Solver_MCSVM_CS Solver(subprob, nr_class, weighted_C, inner_eps, inner_max_iter);
			Solver.Solve(w, alpha+start*nr_class, &stopping_, &solved);
			stopping = max(stopping, stopping_);
		}

		total_time += difftime(time(NULL), start_t);

		// calculate object value
		double v = 0;
		int nSV = 0;
		for(int i=0;i<n*nr_class;i++)
			v += w[i]*w[i];
		v = 0.5*v;
		for(int i=0;i<l*nr_class;i++)
		{
			v += alpha[i];
			if(fabs(alpha[i]) > 0)
				nSV++;
		}
		for(int i=0;i<l;i++)
			v -= alpha[i*nr_class+y[i]];

		printf("iter %d time %.5lf obj %lf stopping %.5lf ", 
				iter, total_time, v, stopping);
		if(param->prob_t != NULL)
			printf("accuracy %lf ", evaluate_testing(w, n, param->prob_t, bprob->label[0]));
		printf("\n");
		fflush(stdout);

		if(solved && stopping < eps)
			break;
	}

	delete [] y;
	delete [] perm;
	delete [] alpha;
}

void block_subsample_solver(BlockProblem *bprob, const parameter *param, double *w, double Cp, double Cn) {
	int nBlocks = bprob->nBlocks;
	int l = bprob->l;
	int n = bprob->n;
	int solver_type = param->solver_type;
	int max_iter=param->max_iter;
	int inner_max_iter=param->inner_max_iter;
	int is_perm =param->is_perm;
	double eps=param->eps;
	double inner_eps = param->inner_eps;
	double PGmax=-INF, PGmin=INF;
	if(solver_type != L2R_L1LOSS_SVC_DUAL_SUBSAMPLE) {
		fprintf(stderr, "Error: unknown solver_type or unsupported solver_type\n");
		return;
	}

	int *y = new int[l];
	int *perm = new int[nBlocks];
	double *alpha = new double[l];
	double *w_tmp = new double[n]; // for temporarily storing w from each block
	memset(alpha, 0, sizeof(double) * l);
	memset(w, 0, sizeof(double) * n);


	int iter = 0;
	time_t start_t, startload_t;
	clock_t startcpu;
	double total_time = 0, total_load = 0, total_cpu = 0;
//	while (iter < max_iter) 
	{
		startcpu = clock();
		start_t = time(NULL);
		startload_t = start_t;
		PGmax = -INF;
		PGmin = INF;
		iter++;
		//	inner_eps /= (iter + 10);
		for(int i=0; i<nBlocks; i++) 
			perm[i] = i;
	/*	if(is_perm) 
		for (int i=0; i<nBlocks; i++) {
			int j = i+rand()%(nBlocks-i);
			swap(perm[i], perm[j]);
		}*/
		bool solved = true;
		memset(w, 0, sizeof(double)*n);

		total_cpu += (double) (clock() - startcpu) / CLOCKS_PER_SEC;
		for (int i=0; i < nBlocks; i++) {
			startload_t = time(NULL);
			struct problem *subprob = bprob->get_acc_block(perm[i]);
			total_load += difftime(time(NULL), startload_t);
			startcpu = clock();
			int start = 0;
			if(iter == 1) {
				for(int j = start; j < start + subprob->l; j++) {
					int lb = subprob->y[j-start];
					if(lb == bprob->label[0]) y[j] = 1;
					else if (lb == bprob->label[1]) y[j] = -1;
					else {
						printf("id=%d, start = %d, l = %d datafmt = %d\n", i, start, subprob->l, bprob->datafmt);
						fprintf(stderr,"The label is wrong %d %d %d\n", lb, bprob->label[0], bprob->label[1]);
						exit(-1);
					}
				}
			}
			subprob->y = y;

			double PGmax_ , PGmin_ ;
			memset(w_tmp, 0, sizeof(double) * n);
			solve_l2r_l1l2_svc(subprob,
					w_tmp, alpha+start, inner_eps, Cp, Cn, solver_type,
					&PGmax_, &PGmin_, inner_max_iter, &solved);
			if(param->prob_t != NULL) {
				printf("block %d ", i);
				printf("aoc %lf ", 100*evaluate_testing(w_tmp, n, param->prob_t, bprob->label[0]));
				printf("\n");
			}
			PGmax = max(PGmax, PGmax_);
			PGmin = min(PGmin, PGmin_);
			for(int j=0; j < n; j++)
				w[j] += w_tmp[j];
			total_cpu += (double) (clock() - startcpu) / CLOCKS_PER_SEC;
		}

		// No scale => no influence on accuracy
		// for(int i=0; i<n; i++) w[i] /= nBlocks;

		total_time += difftime(time(NULL), start_t);
		double v = 0;
		// calculate dual obj
		double diag_p = 0.5/Cp, diag_n = 0.5/Cn;
		if(solver_type == L2R_L1LOSS_SVC_DUAL_SUBSAMPLE)
			diag_p = diag_n = 0;

		int nSV = 0;
		for(int i=0; i<n; i++)
			v += w[i]*w[i];
		for(int i=0; i<l; i++)
		{
			if (y[i] == 1)
				v += alpha[i]*(alpha[i]*diag_p - 2); 
			else
				v += alpha[i]*(alpha[i]*diag_n - 2);
			if(alpha[i] > 0)
				++nSV;
		}

		printf("iter %d time %.5lf runtime %.5lf loadtime %.5lf cputime %.5lf obj %lf  PGmax %.5lf PGmin %.5lf gap %.5lf ", 
				iter, total_time, total_time-total_load, total_load, total_cpu,  v/2,  PGmax, PGmin, PGmax - PGmin);
		if(param->showprimal) {
			double primal = primal_value(bprob, solver_type, y, Cp, w);
			printf("primal %.5lf ", primal);
		}
		if(param->prob_t != NULL) 
			printf("aoc %lf ", 100*evaluate_testing(w, n, param->prob_t, bprob->label[0]));
		printf("\n");
		fflush(stdout);

	}

	delete [] y;
	delete [] perm;
	delete [] alpha;
}

void block_average_solver(BlockProblem *bprob, const parameter *param, double *w, double Cp, double Cn) {
	int nBlocks = bprob->nBlocks;
	int l = bprob->l;
	int n = bprob->n;
	int solver_type = param->solver_type;
	int max_iter=param->max_iter;
	int inner_max_iter=param->inner_max_iter;
	int is_perm =param->is_perm;
	double eps=param->eps;
	double inner_eps = param->inner_eps;
	double PGmax=-INF, PGmin=INF;
	if(solver_type != L2R_L2LOSS_SVC_DUAL_AVG && solver_type != L2R_L1LOSS_SVC_DUAL_AVG) {
		fprintf(stderr, "Error: unknown solver_type or unsupported solver_type\n");
		return;
	}

	int *y = new int[l];
	int *perm = new int[nBlocks];
	double *alpha = new double[l];
	double *w_tmp = new double[n]; // for temporarily storing w from each block
	memset(alpha, 0, sizeof(double) * l);
	memset(w, 0, sizeof(double) * n);


	int iter = 0;
	time_t start_t, startload_t;
	clock_t startcpu;
	double total_time = 0, total_load = 0, total_cpu = 0;
//	while (iter < max_iter) 
	{
		startcpu = clock();
		start_t = time(NULL);
		startload_t = start_t;
		PGmax = -INF;
		PGmin = INF;
		iter++;
		//	inner_eps /= (iter + 10);
		for(int i=0; i<nBlocks; i++) 
			perm[i] = i;
		if(is_perm) 
		for (int i=0; i<nBlocks; i++) {
			int j = i+rand()%(nBlocks-i);
			swap(perm[i], perm[j]);
		}
		bool solved = true;
		memset(w, 0, sizeof(double)*n);

		total_cpu += (double) (clock() - startcpu) / CLOCKS_PER_SEC;
		for (int i=0; i < nBlocks; i++) {
			startload_t = time(NULL);
			struct problem *subprob = bprob->get_block(perm[i]);
			total_load += difftime(time(NULL), startload_t);
			startcpu = clock();
			int start = bprob->start[perm[i]];
			if(iter == 1) {
				for(int j = start; j < start + subprob->l; j++) {
					int lb = subprob->y[j-start];
					if(lb == bprob->label[0]) y[j] = 1;
					else if (lb == bprob->label[1]) y[j] = -1;
					else {
						printf("id=%d, start = %d, l = %d datafmt = %d\n", i, start, subprob->l, bprob->datafmt);
						fprintf(stderr,"The label is wrong %d %d %d\n", lb, bprob->label[0], bprob->label[1]);
						exit(-1);
					}
				}
			}
			subprob->y = y+start;

			double PGmax_ , PGmin_ ;
			memset(w_tmp, 0, sizeof(double) * n);
			solve_l2r_l1l2_svc(subprob,
					w_tmp, alpha+start, inner_eps, Cp, Cn, solver_type,
					&PGmax_, &PGmin_, inner_max_iter, &solved);
			if(param->prob_t != NULL) {
				printf("block %d ", i);
				printf("aoc %lf ", 100*evaluate_testing(w_tmp, n, param->prob_t, bprob->label[0]));
				printf("\n");
			}
			PGmax = max(PGmax, PGmax_);
			PGmin = min(PGmin, PGmin_);
			for(int j=0; j < n; j++)
				w[j] += w_tmp[j];
			total_cpu += (double) (clock() - startcpu) / CLOCKS_PER_SEC;
		}

		// No scale => no influence on accuracy
		// for(int i=0; i<n; i++) w[i] /= nBlocks;

		total_time += difftime(time(NULL), start_t);
		double v = 0;
		// calculate dual obj
		double diag_p = 0.5/Cp, diag_n = 0.5/Cn;
		if(solver_type == L2R_L1LOSS_SVC_DUAL_AVG)
			diag_p = diag_n = 0;

		int nSV = 0;
		for(int i=0; i<n; i++)
			v += w[i]*w[i];
		for(int i=0; i<l; i++)
		{
			if (y[i] == 1)
				v += alpha[i]*(alpha[i]*diag_p - 2); 
			else
				v += alpha[i]*(alpha[i]*diag_n - 2);
			if(alpha[i] > 0)
				++nSV;
		}

		printf("iter %d time %.5lf runtime %.5lf loadtime %.5lf cputime %.5lf obj %lf  PGmax %.5lf PGmin %.5lf gap %.5lf ", 
				iter, total_time, total_time-total_load, total_load, total_cpu,  v/2,  PGmax, PGmin, PGmax - PGmin);
		if(param->showprimal) {
			double primal = primal_value(bprob, solver_type, y, Cp, w);
			printf("primal %.5lf ", primal);
		}
		if(param->prob_t != NULL) 
			printf("aoc %lf ", 100*evaluate_testing(w, n, param->prob_t, bprob->label[0]));
		printf("\n");
		fflush(stdout);

	}

	delete [] y;
	delete [] perm;
	delete [] alpha;
}

void blocktrain_one(BlockProblem *bprob, const parameter *param, double *w, double Cp, double Cn) {
	switch (param->solver_type)
	{
		case L2R_L1LOSS_SVC_DUAL:
			block_solve_linear_c_svc(bprob, param, w, Cp, Cn);
			break;
		case L2R_L2LOSS_SVC_DUAL: 
			block_solve_linear_c_svc(bprob, param, w, Cp, Cn);
			break;
		case L2R_L1LOSS_SVC_PEGASOS:
			block_pegasos(bprob, param, w, Cp, Cn);
			break;
		case L2R_L1LOSS_SVC_PEGASOS_INNER:
			block_pegasos_as_inner(bprob, param, w, Cp, Cn);
			break;
		case L2R_L2LOSS_SVC_DUAL_AVG:
			block_average_solver(bprob, param, w, Cp, Cn);
			break;
		case L2R_L1LOSS_SVC_DUAL_AVG:
			block_average_solver(bprob, param, w, Cp, Cn);
			break;
		case L2R_L1LOSS_SVC_DUAL_SUBSAMPLE:
			block_subsample_solver(bprob, param, w, Cp, Cn);
		default:
			fprintf(stderr,"Not support for block version!\n");
	}

}

struct model* blocktrain(BlockProblem* bprob, const  parameter* param) {
	//int l = bprob->l;
	int n = bprob->n;
	model *model_ = Malloc(model,1);

	if(bprob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = bprob->bias;

	int nr_class = bprob->nr_class;
	vector<int> &label = bprob->label;

	model_->nr_class = nr_class;
	model_->label = Malloc(int, nr_class);
	for(int i=0;i<nr_class;i++)
		model_->label[i] = label[i];

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);

	for(int i=0;i<nr_class;i++)
		weighted_C[i] = param->C;
	for(int i=0;i<param->nr_weight;i++) {
		int j;
		for(j=0;j<nr_class;j++)
			if(param->weight_label[i] == label[j])
				break;
		if(j == nr_class)
			fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
		else
			weighted_C[j] *= param->weight[i];
	}

	srand(1);

	if(param->solver_type == MCSVM_CS)
	{
		model_->w=Malloc(double, n*nr_class);
		blocktrain_MCSVM_CS(bprob, param, model_->w, nr_class, weighted_C);
	} else {
		if(nr_class == 2) {
			model_->w=Malloc(double, n);
			blocktrain_one(bprob, param, model_->w, param->C, param->C);
		} else {
			fprintf(stderr, "Try -s 4 for Multiclass-SVM\n");
			exit(-1);
		}
	}

	free(weighted_C);
	return model_;
}

double block_cross_validation(BlockProblem *bprob, const parameter *param, int nr_fold)
{
	int *fold_start = Malloc(int,nr_fold+1);
	int nBlocks = bprob->nBlocks;
	int *perm = Malloc(int,nBlocks);
	double acc = 0.0;

	for(int i=0;i<nBlocks;i++) perm[i]=i;
	for(int i=0;i<nBlocks;i++) {
		int j = i+rand()%(nBlocks-i);
		swap(perm[i],perm[j]);
	}

	for(int i=0;i<=nr_fold;i++)
		fold_start[i]=i*nBlocks/nr_fold;

	for(int i=0;i<nr_fold;i++)
	{
		vector<int>subblock,valblock;
		int begin = fold_start[i];
		int end = fold_start[i+1];

		for(int j=0;j<begin;j++)
			subblock.push_back(perm[j]);
		for(int j=begin; j < end; j++)
			valblock.push_back(perm[j]);
		for(int j=end;j<nBlocks;j++)
			subblock.push_back(perm[j]);

		BlockProblem subprob = bprob->genSubProblem(subblock);
		BlockProblem valprob = bprob->genSubProblem(valblock);


		struct model *submodel = blocktrain(&subprob, param);
		acc += block_testing(submodel->w, bprob->n, &valprob);
		free_and_destroy_model(&submodel);
	}
	free(fold_start);
	free(perm);
	
	return acc / bprob->l;
}

