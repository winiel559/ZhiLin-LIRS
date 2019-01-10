#include <math.h>
#include <time.h>
#include <algorithm>
#include <set>
#include <assert.h>
#include "zlib/zlib.h"
#include "block.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct problem my_get_block(int sub_id, unsigned long long *offset, int **assign_table, FILE *fp, int *num_feature)
{
	struct problem subprob;
	subprob.l = 200000/20;
	subprob.n = 16609143;
	subprob.bias = -1;
	
	feature_node **node = Malloc(feature_node*,subprob.l);
	int *y = Malloc(int,subprob.l);
	
	subprob.y = y;

	feature_node *node_buffer = Malloc(feature_node,subprob.n);
	
	for(int i=0; i<subprob.l; i++)
	{
		int instance_id = assign_table[sub_id][i];
		
		fseek(fp, offset[instance_id], SEEK_SET);
		
		fread(&subprob.y[i], sizeof(int), 1, fp);
	
		
		feature_node *xi = node_buffer;
		
		for(int j=0; j<num_feature[instance_id]; j++)
		{
			fread(&xi->index, sizeof(int), 1, fp);
			xi++;
		}
		
		xi -= num_feature[instance_id];
	
		for(int j=0; j<num_feature[instance_id]; j++)
		{
			fread(&xi->value, sizeof(double), 1, fp);
			xi++;
		}
		

		node[i] = Malloc(feature_node,num_feature[instance_id]+1);

		for(int j=0; j<num_feature[instance_id]+1; j++)
			node[i][j] = node_buffer[j];
		
		subprob.x = node;

		printf("subprob: %d , instance_id: %d, y: %d\n",sub_id, instance_id, subprob.y[i]);
		for(int j=0; j<10; j++)
		{
			printf("index[%d]=%d , value[%d]=%lf\n",j,subprob.x[i][j].index,j,subprob.x[i][j].value);
		}
		if(instance_id==99)
			for(int j=0; j<num_feature[99]; j++)
				printf("index[%d]=%d , value[%d]=%lf\n",j,subprob.x[i][j].index,j,subprob.x[i][j].value);
	}
	
	return subprob;
}

int main()
{
	int nBlocks = 20;
	int random_assign = 1;
	
	time_t startload_t;
	double init_load_t=0;
	
	startload_t = time(NULL);
	//sleep(3);

    
	struct problem total_prob;	
	total_prob.l = 200000;
	int l = total_prob.l;
	//total_prob.l = 100;
	total_prob.n = 16609143;
	total_prob.bias = -1;


//---------------------get byte_offset-------------------	
	FILE *training_data;
	training_data = fopen("binary_file", "rb");
	
	int num_feature[total_prob.l];
	fread(&num_feature[0], sizeof(int), total_prob.l, training_data);
	
	unsigned long long *byte_offset = Malloc(unsigned long long,total_prob.l+1);
	byte_offset[0] = total_prob.l * 4;
	for(int i=1; i<total_prob.l; i++)
	{
		byte_offset[i] = byte_offset[i-1] + 4 + 12 * num_feature[i-1];
	}
//---------------------get byte_offset-------------------	
	
	//---------set random assign table----
	int **assign_table = (int**) malloc(nBlocks*sizeof(int**));
	for(int i=0;i<nBlocks;i++)
		assign_table[i] = (int*)malloc((l/nBlocks)*sizeof(int*));

	for(int i=0;i<nBlocks;i++)
		for(int j=0;j<l/nBlocks;j++)
			assign_table[i][j] = i*l/nBlocks+j;
				
	//random shuffle
	if(random_assign)
	{
		for(int i=0;i<nBlocks;i++)
		{
			for(int j=0;j<l/nBlocks;j++)
			{
				int k = i+rand()%(nBlocks-i);
				int w = j+rand()%(l/nBlocks-j);

				int temp = assign_table[i][j];
				assign_table[i][j] = assign_table[k][w];
				assign_table[k][w] = temp;
			}
		}
	}
	//---------set random assign table----	
	
	struct problem subprob;
	
	for(int i=0; i<20; i++)
		subprob = my_get_block(i, byte_offset, assign_table, training_data, num_feature);
	
	/*
	printf("sizeof(label): %lu\n", sizeof(subprob.y[0]));
	printf("sizeof(index): %lu\n", sizeof(subprob.x[0][0].index));
	printf("sizeof(value): %lu\n", sizeof(subprob.x[0][0].value));*/

	
	init_load_t += difftime(time(NULL), startload_t);
	//for(int i=0; i<200000; i++)
		//printf("offset[%d] = %llu\n",i,offset[i]);
	printf("initial load time: %f\n",init_load_t);
	
}