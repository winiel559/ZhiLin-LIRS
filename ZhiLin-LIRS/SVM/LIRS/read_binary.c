#include <math.h>
#include <time.h>
#include <algorithm>
#include <set>
#include <assert.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h> 
#include "zlib/zlib.h"
#include "block.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


int main()
{
	int random_assign = 1;
	
	FILE *fp;
	fp = fopen("/media/ssd/workload/SVM/kddb_5G/original/binary_file", "rb");
	
	//meta data
	unsigned int l = 19264097;
	int n = 29890095;
	int nBlocks = 40;
	
	unsigned int *num_feature = Malloc(unsigned, l);
	
	if(fread(&num_feature[0], sizeof(int), l, fp) != l)
	{
		printf("fread error!\n"); exit(1);
	}
	
	unsigned long long *byte_offset = Malloc(unsigned long long,l+1);
	byte_offset[0] = l * 4;
	
	for(unsigned int i=1; i<l; i++)
	{
		byte_offset[i] = byte_offset[i-1] + 4 + 12 * num_feature[i-1];
	}
	
	/* debug message
	for(int i=1; i<10; i++)
		printf("%d\n",num_feature[i]);
	*/
	
	//set random assignment table
	int **assign_table = (int**) malloc(nBlocks*sizeof(int**));
	for(int i=0;i<nBlocks;i++)
		assign_table[i] = (int*)malloc((l/nBlocks)*sizeof(int*));

	for(int i=0;i<nBlocks;i++)
		for(unsigned int j=0;j<l/nBlocks;j++)
			assign_table[i][j] = i*l/nBlocks+j;
	
	//random shuffle
	if(random_assign)
	{
		for(int i=0;i<nBlocks;i++)
		{
			for(unsigned int j=0;j<l/nBlocks;j++)
			{
				int k = i+rand()%(nBlocks-i);
				int w = j+rand()%(l/nBlocks-j);

				int temp = assign_table[i][j];
				assign_table[i][j] = assign_table[k][w];
				assign_table[k][w] = temp;
			}
		}
	}
	
	
	time_t startload_t = time(NULL);
	for(int b=0; b<nBlocks; b++)
	{
		//sync();
		//int fd = open("/proc/sys/vm/drop_caches", O_WRONLY );
		//write(fd, "1", 1);
		//close(fd);
		
		struct problem subprob;
		subprob.l = l/nBlocks;
		subprob.n = n;
		feature_node *node_buffer = Malloc(feature_node,subprob.n);
		int *y = Malloc(int,subprob.l);
		subprob.y = y;
		feature_node **node = Malloc(feature_node*,subprob.l);
		
		
		for(unsigned int i=0; i<l/nBlocks; i++)
		{
			int instance_id = assign_table[b][i];
			
			fseek(fp, byte_offset[instance_id], SEEK_SET);
			
			int y_buf;
			int index_buf[1000];
			double value_buf[1000];
			
			if(fread(&y_buf, sizeof(int), 1, fp) != 1)
			{
				printf("fread error!\n"); exit(1);
			}
			
			if(fread(&index_buf, sizeof(int), num_feature[instance_id], fp) != num_feature[instance_id])
			{
				printf("fread error!\n"); exit(1);
			}
			
			if(fread(&value_buf, sizeof(double), num_feature[instance_id], fp) != num_feature[instance_id])
			{
				printf("fread error!\n"); exit(1);
			}
			
			subprob.y[i] = y_buf;
			feature_node *xi = node_buffer;
			
			for(int j=0; j<num_feature[instance_id]; j++)
			{
				xi->index = index_buf[j];
				xi->value = value_buf[j];
				xi++;
			}
			xi->index = -1;
			
			
			node[i] = Malloc(feature_node,num_feature[instance_id]+1);
			for(int j=0; j<num_feature[instance_id]+1; j++)
				node[i][j] = node_buffer[j];
			
			subprob.x = node;
			
			/* debug message
			if(instance_id%100000==0)
			{
			printf("instance_id=%d\n", instance_id);
			printf("y=%d\n",y);
			for(unsigned int j=0; j<num_feature[instance_id]; j++)
			{
				printf("%d:%f\n", index_buf[j], value_buf[j]);
			}
			printf("\n");
			}
			*/
			
		}
		free(node_buffer);
		
		struct problem *p = &subprob;
		for(int i=0; i<subprob.l; i++)
		{
			free(p->x[i]);
		}
		free(p->x);
		free(p->y);
	}

	printf("Total_load: %.5lf \n", difftime(time(NULL), startload_t));
}