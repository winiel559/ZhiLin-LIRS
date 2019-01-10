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


struct problem my_get_block(int l, int nBlocks, int n, int sub_id, unsigned long long *offset, int **assign_table, FILE *fp, int *num_feature)
{
	struct problem subprob;
	subprob.l = l/nBlocks;
	subprob.n = n;
	subprob.bias = -1;
	
	feature_node **node = Malloc(feature_node*,subprob.l);
	int *y = Malloc(int,subprob.l);
	
	subprob.y = y;

	feature_node *node_buffer = Malloc(feature_node,subprob.n);
	
	//setvbuf( fp , NULL , _IONBF , 0 );
	//setbuf ( fp , NULL );
	//setvbuf( fp , NULL , _IOFBF , 404200 );
	int index_buf[1000000];
	double value_buf[1000000];
	
	time_t startload_t = time(NULL);
	time_t start_test = time(NULL);
	double total_load = 0;
	int temp = 0;
	for(int i=0; i<subprob.l; i++)
	{
		int instance_id = assign_table[sub_id][i];
		//printf("instance_id:%d\n", instance_id);
		//int instance_id = sub_id*19264097/40+(rand() % (19264097/40));
		startload_t = time(NULL);
		fseek(fp, offset[instance_id], SEEK_SET);
		total_load += difftime(time(NULL), startload_t);
		
		fread(&subprob.y[i], sizeof(int), 1, fp);
	
		
		feature_node *xi = node_buffer;
		
		//fread(&index_buf, sizeof(int), num_feature[instance_id], fp);
		//fread(&value_buf, sizeof(double), num_feature[instance_id], fp);
		
		
		
		for(int j=0; j<num_feature[instance_id]; j++)
		{
			fread(&xi->index, sizeof(int), 1, fp);
			xi++;
			temp++;
		}
		xi->index = -1;
		
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
		
		
		/*
		printf("subprob: %d , instance_id: %d, y: %d\n",sub_id, instance_id, subprob.y[i]);
		
		for(int j=0; j<num_feature[instance_id]; j++)
		{
			printf("index[%d]=%d , value[%d]=%lf\n",j,subprob.x[i][j].index,j,subprob.x[i][j].value);
		}
		
		if(instance_id==99)
			for(int j=0; j<num_feature[99]; j++)
				printf("index[%d]=%d , value[%d]=%lf\n",j,subprob.x[i][j].index,j,subprob.x[i][j].value);
		*/
		
	}
	//printf("\n\nTotal %.5lf test:%.5lf\n\n", difftime(time(NULL), start_test), total_load);
	//printf("BUFSIZ: %d \n", BUFSIZ);

	free(node_buffer);
	return subprob;
}

int main()
{
	//kdd
	
	int l = 19264097;
	int n = 29890095;
	int nBlocks = 40;
	
	//web
	/*
	int l = 200000;
	int n = 16609143;
	int nBlocks = 40;*/
	
	//---------------------get byte_offset-------------------	
	FILE *training_data;
	char training_binary_dir[1024];
	
	strcpy(training_binary_dir, "/media/ssd/workload/SVM/kddb_5G/original/binary_file");
	//strcpy(training_binary_dir, "/media/3DXpoint/workload/SVM/kddb_5G/original/binary_file");
	//strcpy(training_binary_dir, "/media/disk/workload/SVM/kddb_5G/original/binary_file");
	//strcpy(training_binary_dir, "/media/ssd/workload/SVM/webspam_14G/binary_file");
	printf("training_binary_dir: %s\n", training_binary_dir);
	training_data = fopen(training_binary_dir, "rb");
	
	
	int *num_feature = Malloc(int, l);
	fread(&num_feature[0], sizeof(int), l, training_data);
	
	unsigned long long *byte_offset = Malloc(unsigned long long,l+1);
	byte_offset[0] = l * 4;
	for(int i=1; i<l; i++)
	{
		byte_offset[i] = byte_offset[i-1] + 4 + 12 * num_feature[i-1];
		//if(i%10000==0)
			//printf("byte_offset[i]: %llu\n", i, byte_offset[i]);
	}
	printf("byte_offset[%d]: %llu\n", 0, byte_offset[0]);
	printf("byte_offset[%d]: %llu\n", l-1, byte_offset[l-1]);
	//free(num_feature);
	
	//---------set random assign table----
	int **assign_table = (int**) malloc(nBlocks*sizeof(int**));
	for(int i=0;i<nBlocks;i++)
			assign_table[i] = (int*)malloc((l/nBlocks)*sizeof(int*));

	for(int i=0;i<nBlocks;i++)
			for(int j=0;j<l/nBlocks;j++)
					assign_table[i][j] = i*l/nBlocks+j;
				
	if (((byte_offset[l-1] - byte_offset[0]) / l) < 4096)
	//if (4097 < 4096)
	{
		printf("Page assignment.\n");
		int *blk_size = Malloc(int, nBlocks); //The instance number of each block
		for(int i=0; i<nBlocks; i++)
			blk_size[i] = 0;
		int page_id = byte_offset[0]/4096; //start page
		if(0)
		{
			int select_blk = rand()%40; //select a block
			for(int i=0; i<l; i++)
			{
				if( (byte_offset[i]/4096) > page_id) //next page
				{
					select_blk = rand()%40;
					page_id++;
				}
				if(blk_size[select_blk] >= (l/nBlocks))
				{
					select_blk = rand()%40;
				}
				else
				{
					assign_table[select_blk][blk_size[select_blk]] = i;
					blk_size[select_blk]++;
					//printf("instance %d(Page_id %d) assign to block %d \n", i, page_id, select_blk);

				}
			}
			//printf("Done.\n");
		}
		
	}
	
	//Average instance size > page size (for webspam and epsilon)
	else
	{
		printf("Instance assignment.\n");
		if(1)
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
		
	}
	

	time_t startload_t;
	double total_load = 0;
	double total_load_2 = 0;

	for (int i=0; i < nBlocks; i++) {
	//for (int i=0; i < 3; i++) {
		
		startload_t = time(NULL);

		struct problem subprob = my_get_block(l, nBlocks, n, i, byte_offset, assign_table, training_data, num_feature);
		total_load += difftime(time(NULL), startload_t);


		printf("Total load: %f\n", total_load);

		
		struct problem *p = &subprob;
		
		for(int i=0; i<subprob.l; i++)
		{
			free(p->x[i]);
		}
		free(p->x);
		free(p->y);
		
	}
}