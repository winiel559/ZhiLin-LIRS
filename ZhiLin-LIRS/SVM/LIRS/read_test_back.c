#include <math.h>
#include <time.h>
#include <algorithm>
#include <set>
#include <assert.h>
#include "zlib/zlib.h"
#include "block.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct problem my_get_block(int sub_id, unsigned long long *offset, int **assign_table, FILE *fp)
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
		fseek(fp, offset[assign_table[sub_id][i]], SEEK_SET);
		
		fscanf(fp,"%d",&subprob.y[i]);

		feature_node *xi = node_buffer;
		
		int k=0;
		
		while(k < subprob.n)
		{
			fscanf(fp,"%d:%lf",&xi->index,&xi->value);
			
			//printf("xi->index=%d xi->value=%f \n",xi->index,xi->value);
			if(xi->index==1 || xi->index==-1)
			{
				//printf("y=%d\n",subprob.y[i+1]); 
				xi->index = -1;
				break;
			}
			xi++;
			k++;
		}
		node[i] = Malloc(feature_node,k+1);

		for(int j=0; j<k+1; j++)
			node[i][j] = node_buffer[j];
		
		subprob.x = node;	
	}
	
	/*
	for(int i=0; i<subprob.l; i++)
	{
		printf("subprob: %d , instance_id: %d, y: %d\n",sub_id, assign_table[sub_id][i], subprob.y[i]);
		for(int j=0; j<10; j++)
		{
			printf("index[%d]=%d , value[%d]=%lf\n",j,subprob.x[i][j].index,j,subprob.x[i][j].value);
		}
	}*/
	
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
	FILE *fp;
	fp = fopen("../data/webspam.train_14G", "r");	
	//fp = fopen("../data/webspam.train", "r");
	//fp = fopen("123", "r");
    
	struct problem total_problem;	
	total_problem.l = 200000;
	int l = total_problem.l;
	//total_problem.l = 100;
	total_problem.n = 16609143;
	total_problem.bias = -1;
	
	unsigned long long *offset = Malloc(unsigned long long,total_problem.l+1);

	int *y = Malloc(int,total_problem.l);
	
	total_problem.y = y;

	
	char *buf = Malloc(char,1000000);
	int i = 0;
	unsigned long long accu = 0;
	while(fgets(buf,1000000,fp) != NULL)
    {
		/*
		char *token = strtok(buf, " ");
		while (token != NULL)
		{
			char *token2 = strtok(token, ":");
			while (token2 != NULL)
			{
				token2 = strtok(NULL, ":");
			}
			if(i<10)
				printf("%s\n", token2);
			token = strtok(NULL, " ");
		}*/
		/*
		char index_buf[20];
		char value_buf[20];
		int start = 3;
		for(int j=3; j<1000000; j++)
		{
			if(buf[j] == ':')
			{
				for(int k=start; k<j; k++)
				{
					index_buf[k-start] = buf[k];
				}
				start = j + 1 ; //value start
			}
			
			if(buf[j] == ' ')
			{
				for(int k=start; k<j; k++)
				{
					value_buf[k-start] = buf[k];
				}
				start = j + 1 ;
				if(j<1000 && i<1000)
					printf("value_buf: %s\n",value_buf);
			}
			
			if(buf[j] == '\n' || buf[j] == '\0')
			{
				for(int k=start; k<j; k++)
				{
					value_buf[k-start] = buf[k];
				}
				break;
			}
		}
		*/
		
		/*
		int start;
		char index_buf[10];
		for(int j=3; j<9; j++)
			index_buf[j-3] = buf[j];
		printf("index: %s\n",index_buf);
		int a = atoi ( index_buf );
		printf("index: %d\n",a);*/
		
        //printf("%d\t%s\n",i,buf);
		
		for(int j=0; j<1000000; j++)
			if(buf[j] == '\n' || buf[j] == '\0')
			{
				offset[i] = accu;
				accu += j+1;
				break;
			}
		
		i++;
		
	}
	
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
	/*
	for(int i=0; i<1; i++)
		subprob = my_get_block(i, offset, assign_table, fp);
	*/
	
	init_load_t += difftime(time(NULL), startload_t);
	//for(int i=0; i<200000; i++)
		//printf("offset[%d] = %llu\n",i,offset[i]);
	printf("initial load time: %f\n",init_load_t);
	
}
