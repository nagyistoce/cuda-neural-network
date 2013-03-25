#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include <algorithm>
#include "..\\..\\..\\wenx.h"
#pragma once

__device__ float expf(float x);

const int kInputSize=4;
//This indicates how many inputs are there in one sample.
const int kOutputSize=4;
//This indicates how many outputs are there in one sample.
const int kLevelOfNeuralNetwork=3;
//This indicates how many levels are there in a Neural Network(NN).
//if kLevelOfNeuralNetwork==3, it means a NN with one input level,
//one output level and one hidden level.
const int kNumberOfUnits[]={4,200,4};
/*This indicates how many neural units in each level.
 *kNumberOfUnits[0] == kInputSize;
 *kNumberOfUnits[kLevelOfNeuralNetwork-1] == kOutputSize.
 */
__global__
void ForwardCalc(float *wei,float *in,float *o,float *delta,int number_of_last_level,int number_of_this_level);
__global__
void BackPropagation(float *o,float *o_last,float *target,float *delta,float* delta_front,float *wei,
	const int k_level,const int k_number_of_this_level,const int k_number_of_last_level);

void OutputWeight(float *(h_weight[]),float *(d_weight[])){
	for(int i=0;i<kLevelOfNeuralNetwork-1;i++){
			cudaMemcpy(h_weight[i],d_weight[i],kNumberOfUnits[i]*kNumberOfUnits[i+1]*sizeof(float),cudaMemcpyDeviceToHost);
			for(int j=0;j<kNumberOfUnits[i];j++){
				for(int k=0;k<kNumberOfUnits[i+1];k++)
					printf("%lf  ",*(h_weight[i]+j+k*kNumberOfUnits[i]));
				printf("\n");
			}
	}
}


void train(int numberofsample,const float* input,const float* output){
		//Numberofsample: the number of samples for trainning.
		//Input[i][j]   : the jth input for the ith sample.
	
		cudaError_t cudaStatus;
		float *d_input,*d_output;
			//d_input[i+kInputSize*j] indicates the ith input of the jth sample.
			//d_output[i+kOuputSize*j] indicates the ith output of the jth sample.
		cudaMalloc(&d_input,kInputSize*numberofsample*sizeof(float));
			//malloc spaces for input in GDDR
		cudaMalloc(&d_output,kOutputSize*numberofsample*sizeof(float));
			//malloc spaces for output in GDDR
		cudaMemcpy(d_input,input,kInputSize*numberofsample*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d_output,output,kOutputSize*numberofsample*sizeof(float),cudaMemcpyHostToDevice);
			//Copy both the input data and the output data to the RAM of the vedio card.

		float *(d_weight[kLevelOfNeuralNetwork-1]);
		float *(h_weight[kLevelOfNeuralNetwork-1]);
			//IMPORTANT:
			//d_weight[i]+j*kNumberOfUnits[i]+k means the weight of the edge 
			//from the kth unit in level i to the jth unit in level i+1.
		for(int i=0;i<kLevelOfNeuralNetwork-1;i++){
			srand(time(NULL));
			cudaMalloc(&d_weight[i],kNumberOfUnits[i]*kNumberOfUnits[i+1]*sizeof(float));
			h_weight[i]=(float*)malloc(kNumberOfUnits[i]*kNumberOfUnits[i+1]*sizeof(float));
			for(int j=0;j<kNumberOfUnits[i]*kNumberOfUnits[i+1];j++)
				*(h_weight[i]+j)=rand()/(float)RAND_MAX-0.5f;
			cudaMemcpy(d_weight[i],h_weight[i],kNumberOfUnits[i]*kNumberOfUnits[i+1]*sizeof(float),cudaMemcpyHostToDevice);
		}
		
		float *(d_o[kLevelOfNeuralNetwork]);
		float *h_o_o;
			//the output of each level of Neural Network
		for(int i=0;i<kLevelOfNeuralNetwork;i++){
			cudaMalloc(&d_o[i],kNumberOfUnits[i]*sizeof(float));
		}
		h_o_o=(float *)malloc(kNumberOfUnits[kLevelOfNeuralNetwork-1]*sizeof(float));

		float *(d_delta[kLevelOfNeuralNetwork]);
		for(int i=0;i<kLevelOfNeuralNetwork;i++){
			cudaStatus=cudaMalloc(&d_delta[i],kNumberOfUnits[i]*sizeof(float));
		}
		float errr=0;
		for(int i=0;i<300;i++){
			for(int k=0;k<numberofsample;k++){
				//copy the input of the sample to the d_o[0]
				cudaMemcpy(d_o[0],d_input+k*kInputSize,kInputSize*sizeof(float),cudaMemcpyDeviceToDevice);
				for(int j=1;j<kLevelOfNeuralNetwork;j++){
					int threads_per_block=std::min(1024,kNumberOfUnits[j]);
					int blocks_per_grid=(kNumberOfUnits[j]+threads_per_block-1)/threads_per_block;
					if(blocks_per_grid>1)
						err("TOO MANY BLOCKS RUNNING AT THE SAME TIME",__FILE__,__LINE__);
					ForwardCalc<<<1,threads_per_block>>>(d_weight[j-1],d_o[j-1],d_o[j],d_delta[j],kNumberOfUnits[j-1],kNumberOfUnits[j]);
				}
				cudaMemcpy(h_o_o,d_o[kLevelOfNeuralNetwork-1],kOutputSize*sizeof(float),cudaMemcpyDeviceToHost);
				for(int j=0;j<kOutputSize;j++){
					if(*(output+kOutputSize*k+j)>0.5f)
						errr+=abs(1-h_o_o[j]);
					else
						errr+=abs(h_o_o[j]);
				}
				for(int j=kLevelOfNeuralNetwork-1;j>0;j--){
					//cudaMemset(d_delta[j-1],0,kNumberOfUnits[j-1]*sizeof(float));
					//set all the delta in the fronter level to be zero
					int threads_per_block=std::min(1024,kNumberOfUnits[j]);
// 					int blocks_per_grid=(kNumberOfUnits[j]+threads_per_block-1)/threads_per_block;
// 					if(blocks_per_grid>1)
// 						err("TOO MANY BLOCKS RUNNING AT THE SAME TIME",__FILE__,__LINE__);
					
					
					BackPropagation<<<1,threads_per_block>>>(d_o[j],d_o[j-1],d_output+kOutputSize*k,
										d_delta[j],d_delta[j-1],d_weight[j-1],j,kNumberOfUnits[j],kNumberOfUnits[j-1]);	
				}
				
			}
			if(i%1000==999)
				printf("%10d::%10.5lf\n",i+1,errr),errr=0;
			if(i==299000)
				printf("\n");
		}

		OutputWeight(h_weight,d_weight);
		//system("PAUSE");
		return;
		
}

__global__
	void ForwardCalc(float *wei,float *in,float *o,float *delta,int number_of_last_level,int number_of_this_level){
	int i=/*blockDim.x*blockIdx.x+*/threadIdx.x;
	if(i<number_of_this_level){
		o[i]=0;
		for(int j=0;j<number_of_last_level;j++){
			o[i]+=wei[i*number_of_last_level+j]*in[j];
		}
		o[i]=1/(1+expf(-o[i]));
		delta[i]=0.0f;
	}
	__syncthreads();
}

__global__
	void BackPropagation(float *o,float *o_last,float *target,float *delta,float* delta_front,float *wei,
		const int k_level,const int k_number_of_this_level,const int k_number_of_last_level){
		//in this section, when calculating the back propagation, the calculation do as follows:
		//calculate the delta of one unit
		//add up \sum delta_j * weight_ij for each unit i
		//calc each delta wij
		int i=/*blockDim.x*blockIdx.x+*/threadIdx.x;
		
		if(i<k_number_of_this_level){
			if(k_level==(kLevelOfNeuralNetwork-1)){
				delta[i]=o[i]*(1-o[i])*(target[i]-o[i]);
			}else{
				delta[i]=o[i]*(1-o[i])*delta[i];
			}
			
			if(k_level>1){
				for(int j=0;j<k_number_of_last_level;j++)
					delta_front[j]+=delta[i]*wei[j+i*k_number_of_last_level];
			}
				
			for(int j=0;j<k_number_of_last_level;j++)
				wei[j+i*k_number_of_last_level]+=0.7*delta[i]*o_last[j];
	
		}

	    //NO CODE HERE
		__syncthreads();
	}