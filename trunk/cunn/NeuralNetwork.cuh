#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
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
const int kNumberOfUnits[]={4,20,4};
/*This indicates how many neural units in each level.
 *kNumberOfUnits[0] == kInputSize;
 *kNumberOfUnits[kLevelOfNeuralNetwork-1] == kOutputSize.
 */

struct NN_setting{
	float	learning_rate;
	float	impulse;
	int		max_threads_per_block;
	bool	back_propagation_to_level_one;
	NN_setting(){
		learning_rate=0.5f;
		impulse=0.5f;
		max_threads_per_block=512;
		back_propagation_to_level_one=false;
	}
};

void OutputWeight(float *(h_weight[]),float *(d_weight[]));
void set(float learning_rate,float impulse,int max_threads_per_block);
void train(const int k_number_of_training,const int k_number_of_testing,const float* input,const float* output);

__global__
	void ForwardCalc(float *wei,float *in,float *o,float *delta,int number_of_last_level,int number_of_this_level);
__global__
	void BackPropagation(float *o,float *o_last,float *target,float *delta,float* delta_front,float *wei,float *delta_wei,
		const int k_level,const int k_number_of_this_level,const int k_number_of_last_level,NN_setting* d_setting);
__global__
	void BackPropagation_1st_level(float *input_vector,float *delta,const int k_number_of_this_level);

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

NN_setting h_setting;
NN_setting *d_setting;
void set(float learning_rate=0.5f,float impulse=0.5f,int max_threads_per_block=512){
		h_setting.learning_rate=learning_rate;
		h_setting.impulse=impulse;
		h_setting.max_threads_per_block=max_threads_per_block;
		cudaMalloc(&d_setting,sizeof(NN_setting));
		cudaMemcpy(d_setting,&h_setting,sizeof(NN_setting),cudaMemcpyHostToDevice);
	}
void train(const int k_number_of_training,const int k_number_of_testing,const float* input,const float* output){
		//Numberofsample: the number of samples for trainning.
		//Input[i][j]   : the jth input for the ith sample.
	
		cudaError_t cudaStatus;
		float *d_input,*d_output;
			//d_input[i+kInputSize*j] indicates the ith input of the jth sample.
			//d_output[i+kOuputSize*j] indicates the ith output of the jth sample.
		cudaStatus=cudaMalloc(&d_input,kInputSize*(k_number_of_training+k_number_of_testing)*sizeof(float));
			//malloc spaces for input in GDDR
		cudaStatus=cudaMalloc(&d_output,kOutputSize*(k_number_of_training+k_number_of_testing)*sizeof(float));
			//malloc spaces for output in GDDR
		cudaStatus=cudaMemcpy(d_input,input,kInputSize*(k_number_of_training+k_number_of_testing)*sizeof(float),cudaMemcpyHostToDevice);
		cudaStatus=cudaMemcpy(d_output,output,kOutputSize*(k_number_of_training+k_number_of_testing)*sizeof(float),cudaMemcpyHostToDevice);
			//Copy both the input data and the output data to the RAM of the vedio card.

		float *(d_weight[kLevelOfNeuralNetwork-1]);
		float *(d_delta_weight[kLevelOfNeuralNetwork-1]);
		float *(h_weight[kLevelOfNeuralNetwork-1]);
			//IMPORTANT:
			//d_weight[i]+j*kNumberOfUnits[i]+k means the weight of the edge 
			//from the kth unit in level i to the jth unit in level i+1.
		for(int i=0;i<kLevelOfNeuralNetwork-1;i++){
			srand(time(NULL));
			cudaStatus=cudaMalloc(&d_weight[i],kNumberOfUnits[i]*kNumberOfUnits[i+1]*sizeof(float));
			cudaStatus=cudaMalloc(&d_delta_weight[i],kNumberOfUnits[i]*kNumberOfUnits[i+1]*sizeof(float));
			h_weight[i]=(float*)malloc(kNumberOfUnits[i]*kNumberOfUnits[i+1]*sizeof(float));
			for(int j=0;j<kNumberOfUnits[i]*kNumberOfUnits[i+1];j++)
				*(h_weight[i]+j)=rand()/(float)RAND_MAX-0.5f;
			cudaStatus=cudaMemcpy(d_weight[i],h_weight[i],kNumberOfUnits[i]*kNumberOfUnits[i+1]*sizeof(float),cudaMemcpyHostToDevice);
			memset(h_weight[i],0,kNumberOfUnits[i]*kNumberOfUnits[i+1]*sizeof(float));
			cudaStatus=cudaMemcpy(d_delta_weight[i],h_weight[i],kNumberOfUnits[i]*kNumberOfUnits[i+1]*sizeof(float),cudaMemcpyHostToDevice);
		}
		
		float *(d_o[kLevelOfNeuralNetwork]);
		float *h_o_o;
			//the output of each level of Neural Network
		for(int i=0;i<kLevelOfNeuralNetwork;i++){
			cudaStatus=cudaMalloc(&d_o[i],kNumberOfUnits[i]*sizeof(float));
		}
		h_o_o=(float *)malloc(kNumberOfUnits[kLevelOfNeuralNetwork-1]*sizeof(float));

		float *(d_delta[kLevelOfNeuralNetwork]);
		for(int i=0;i<kLevelOfNeuralNetwork;i++){
			cudaStatus=cudaMalloc(&d_delta[i],kNumberOfUnits[i]*sizeof(float));
		}
		float training_error=0,testing_error=0,testing_wrong=0;
		for(int i=0;i<100000;i++){
			for(int k=0;k<k_number_of_training;k++){
				//copy the input of the sample to the d_o[0]
				cudaMemcpy(d_o[0],d_input+k*kInputSize,kInputSize*sizeof(float),cudaMemcpyDeviceToDevice);
				for(int j=1;j<kLevelOfNeuralNetwork;j++){
					int threads_per_block=std::min(h_setting.max_threads_per_block,kNumberOfUnits[j]);
					int blocks_per_grid=(kNumberOfUnits[j]+threads_per_block-1)/threads_per_block;
					ForwardCalc<<<blocks_per_grid,threads_per_block>>>(d_weight[j-1],d_o[j-1],d_o[j],d_delta[j],kNumberOfUnits[j-1],kNumberOfUnits[j]);
					cudaStatus = cudaGetLastError();
				}
				cudaMemcpy(h_o_o,d_o[kLevelOfNeuralNetwork-1],kOutputSize*sizeof(float),cudaMemcpyDeviceToHost);
				for(int j=0;j<kOutputSize;j++){
					if(*(output+kOutputSize*k+j)>0.5f)
						training_error+=abs(1-h_o_o[j]);
					else
						training_error+=abs(h_o_o[j]);
				}
				for(int j=kLevelOfNeuralNetwork-1;j>=1;j--){
					int threads_per_block=std::min(h_setting.max_threads_per_block,kNumberOfUnits[j]);
 					int blocks_per_grid=(kNumberOfUnits[j]+threads_per_block-1)/threads_per_block;
					BackPropagation<<<blocks_per_grid,threads_per_block>>>(d_o[j],d_o[j-1],d_output+kOutputSize*k,
										d_delta[j],d_delta[j-1],d_weight[j-1],d_delta_weight[j-1],j,kNumberOfUnits[j],kNumberOfUnits[j-1], d_setting);	
					cudaStatus = cudaGetLastError();
				}
				if(h_setting.back_propagation_to_level_one==true){
					int threads_per_block=std::min(h_setting.max_threads_per_block,kNumberOfUnits[0]);
 					int blocks_per_grid=(kNumberOfUnits[0]+threads_per_block-1)/threads_per_block;
					BackPropagation_1st_level<<<blocks_per_grid,threads_per_block>>>(d_o[0],d_delta[0],kNumberOfUnits[0]);
					//copy the word vector back to the dictionary
					//cudaMemcpy(???to???,kInputSize*sizeof(float),cudaMemcpyDeviceToDevice);
				}
			}//for k, end of training
			for(int k=k_number_of_training;k<k_number_of_training+k_number_of_testing;k++){
				cudaMemcpy(d_o[0],d_input+k*kInputSize,kInputSize*sizeof(float),cudaMemcpyDeviceToDevice);
				cudaStatus = cudaGetLastError();
				for(int j=1;j<kLevelOfNeuralNetwork;j++){
					int threads_per_block=std::min(h_setting.max_threads_per_block,kNumberOfUnits[j]);
					int blocks_per_grid=(kNumberOfUnits[j]+threads_per_block-1)/threads_per_block;
					ForwardCalc<<<blocks_per_grid,threads_per_block>>>(d_weight[j-1],d_o[j-1],d_o[j],d_delta[j],kNumberOfUnits[j-1],kNumberOfUnits[j]);
					cudaStatus = cudaGetLastError();
				}
				cudaMemcpy(h_o_o,d_o[kLevelOfNeuralNetwork-1],kOutputSize*sizeof(float),cudaMemcpyDeviceToHost);
				for(int j=0;j<kOutputSize;j++){
					if(*(output+kOutputSize*k+j)>0.5f){
						testing_error+=abs(1-h_o_o[j]);
						if(h_o_o[j]<0.5f)
							testing_wrong++;
					}else{
						testing_error+=abs(h_o_o[j]);
						if(h_o_o[j]>0.5f)
							testing_wrong++;
					}
				}
			}//for k, end of testing
			if(i%50==49){
				printf("trainning round %6d: ",i+1);
				printf("trainning error: %10.5f ",training_error);training_error=0;
				printf("testing error: %10.5f ",testing_error);testing_error=0;
				printf("wrong classification: %10.5f ",testing_wrong);testing_wrong=0;
				printf("\n");
			}
		}//for i

		//OutputWeight(h_weight,d_weight);
		//system("PAUSE");
		return;
		
}

__global__
	void ForwardCalc(float *wei,float *in,float *o,float *delta,int number_of_last_level,int number_of_this_level){
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	if(i<number_of_this_level){
		o[i]=0;
		for(int j=0;j<number_of_last_level;j++){
			o[i]+=wei[i*number_of_last_level+j]*in[j];
		}
		o[i]=2/(1+expf(-o[i]));
		delta[i]=0.0f;
	}
	__syncthreads();
}

__global__
	void BackPropagation(float *o,float *o_last,float *target,float *delta,float* delta_front,float *wei,float *delta_wei,
		const int k_level,const int k_number_of_this_level,const int k_number_of_last_level,NN_setting* d_setting){
		//in this section, when calculating the back propagation, the calculation do as follows:
		//calculate the delta of one unit
		//add up \sum delta_j * weight_ij for each unit i
		//calc each delta wij
		int i=blockDim.x*blockIdx.x+threadIdx.x;
		
		if(i<k_number_of_this_level){
			if(k_level==(kLevelOfNeuralNetwork-1)){
				delta[i]=o[i]*(2-o[i])*(target[i]-o[i]);
			}else{
				delta[i]=o[i]*(2-o[i])*delta[i];
			}
			
			//This step can be skiped if k_level==1 && back_propagation_to_level_one==false
			if(d_setting->back_propagation_to_level_one == false && k_level==1)
				for(int j=0;j<k_number_of_last_level;j++)
					delta_front[j]+=delta[i]*wei[j+i*k_number_of_last_level];
	
			for(int j=0;j<k_number_of_last_level;j++){
				wei[j+i*k_number_of_last_level]+=0.5*delta[i]*o_last[j]+0.5*delta_wei[j+i*k_number_of_last_level];
				delta_wei[j+i*k_number_of_last_level]=0.5*delta[i]*o_last[j];
				//wei[j+i*k_number_of_last_level]*=(1-0.7*0.001);
			}
	
		}

	    //NO CODE HERE
		__syncthreads();
	}

__global__
	void BackPropagation_1st_level(float *input_vector,float *delta,const int k_number_of_this_level){
		//in this section, when calculating the back propagation of the 1st value
		//to change the input vector
		int i=blockDim.x*blockIdx.x+threadIdx.x;
		
		if(i<k_number_of_this_level){
			input_vector[i]+=0.5*delta[i];
		}

	    //NO CODE HERE
		__syncthreads();
	}