############### IMPORTS ###############
import json
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, vstack, coo_matrix
import boto3
from io import BytesIO, StringIO 
import botocore
from botocore.exceptions import ClientError
import math
import ast 
import sys
import time
import timeit 
import zlib
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed
import os
from operator import itemgetter
############### CLIENTS ###############
lambdaClient = boto3.client('lambda')
s3 = boto3.client('s3')
############### CONSTANTS ###############
# Below params as per Sparse DNN Graph Challenge
BIAS_1024 = -0.3
BIAS_4096 = -0.35
BIAS_16384 = -0.4
BIAS_65536 = -0.45
BIAS = 0
ZMAX = 32
S3_LIST_SLEEP_TIME = 0.1
###############     MISC     ###############
USE_COMPRESSION = 1
COMPRESSION_LEVEL = 3 # Zlib
MAX_MSG_LINE_WIDTH = 262144
S3_POLLING_TIMEOUT = 30
S3_BUCKET_MODULO = 10
NUM_THREADS = 3
MIN_JOBS_PER_THREADPOOL = 10
MAX_JOBS_PER_THREADPOOL = 2*MIN_JOBS_PER_THREADPOOL
############### STATS LABELS ###############
NUM_METRICS = 12
METRIC_S3_NUM_WRITES = 0
METRIC_S3_NUM_READS = 1
METRIC_S3_WRITE_SIZE = 2
METRIC_S3_READ_SIZE = 3
METRIC_S3_FOLDER_SCANS = 4
METRIC_ELAPSED_TIME_TOTAL = 5
METRIC_ELAPSED_TIME_COMMUNICATION = 6
METRIC_ELAPSED_TIME_COMPRESSION = 7
METRIC_ELAPSED_TIME_COMPUTATION = 8
METRIC_LAMBDA_RUNTIME = 9
METRIC_MAX_FILE_SIZE = 10
METRIC_MAX_ROW_NNZ = 11
############### GLOBAL VARIABLES ###############
id = 0
level = 0
lambda_handler_start_time = 0
############### IMPORTANT VARIABLES ###############
W = [] # List of CSR
stats = np.zeros(NUM_METRICS, dtype=np.float32)
############################## FUNCTION / CLASS DEFINITIONS ##############################
def process_timer(metric, start_timer):
    end_timer = timeit.default_timer()
    duration = end_timer - start_timer
    update_stats(metric, duration)
#############################################
def debug_timer(reference_time, message):
    current_time = timeit.default_timer()
    # print("[DEBUG TIMER] " , message , "            Elapsed: ", str(current_time - reference_time))
#############################################
def update_stats(metric, val, mode="sum"):
    if mode == "sum":
        stats[metric] += val
    elif mode == "max":
        if val > stats[metric]:
            stats[metric] = val
    elif mode == "min":
        if val < stats[metric]:
            stats[metric] = val
#############################################
def utf8len(s):
# Function to calculate size in bytes of string. 
    return len(s.encode('utf-8'))
#############################################
def multiply_post_process(Z_layer):
    # Add bias to all non-zero elements
    Z_layer.data = np.where(Z_layer.data != 0, Z_layer.data+BIAS, Z_layer.data)
    
    # ReLU
    Z_layer.data = np.where(Z_layer.data < 0, 0, Z_layer.data)
    
    # Threshold with ZMAX
    Z_layer.data = np.where(Z_layer.data > ZMAX, ZMAX, Z_layer.data)
    
    return Z_layer
#############################################
def sigmoid(csr):
    # Function to calculate Sigmoid for a CSR matrix
    x = csr.todense()
    z = np.exp(-x)
    sig = 1 / (1 + z)
    
    return csr_matrix(sig)
#############################################
def readDNNP(weights_loc_in, model_params_in):
    weights_bucket = weights_loc_in["bucket"]
    weights_subfolder = weights_loc_in["subfolder"]
    numNeurons = int(model_params_in["numNeurons"])
    
    global W
    W = []
    
    # Construct filename from event information and ID
    weightsFile = weights_subfolder + "/nnp." + str(id)
    
    # Get nnp.ID from s3/mnist-radixnet-weights/1024
    result = s3.get_object(Bucket=weights_bucket, Key=weightsFile) 
    
    # Define numpy arrays. When we find a length 2 line (with ELEMENT1 > 0), create CSR and append to W. 
    # Then re-instantiate layerWeightData = np.zeros(3, ELEMENT2). R0 = row, R1 = cols, R2 = data. 
    for line in result["Body"].read().splitlines():
        line = line.decode('utf-8')
        elements = line.split()
        
        if len(elements) == 2:
            print(elements)
            # This means the current line is a (layerNum, nnz) pair and marks the start of a new layer.
            layerCount = int(elements[0])
            nnz = int(elements[1])
            
            if elements[0] == '0':
                # Instantiate np zeros array layerWeightData
            
                layerRowIndices = np.zeros(nnz, dtype=int)
                layerColIndices = np.zeros(nnz, dtype=int)
                layerVals = np.zeros(nnz, dtype=np.float32)
                nzCount = 0
            else:
                # First create CSR matrix for previous layer's data and append to W/WT
                layerCSR = csr_matrix((layerVals, (layerRowIndices,layerColIndices)),shape=(numNeurons, numNeurons))
                W.append(layerCSR)
                
                # Then re-instantiate layerWeightData np zeros array
                layerRowIndices = np.zeros(nnz, dtype=int)
                layerColIndices = np.zeros(nnz, dtype=int)
                layerVals = np.zeros(nnz, dtype=np.float32)
                nzCount = 0
                layerCount += 1
        else:
            layerRowIndices[nzCount] = int(elements[0])
            layerColIndices[nzCount] = int(elements[1])
            layerVals[nzCount] = np.float32(elements[2])
            nzCount+= 1
    
    # After final layer's data, need one more flush since we don't encounter another line of length 2
    layerCSR = csr_matrix((layerVals, (layerRowIndices,layerColIndices)),shape=(numNeurons, numNeurons))
    W.append(layerCSR)
#############################################
def readInferenceData(inf_data_loc_in, model_params_in):
    inf_data_bucket = inf_data_loc_in["bucket"]
    inf_data_subfolder = inf_data_loc_in["subfolder"]
    numData = int(model_params_in["numData"])
    numNeurons = int(model_params_in["numNeurons"])
    
    print("In readInferenceData, numData = " , str(numData), " numNeurons = ", str(numNeurons))
    
    # Instantiate lists - we don't know number of lines in train.rank
    infDataRowIndices, infDataColIndices, infDataVals = [], [], []
    
    # Construct filename: train.rank
    infDataFile = inf_data_subfolder + "/train." + str(id)
    
    # Get inference data file from S3
    result = s3.get_object(Bucket=inf_data_bucket, Key=infDataFile) 
    
    # Iterate through lines in train file
    for line in result["Body"].read().splitlines():
    
        # Split line into list of elements (row,col,val) -> (imageID, PixelLoc, 1)
        line = line.decode('utf-8')
        elements = line.split()
        
        # Add values to np array
        infDataRowIndices.append(int(elements[0]))
        infDataColIndices.append(int(elements[1]))
        infDataVals.append(np.float32(elements[2]))
        
    # Create CSR matrix for inference data
    return csr_matrix((infDataVals, (infDataRowIndices,infDataColIndices)),shape=(numData, numNeurons))
#############################################
def SpFF(model_params_in, infDataCSR_in, invoc_params_in, hidden_loc_in, metrics_loc_in):
    # Main Sparse Feed Forward Inference Function

    # Retrieve important variables from JSON parameters.
    numLayers = int(model_params_in["numLayers"])
    numData = int(model_params_in["numData"])
    infDataCSR = infDataCSR_in
    numNeurons = int(model_params_in["numNeurons"])
    total_nworkers = int(invoc_params_in["total_nworkers"])
    batchSize = int(model_params_in["batchSize"])
    numBatches = int(model_params_in["numBatches"])
    
    numDataToUse = batchSize * numBatches
    
    global stats
    stats = np.zeros(NUM_METRICS, dtype=np.float32)
  
    # Set bias global variable
    global BIAS
    if numNeurons == 1024:
        BIAS = BIAS_1024
    elif numNeurons == 4096:
        BIAS = BIAS_4096
    elif numNeurons == 16384:
        BIAS = BIAS_16384
    elif numNeurons == 65536:
        BIAS = BIAS_65536
    else:
        BIAS = 0
    
    #===========================================================================
    # [PHASE 1] Populate list of sources/targets for all layers.
    #===========================================================================
    inf_batch_start_time = timeit.default_timer()
    
    # Start looping through Inference data examples
    batchCount = 0
    for batchStart in range(0,numDataToUse,batchSize):
        
        # Adjust batchsize to cater for non-divisible batchsize with numData
        if batchCount == (numBatches - 1):
            batchSize = numDataToUse - (batchCount*batchSize)
        
        print("batchStart: ", str(batchStart))
        print("batchSize : ", str(batchSize))
        
        print("**********************************")
        print("Starting inference batch: ", str(batchCount), " from example " , str(batchStart), " to ", str(batchStart+batchSize - 1))
        print("**********************************")     
       
        if batchSize == numData:
            print("Appending infDataCSR to H, since batchSize == numData")
            H = infDataCSR.transpose()
        else:
            batch_indexer_start_time = timeit.default_timer()
            H = infDataCSR[batchStart:(batchStart+batchSize),:].transpose()
            batch_indexer_end_time = timeit.default_timer()
            print("Worker id: ", str(id), ", batch_indexer_time_elapsed: ", str(batch_indexer_end_time - batch_indexer_start_time))
            
        print("H.shape: ", str(H.shape))
        
        #===========================================================================
        # [PHASE 3] Feedforward inference loop
        #===========================================================================
        for layer in range(numLayers):
            print("Inference Batch: ", str(batchStart), " to ", str(batchStart+batchSize-1) , " Layer: ", str(layer))
            
            layer_start_reference_time = timeit.default_timer()

            if layer > 0:
                z_delete_end_time = timeit.default_timer()
                del W[0]
                w_delete_end_time = timeit.default_timer()
                debug_timer(z_delete_end_time, "Deleted W[layer - 1]")
           
            # [START COMPUTATION TIMER]
            first_multiply_start_time = timeit.default_timer()
            Z = W[0] * H
            process_timer(METRIC_ELAPSED_TIME_COMPUTATION, first_multiply_start_time)
            
            # [START POST PROCESS COMPUTATION TIMER]
            multiply_post_process_start_time = timeit.default_timer()
            H = multiply_post_process(Z)
            process_timer(METRIC_ELAPSED_TIME_COMPUTATION, multiply_post_process_start_time)
            
            # # End of layer
            
        # End of layer loop for current inference example

        # Get max rows and max vals for current inference batch. For max_rows, we can easily identify max_rows with argmax function, it returns row 0
        # if all rows = 0. For max vals, we first call max function to give a COO matrix of max values. We then pull the column indices and data,
        # and map over the np.zeros array my_max_vals which we declared at the start. This only updates the max for columns (i.e. samples in batch) 
        # where there is a non-zero value, and leaves others as 0.
        
        print("H shape: ", str(H.shape))
        
        my_max_vals = np.zeros(batchSize,dtype=np.float32)
        max_vals_coo = H.max(axis=0)                                                    # COO matrix (1,20)
        max_val_coo_indices = max_vals_coo.col                                          # Col array from above coo
        max_val_coo_data = max_vals_coo.data                                            # Data array from above coo
        
        my_max_vals[max_val_coo_indices] += max_val_coo_data         
        
        print("*************************************************")
        print("INFERENCE RESULTS FOR WORKER ", str(id))
        print("*************************************************")
        print()
        print("type(max_vals_coo)                    : ",type(max_vals_coo))
        print("max_vals_coo nnz                      : ", max_vals_coo.getnnz())
        print(max_vals_coo)
        print()
        print("*************************************************")
        print("END OF INFERENCE RESULTS FOR WORKER ", str(id))
        print("*************************************************")
        
        # Timing
        process_timer(METRIC_ELAPSED_TIME_TOTAL, inf_batch_start_time)
        

#############################################
def output_metrics(s3, metrics_loc, invoc_params, model_params):
    
    # Extract S3 location from metrics_loc
    bucket = str(metrics_loc["bucket"])
    subfolder = str(metrics_loc["subfolder"])
    filename = str(metrics_loc["filename"])
    
    output_list = []
    
    # Metric Params
    output_list.append(str(metrics_loc["expt_id"]))
    
    # Config
    config = str(metrics_loc['filename'].split("_",1)[1])
    output_list.append(config.split(".")[0])
    
    # Invoc Params
    output_list.append(str(invoc_params["bfr"]))
    output_list.append(str(invoc_params["nlevels"]))
    output_list.append(str(invoc_params["total_nworkers"]))
     
    # Model Params
    output_list.append(str(model_params["numNeurons"]))
    output_list.append(str(model_params["numLayers"]))
    output_list.append(str(model_params["numData"]))
    output_list.append(str(model_params["batchSize"]))
    output_list.append(str(model_params["numBatches"]))
    
    for i in range(NUM_METRICS):
        output_list.append(str(stats[i]))
    
    csvio = StringIO()
    writer = csv.writer(csvio)
    writer.writerow(output_list)
    s3.put_object(Body=csvio.getvalue(), ContentType='text/csv', Bucket=bucket, Key=subfolder + "/" + filename) 
    csvio.close()
   
#############################################
def lambda_handler(event, context):
    
    global lambda_handler_start_time
    lambda_handler_start_time = timeit.default_timer()
    
    #################### PARSING EVENT, SETTTING VARIABLES ####################
    weights_loc = event["data_params"]["weights_loc"]
    inf_data_loc = event["data_params"]["inf_data_loc"]
    hidden_loc = event["data_params"]["hidden_loc"]
    metrics_loc = event["data_params"]["metrics_loc"]
    invoc_params = event["invoc_params"]
    parent_params = event["parent_params"]
    model_params = event["model_params"]

    ######################### CHILD INVOCATION #########################
    global level
    level = int(parent_params["p_level"]) + 1
    
    global id
    print("I am id: ", str(id))
    
    ######################### READING FROM S3 #########################
    readDNNP(weights_loc, model_params)
    print("readDNNP done")
    
    infDataCSR = readInferenceData(inf_data_loc, model_params)
    print("readInferenceData done")
    
    ######################### INFERENCE #########################
    print("Entering SpFF")
    SpFF(model_params, infDataCSR, invoc_params, hidden_loc, metrics_loc)
    print("Inference run complete!")
    
    process_timer(METRIC_LAMBDA_RUNTIME, lambda_handler_start_time)
    
    ######################### METRICS #########################
    if (id == 0):

        print()
        print("*********************************************************************************************")
        print("[BASELINE TOTALS] Inference run metric totals, averaged across all workers:")
        print()
        print("METRIC_ELAPSED_TIME_TOTAL         : " + str(stats[METRIC_ELAPSED_TIME_TOTAL]))
        print("METRIC_ELAPSED_TIME_COMMUNICATION : " + str(stats[METRIC_ELAPSED_TIME_COMMUNICATION]))
        print("METRIC_ELAPSED_TIME_COMPUTATION   : " + str(stats[METRIC_ELAPSED_TIME_COMPUTATION]))
        print("*********************************************************************************************")
        print()
        print("*********************************************************************************************")
        print("[MISC METRICS]")
        print()
        print("% TIME COMMUNICATION              : " + str((stats[METRIC_ELAPSED_TIME_COMMUNICATION] / stats[METRIC_ELAPSED_TIME_TOTAL]) * 100))
        print("% TIME COMPUTATION                : " + str((stats[METRIC_ELAPSED_TIME_COMPUTATION] / stats[METRIC_ELAPSED_TIME_TOTAL]) * 100))
        print("TOTAL LAMBDA RUNTIME              : " + str(stats[METRIC_LAMBDA_RUNTIME]))
        print("*********************************************************************************************")
   
        output_metrics(s3, metrics_loc, invoc_params, model_params)
        

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Sparse DNN Worker Completed Successfully!",
        }),
    }  