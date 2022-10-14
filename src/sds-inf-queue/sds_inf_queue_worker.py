############### IMPORTS ###############
import json
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, vstack, coo_matrix
import boto3
from io import BytesIO, StringIO 
from boto3.dynamodb.conditions import Key
import botocore
from botocore.exceptions import ClientError
import math
import time
import ast 
import sys
import timeit 
import zlib
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed
import os
from operator import itemgetter

############### CLIENTS ###############
lambdaClient = boto3.client('lambda', region_name='eu-west-1')
s3 = boto3.client('s3', region_name='eu-west-1')
dynamodb = boto3.resource('dynamodb', region_name='eu-west-1')
dynamodb_client = boto3.client('dynamodb', region_name='eu-west-1')
############### CONSTANTS ###############
SPARSE_DNN_WORKER = "" # Replace with ARN of deployed Lambda function for this worker.
TOPIC_NAME_PREFIX = 'sparseDNNTopic' 
COORDINATION_TOPIC_NAME = 'sparseDNNCoordinationTopic' 
QUEUE_NAME_PREFIX = '_SMPI_WQ_DEMO' 
MAX_BATCH_MESSAGE_BYTES = 262144
MAX_ALLOWED_BATCH_MESSAGE_SIZE = int(MAX_BATCH_MESSAGE_BYTES * 0.99)
MAX_NUM_MESSAGES = 10
VISIBILITY_TIMEOUT = 10
WAIT_TIME_SECS = 1
MAX_MSG_LINE_WIDTH = 262144
POLL_TIMEOUT_DURATION = 60
MAX_NNZ_PER_MESSAGE = 40000      
MAX_NNZ_PER_ROW = 10000          
COMPRESSION_LEVEL = 3
USE_COMPRESSION = 1
DYNAMODB_TABLE_NAME = "SPARSEDNN_NULL_TARGETS"
SNS_PUBLISH_THREADS = 4
MAX_SNS_PUBLISH_RETRIES = 4
USE_MULTITOPIC = True

# Below params as per Sparse DNN Graph Challenge
BIAS_1024 = -0.3
BIAS_4096 = -0.35
BIAS_16384 = -0.4
BIAS_65536 = -0.45
BIAS = 0
ZMAX = 32
############### STATS LABELS ###############
NUM_METRICS = 17

METRIC_PUBLISH_CALLS = 0
METRIC_NUM_MESSAGES_SENT = 1
METRIC_SIZE_MESSAGES_SENT = 2
METRIC_RECEIVE_POLL_CALLS = 3
METRIC_NUM_MESSAGES_RECEIVED = 4
METRIC_SIZE_MESSAGES_RECEIVED = 5
METRIC_ELAPSED_TIME_BATCH = 6
METRIC_ELAPSED_TIME_COMMUNICATION = 7
METRIC_ELAPSED_TIME_COMPUTATION = 8
METRIC_ELAPSED_TIME_COMPRESSION = 9
METRIC_LAMBDA_RUNTIME = 10
METRIC_DYNAMODB_ITEMS_WRITTEN = 11
METRIC_DYNAMODB_TOTAL_ITEMS_READ = 12
METRIC_DYNAMODB_UNIQUE_ITEMS_READ = 13
METRIC_MAX_ROW_NNZ = 14
METRIC_MAX_MESSAGE_SIZE = 15
METRIC_MAX_BATCH_MSG_SIZE = 16
############### GLOBAL VARIABLES ###############
id = 0
topic_arn = ""
coordination_topic_arn=""
queue_url = ""
queue_arn = ""
level = 0
lambda_handler_start_time = 0
############### IMPORTANT VARIABLES ###############
W = [] # List of CSR
Hsend, Hrecv = [], [] # List of CSC
stats = np.zeros(NUM_METRICS, dtype=np.float32)
stats_allworkers = np.zeros(NUM_METRICS, dtype=np.float32)
############################## FUNCTION / CLASS DEFINITIONS ##############################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# CLASS DEFINITION
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
class LayerStats:
    
    # Class Variables
    LS_NUM_LAYERS = 120
    # LS_NUM_LAYERS = 4
    LS_NUM_METRICS = 27
    
    LS_WORKER_ID = 0
    LS_LAYER = 1
    LS_LAYER_START_TIME = 2
    LS_PUBLISH_NUM = 3
    LS_PUBLISH_TOTAL_SIZE = 4
    LS_PUBLISH_MAX_SIZE = 5
    LS_PUBLISH_MIN_SIZE = 6
    LS_PUBLISH_AVG_SIZE = 7
    LS_MSG_NUM = 8
    LS_MSG_TOTAL_SIZE = 9
    LS_MSG_MAX_SIZE = 10
    LS_MSG_MIN_SIZE = 11
    LS_MSG_AVG_SIZE = 12
    LS_NUM_TARGETS = 13
    LS_NNZ = 14 # NNZ in H[layer]
    LS_CHUNKS_SENT = 15
    LS_CHUNKS_PER_TARGET = 16 # Average
    LS_CHUNK_TOTAL_SIZE = 17
    LS_CHUNK_MIN_SIZE = 18
    LS_CHUNK_MAX_SIZE = 19
    LS_CHUNK_AVG_SIZE = 20
    LS_NNZ_SENT = 21
    LS_NNZ_PER_TARGET = 22 # Average
    LS_ROWS_SENT = 23
    LS_ROWS_PER_TARGET = 24 # Average
    LS_LAYER_END_TIME = 25
    LS_LAYER_DURATION = 26

    INTEGER_METRICS = [0,1,3,8,13,14,15,21,23]

    METRICS = ["WORKER_ID", "LAYER", "LAYER_START_TIME",  "PUBLISH_NUM",       "PUBLISH_TOTAL_SIZE",   "PUBLISH_MAX_SIZE", "PUBLISH_MIN_SIZE", "PUBLISH_AVG_SIZE", 
               "MSG_NUM",           "MSG_TOTAL_SIZE",    "MSG_MAX_SIZE",       "MSG_MIN_SIZE",         "MSG_AVG_SIZE",     "LS_NUM_TARGETS",
               "NNZ",               "CHUNKS_SENT",       "CHUNKS_PER_TARGET",  "CHUNK_TOTAL_SIZE",     "CHUNK_MIN_SIZE",   "CHUNK_MAX_SIZE",
               "CHUNK_AVG_SIZE",    "NNZ_SENT",          "NNZ_PER_TARGET",     "ROWS_SENT",            "ROWS_PER_TARGET",  "LAYER_END_TIME",   "LAYER_DURATION"] 

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    # Constructor
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
    def __init__(self, worker_id=0, metrics_loc=""):
        self.worker_id = worker_id
        self.metrics_loc = metrics_loc
        self.data = np.zeros([LayerStats.LS_NUM_LAYERS,LayerStats.LS_NUM_METRICS],dtype=np.float32)
        self.s3 = boto3.client('s3', region_name='eu-west-1')

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    # Private Methods
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    def _metric_increment(self,layer,metric):
        if metric in [LayerStats.LS_MSG_NUM, LayerStats.LS_PUBLISH_NUM, LayerStats.LS_CHUNKS_SENT, LayerStats.LS_NNZ_SENT, LayerStats.LS_ROWS_SENT ]:
            self.data[layer,metric] += 1
        else:
            print("Layerstats : _metric_increment >> Invalid metric passed - ", str(metric), flush=True)

    def _metric_add(self,layer,metric,value):
        if metric in [LayerStats.LS_MSG_TOTAL_SIZE, LayerStats.LS_PUBLISH_TOTAL_SIZE, LayerStats.LS_CHUNK_TOTAL_SIZE, 
                      LayerStats.LS_CHUNKS_SENT, LayerStats.LS_NNZ_SENT, LayerStats.LS_ROWS_SENT, LayerStats.LS_MSG_NUM ]:
            self.data[layer,metric] = self.data[layer,metric] + value
        else:
            print("Layerstats : _metric_add >> Invalid metric passed - ", str(metric), flush=True)

    def _metric_max(self,layer,metric,value):
        if metric in [LayerStats.LS_MSG_MAX_SIZE, LayerStats.LS_PUBLISH_MAX_SIZE, LayerStats.LS_CHUNK_MAX_SIZE]:
            if value > self.data[layer,metric]:
                self.data[layer,metric] = value
        else:
            print("Layerstats : _metric_max >> Invalid metric passed - ", str(metric), flush=True)            

    def _metric_min(self,layer,metric,value):       
        if metric in [LayerStats.LS_MSG_MIN_SIZE, LayerStats.LS_PUBLISH_MIN_SIZE, LayerStats.LS_CHUNK_MIN_SIZE]:        
            if value < self.data[layer,metric] or self.data[layer,metric] == 0:
                self.data[layer,metric] = value
        else:
            print("Layerstats : _metric_min >> Invalid metric passed - ", str(metric), flush=True)            
        
    def _metric_avg(self,layer,metric):
        if metric == LayerStats.LS_MSG_AVG_SIZE:
            self.data[layer,metric] = self.data[layer, LayerStats.LS_MSG_TOTAL_SIZE] / LayerStats.one_if_zero(self.data[layer, LayerStats.LS_MSG_NUM])
        elif metric == LayerStats.LS_PUBLISH_AVG_SIZE:
            self.data[layer,metric] = self.data[layer,LayerStats.LS_PUBLISH_TOTAL_SIZE] / LayerStats.one_if_zero(self.data[layer,LayerStats.LS_PUBLISH_NUM])
        elif metric == LayerStats.LS_CHUNK_AVG_SIZE:
            self.data[layer,metric] = self.data[layer,LayerStats.LS_CHUNK_TOTAL_SIZE] / LayerStats.one_if_zero(self.data[layer,LayerStats.LS_CHUNKS_SENT])
        elif metric == LayerStats.LS_CHUNKS_PER_TARGET:
            self.data[layer,metric] = self.data[layer,LayerStats.LS_CHUNKS_SENT] / LayerStats.one_if_zero(self.data[layer,LayerStats.LS_NUM_TARGETS])
        elif metric == LayerStats.LS_NNZ_PER_TARGET:
            self.data[layer,metric] = self.data[layer,LayerStats.LS_NNZ_SENT] / LayerStats.one_if_zero(self.data[layer,LayerStats.LS_NUM_TARGETS])
        elif metric == LayerStats.LS_ROWS_PER_TARGET:
            self.data[layer,metric] = self.data[layer,LayerStats.LS_ROWS_SENT] / LayerStats.one_if_zero(self.data[layer,LayerStats.LS_NUM_TARGETS])
        else:
            print("Layerstats : _metric_avg >> Invalid metric passed - ", str(metric), flush=True)
    
    @staticmethod
    def one_if_zero(val):
        return 1 if val == 0 else val

    def __repr__(self,layer=None):
        rep = ""
        rep =   "LAYERSTATS : Layer-wise Metrics for Inferencing Run \n"
        rep +=  "=================================================== \n"

        np.set_printoptions(precision=2, suppress=True, linewidth=100)
        for lay in range(LayerStats.LS_NUM_LAYERS):
            rep += ("\nLAYER : " + str(lay)).ljust(31) + "\n"
            for ind, metric_str in enumerate(LayerStats.METRICS):
                if ind in LayerStats.INTEGER_METRICS:
                    val = int(self.data[lay,ind])
                else:
                    val = self.data[lay,ind]
                rep += metric_str.ljust(30) + str(val)  + "\n"

        return rep

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Public Methods
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    def record_layer_start(self,layer,targets,nnz):
        self.data[layer, LayerStats.LS_LAYER_START_TIME] = timeit.default_timer()
        self.data[layer, LayerStats.LS_NUM_TARGETS] = targets
        self.data[layer, LayerStats.LS_NNZ] = nnz      
        self.data[layer,LayerStats.LS_WORKER_ID]   = self.worker_id
        self.data[layer,LayerStats.LS_LAYER] = layer

    def record_layer_end(self,layer,printstats=False):
        self.data[layer,LayerStats.LS_LAYER_END_TIME] = timeit.default_timer()
        self._metric_avg(layer, LayerStats.LS_PUBLISH_AVG_SIZE)
        self._metric_avg(layer, LayerStats.LS_MSG_AVG_SIZE)
        self._metric_avg(layer, LayerStats.LS_CHUNK_AVG_SIZE)
        self._metric_avg(layer, LayerStats.LS_CHUNKS_PER_TARGET)        
        self._metric_avg(layer, LayerStats.LS_NNZ_PER_TARGET)        
        self._metric_avg(layer, LayerStats.LS_ROWS_PER_TARGET)  
        self.record_layer_duration(layer)              
        if printstats:
            self.print_layer(layer)

    # PUBLISHES
    def record_publish(self,layer,size):
        self._metric_increment(layer, LayerStats.LS_PUBLISH_NUM)
        self._metric_add(layer, LayerStats.LS_PUBLISH_TOTAL_SIZE,size)
        self._metric_max(layer, LayerStats.LS_PUBLISH_MAX_SIZE,size)
        self._metric_min(layer, LayerStats.LS_PUBLISH_MIN_SIZE,size)

    # MESSAGES
    def record_message(self,layer,size):
        self._metric_increment(layer, LayerStats.LS_MSG_NUM)
        self._metric_add(layer, LayerStats.LS_MSG_TOTAL_SIZE,size)
        self._metric_max(layer, LayerStats.LS_MSG_MAX_SIZE,size)
        self._metric_min(layer, LayerStats.LS_MSG_MIN_SIZE,size)

    def record_messages(self,layer,number,totsize):
        self._metric_add(layer, LayerStats.LS_MSG_NUM, number)
        self._metric_add(layer, LayerStats.LS_MSG_TOTAL_SIZE, totsize)

    # CHUNKS
    def record_chunk(self,layer,size):
        self._metric_increment(layer, LayerStats.LS_CHUNKS_SENT)
        self._metric_add(layer, LayerStats.LS_CHUNK_TOTAL_SIZE,size)
        self._metric_max(layer, LayerStats.LS_CHUNK_MAX_SIZE,size)
        self._metric_min(layer, LayerStats.LS_CHUNK_MIN_SIZE,size)

    def record_chunks(self,layer,number,totsize):
        self._metric_add(layer, LayerStats.LS_CHUNKS_SENT, number)
        self._metric_add(layer, LayerStats.LS_CHUNK_TOTAL_SIZE, totsize)

    # NNZ
    def record_nnz_sent(self,layer,number):
        self._metric_add(layer, LayerStats.LS_NNZ_SENT, number)

    # ROWS
    def record_rows_sent(self,layer,number):
        self._metric_add(layer, LayerStats.LS_ROWS_SENT, number)

    # DURATION
    def record_layer_duration(self,layer):
        self.data[layer,LayerStats.LS_LAYER_DURATION] = self.data[layer,LayerStats.LS_LAYER_END_TIME] - self.data[layer,LayerStats.LS_LAYER_START_TIME]
       

    def print_layer(self,layer,printstats=False):
        rep = ""
        rep =   "LAYERSTATS : Layer-wise Metrics for Inferencing Run \n"
        rep +=  "=================================================== \n"

        np.set_printoptions(precision=2, suppress=True, linewidth=100)
        rep += ("LAYER : " + str(layer)).ljust(30) + "\n"
        for ind, metric_str in enumerate(LayerStats.METRICS):
            if ind in LayerStats.INTEGER_METRICS:
                val = int(self.data[layer,ind])
            else:
                val = self.data[layer,ind]
            rep += metric_str.ljust(30) + str(val)  + "\n"
        print(rep,flush=True)

    def write_metrics_to_s3(self):
        
        # Extract S3 location from metrics_loc
        bucket = str(self.metrics_loc["bucket"])
        subfolder = str(self.metrics_loc["subfolder"])
        expt_id = str(self.metrics_loc["expt_id"])

        filename = subfolder + "/" + str(expt_id) + "/W" + str(self.worker_id) + "_LS.csv"
    
        csvio = StringIO()
        writer = csv.writer(csvio)
        for layer in range(LayerStats.LS_NUM_LAYERS):
            config = self.metrics_loc['filename'].split("_",1)[1]
            row = [self.metrics_loc['expt_id'], config.split(".")[0]]
            for ind in range(LayerStats.LS_NUM_METRICS):
                if ind in LayerStats.INTEGER_METRICS:
                    row.append(int(self.data[layer,ind]))
                else:
                    row.append(self.data[layer,ind])

            writer.writerow(row)
        self.s3.put_object(Body=csvio.getvalue(), ContentType='text/csv', Bucket=bucket, Key=filename) 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# END CLASS DEFINITION
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
def multiply_post_process(Z_layer):
    # Add bias to all non-zero elements
    Z_layer.data = np.where(Z_layer.data != 0, Z_layer.data+BIAS, Z_layer.data)
    
    # ReLU
    Z_layer.data = np.where(Z_layer.data < 0, 0, Z_layer.data)
    
    # Threshold with ZMAX
    Z_layer.data = np.where(Z_layer.data > ZMAX, ZMAX, Z_layer.data)
    
    return Z_layer
#############################################
def worker_invoke_children(weights_loc_in, inf_data_loc_in, connectivity_loc_in, metrics_loc_in, invoc_params_in, parent_params_in, model_params_in):
    nlevels = int(invoc_params_in["nlevels"])
    p_id = int(parent_params_in["p_id"])
    p_iter = int(parent_params_in["p_iter"])
    p_js = int(parent_params_in["p_js"])
    bfr = int(invoc_params_in["bfr"])
    
    # Case 1: Internal node, not at penultimate layer. Invoke with jumps.
    if(nlevels - level) > 1:
        
        id = int(p_id + (p_iter * p_js) + 1)
        js = math.ceil((p_js - 1) / bfr)
        
        for i in range(bfr):
            payload = {
                "data_params":
                    {
                        "weights_loc": weights_loc_in,
                        "inf_data_loc": inf_data_loc_in,
                        "connectivity_loc": connectivity_loc_in,
                        "metrics_loc": metrics_loc_in
                    },
                "invoc_params": invoc_params_in,
                "parent_params":
                    {
                        'p_iter': i,
                        'p_js': js,
                        'p_id': id,
                        'p_level': level
                    },
                "model_params": model_params_in
            }
            response = lambdaClient.invoke(
                FunctionName = SPARSE_DNN_WORKER,
                InvocationType = 'Event',
                Payload = json.dumps(payload)
            )
    ## Case 2: Penultimate layer - invoke without jumping
    elif (nlevels - level) == 1:
        
        id = int(p_id + (p_iter * p_js) + 1)
        
        for i in range(bfr):
            payload = {
                "data_params":
                    {
                        "weights_loc": weights_loc_in,
                        "inf_data_loc": inf_data_loc_in,
                        "connectivity_loc": connectivity_loc_in,
                        "metrics_loc": metrics_loc_in
                    },
                "invoc_params": invoc_params_in,
                "parent_params": 
                    {
                        'child_id': id + i + 1,
                        'p_iter': i,
                        'p_js': 0,
                        'p_id': id,
                        'p_level': level
                    },
                "model_params": model_params_in
            }
            response = lambdaClient.invoke(
                FunctionName = SPARSE_DNN_WORKER,
                InvocationType = 'Event',
                Payload = json.dumps(payload)
            )
    # Case 3: Leaf node, no invocations needed. Just get ID from payload.
    elif (nlevels - level) == 0:
        id = int(parent_params_in["child_id"])
    else:
        exit
    
    return id
#############################################
def readDNNP(weights_loc_in, model_params_in):
    weights_bucket = weights_loc_in["bucket"]
    weights_subfolder = weights_loc_in["subfolder"]
    numNeurons = int(model_params_in["numNeurons"])
    
    global W
    W = []
    
    # Construct filename from event information and ID
    weightsFile = weights_subfolder + "/nnp." + str(id)
    
    result = s3.get_object(Bucket=weights_bucket, Key=weightsFile) 
    
    # Define numpy arrays. When we find a length 2 line (with ELEMENT1 > 0), create CSR and append to W. 
    # Then re-instantiate layerWeightData = np.zeros(3, ELEMENT2). R0 = row, R1 = cols, R2 = data. 
    for line in result["Body"].read().splitlines():
        line = line.decode('utf-8')
        elements = line.split()
        
        if len(elements) == 2:
            # This means the current line is a (layerNum, nnz) pair and marks the start of a new layer.
            layerCount = int(elements[0])
            nnz = int(elements[1])
            
            if elements[0] == '0':            
                layerRowIndices = np.zeros(nnz, dtype=int)
                layerColIndices = np.zeros(nnz, dtype=int)
                layerVals = np.zeros(nnz, dtype=np.float32)
                nzCount = 0
            else:

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
def readConnectivity(connectivity_loc_in, model_params_in, invoc_params_in):
    numLayers = int(model_params_in["numLayers"])
    connectivity_bucket = connectivity_loc_in["bucket"]
    connectivity_subfolder = connectivity_loc_in["subfolder"]
    numNeurons = int(model_params_in["numNeurons"])
    total_nworkers = int(invoc_params_in["total_nworkers"])
    
    global Hsend, Hrecv
    Hsend, Hrecv = [], []
    
    for i in range(numLayers):
    
        sendRows, recvRows, targetIDs, sourceIDs, sendData, recvData = [], [], [], [], [], []
        connFile = connectivity_subfolder + "/conn." + str(i)
        result = s3.get_object(Bucket=connectivity_bucket, Key=connFile) 
        
        for line in result["Body"].read().splitlines():
            
            # elements[0] is row, elements[1] is source, elements[2+] are targets
            line = line.decode('utf-8')
            elements = line.split()
            
            if int(elements[1]) == id:
                # Iterate through remaining elements of list (loop syntax to go from 2 up to len(elements))
                for j in range(2, len(elements)):
                    # For each element index k , append elements[0] to sendRows, append elements[k] to targetIDs and 1 to sendData
                    sendRows.append(int(elements[0]))
                    targetIDs.append(int(elements[j]))
                    sendData.append(1)
                
            else:
                # Iterate through remaining elements of list (loop to go from 2 up to len(elements))
                for k in range(2, len(elements)):
                
                    # If elements[k] == id:
                    if int(elements[k]) == id:
                    
                        # Append elements[0] to recvRows, elements[1] to sourceIDs, 1 to recvData
                        recvRows.append(elements[0])
                        sourceIDs.append(elements[1])
                        recvData.append(1)
            

        layerSendCSC = csc_matrix((sendData, (sendRows,targetIDs)),shape=(numNeurons, total_nworkers))
        layerRecvCSC = csc_matrix((recvData, (recvRows,sourceIDs)),shape=(numNeurons, total_nworkers))
        
        Hsend.append(layerSendCSC)
        Hrecv.append(layerRecvCSC)
#############################################
def readInferenceData(inf_data_loc_in, model_params_in):
    inf_data_bucket = inf_data_loc_in["bucket"]
    inf_data_subfolder = inf_data_loc_in["subfolder"]
    numData = int(model_params_in["numData"])
    numNeurons = int(model_params_in["numNeurons"])
    
    print("In readInferenceData, numData = " , str(numData), " numNeurons = ", str(numNeurons))
    
    infDataRowIndices, infDataColIndices, infDataVals = [], [], []
    infDataFile = inf_data_subfolder + "/train." + str(id)
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
def sigmoid(csr):
    # Function to calculate Sigmoid for a CSR matrix
    x = csr.todense()
    z = np.exp(-x)
    sig = 1 / (1 + z)
    
    return csr_matrix(sig)
#############################################
def send_row_extractor(csr, row_selector, oLS, layer):
#===========================================================================
# Controlling function for row extraction
#===========================================================================

    # Given a csr (H[layer]) and a list of rows to send to a single target, produces a list of 
    # bytes strings to be added to the send_buffer. If the bytes string is small, a single element
    # list will be returned. If its length exceeds the max allowed size, the sub-csr will be split 
    # into chunks and their bytes strings will be returned as a list of length num_chunks.
    #
    # NOTE: row_selector is now a dictionary containing (row_id: row_nnz) pairs
    #

    output_strs = []
    
    #==============================================================================
    # Group row selectors into chunks based on nnz counts
    #==============================================================================
    total_nnz = 0
    
    # First, get total nnz for required rows
    for nnz in row_selector.values():
        total_nnz += nnz

    # LAYERSTATS
    oLS.record_rows_sent(layer, len(row_selector))
    oLS.record_nnz_sent(layer, total_nnz)
    
    # If total nnz is zero (which shouldn't happen at this point), construct null message string and exit.
    if total_nnz == 0:
        csr_zero = csr_matrix(([0],([0],[0])), shape=csr.shape)
        csr_zero_bytes_string = csr_to_bytes_string_new(csr_zero, [0], csr.shape)
        output_strs.append(csr_zero_bytes_string)
        return output_strs       
    
    # If total nnz will fit in a single message, construct and return output string
    elif total_nnz <= MAX_NNZ_PER_MESSAGE:
        bytes_string_outer, stack_csr_outer_nnz = send_row_chunker(csr, row_selector)        # Keys are the rowids

        if len(bytes_string_outer) > MAX_ALLOWED_BATCH_MESSAGE_SIZE:
            print("WARNING: Output string size ",str(len(bytes_string_outer)), " exceeds maximum allowed ", str(MAX_ALLOWED_BATCH_MESSAGE_SIZE), " : Total NNZ ",str(total_nnz) , " - going to multichunking.")
        else:    
            output_strs.append(bytes_string_outer)
            return output_strs
    
    # If reached this point, need to subdivide the rows into chunks based on nnz sizes
    chunks = []
    chunk = {}     
    remove_list = []
    cumulative_nnz = 0
    row_selector_temp = row_selector.copy()
    
    # Repeatedly loop through the row selector, filling chunks and removing them from the list until it is empty
    while len(row_selector_temp) > 0:

        for row_id, row_nnz in row_selector_temp.items():

            # If a row exceeds the MAX we have catered for (based on previous experiments), abort
            if row_nnz > MAX_NNZ_PER_ROW:
                print("PROBLEM : FOR ROW ", str(row_id), " NNZ ", str(row_nnz), " EXCEEDS MAXIMUM CATERED FOR - ", str(MAX_NNZ_PER_ROW), flush=True)
                sys.exit(1)

            # If a row (by itself) exceeds the cap for a message, put it in its own chunk
            elif row_nnz > MAX_NNZ_PER_MESSAGE:
                chunk[row_id] = row_nnz
                cumulative_nnz += row_nnz
                remove_list.append(row_id)
                break   

            # If there's room in the current chunk, add the current row to it
            elif ( (cumulative_nnz + row_nnz) <= MAX_NNZ_PER_MESSAGE) and (row_nnz > 0):
                chunk[row_id] = row_nnz
                cumulative_nnz += row_nnz
                remove_list.append(row_id)
                
        chunks.append(chunk)
        chunk = {}
        cumulative_nnz = 0
        for r in remove_list:
            del row_selector_temp[r]    
        remove_list.clear()
  
    #========================================
    # MULTIPROCESSING SECTION
    #========================================

    # Iterate through the chunks sending each to send_row_chunker_wrapper
    chunking_start_time = timeit.default_timer()
    with ThreadPoolExecutor() as executor:
     
        futures = []
        print()
        for ind, chunk in enumerate(chunks):
            futures.append( executor.submit(send_row_chunker_wrapper, csr=csr, chunk=chunk) )

        for future in as_completed(futures):
            for item in future.result():
                output_strs.append(item)

    chunking_end_time = timeit.default_timer()
    chunking_elapsed_time = chunking_end_time - chunking_start_time
    if chunking_elapsed_time > 1:
        print()
        print("CHUNKING Elapsed Time :", str(chunking_elapsed_time))
        print()
        
    
    #==============================================================================
    # END OF MULTIPROCESSING SECTION
    #==============================================================================

    return output_strs
#############################################
def send_row_chunker_wrapper(csr, chunk):
#======================================================================================
# Wrapper around send_row_chunker to allow multiprocessing via a discrete function call
#======================================================================================
    wrapper_output_strs = []    
    chunk_bytes_string, chunk_stack_csr_nnz = send_row_chunker(csr, chunk)
    total_chunk_nnz = 0

    for nnz in chunk.values():
        total_chunk_nnz += nnz

    # Cater for any of the output strings exceeding the max allowed length.
    # In this situation, remove the largest row from the chunk and process it separately.
    chunk_bytes_string_length = utf8len(chunk_bytes_string)
    
    if chunk_bytes_string_length <= MAX_ALLOWED_BATCH_MESSAGE_SIZE:
        wrapper_output_strs.append(chunk_bytes_string)
    
    while chunk_bytes_string_length > MAX_ALLOWED_BATCH_MESSAGE_SIZE:  
        overflow_chunk = {}                
        if len(chunk) == 1:
            # Nothing to be done - single row exceeds max message size
            print("PROBLEM (ABORTING) : Chunk with single row exceeds max message size : ",str(chunk_bytes_string_length),flush=True)
            os._exit(1)
        
        overflow_chunk_nnz = 0
        
        while (overflow_chunk_nnz < (total_chunk_nnz / 2)) and (len(chunk) > 1):
            k_max = list(chunk.keys())[0]
            v_max = list(chunk.values())[0]
            overflow_chunk[k_max] = v_max
            overflow_chunk_nnz += v_max
            del chunk[k_max]

        # Process the overflow chunk
        overflow_chunk_bytes_string, overflow_chunk_stack_csr_nnz = send_row_chunker(csr, overflow_chunk)
        wrapper_output_strs.append(overflow_chunk_bytes_string)
        chunk_bytes_string, chunk_stack_csr_nnz = send_row_chunker(csr, chunk)
        chunk_bytes_string_length = utf8len(chunk_bytes_string)

        if chunk_bytes_string_length < MAX_ALLOWED_BATCH_MESSAGE_SIZE:
            wrapper_output_strs.append(chunk_bytes_string)
        else:
            pass

    return wrapper_output_strs
#############################################
def send_row_chunker(csr_inner, row_selector_inner):
    #======================================================================================
    # Given a csr and a list of row indices, produces a sub-csr only
    # containing those rows, but keeping original shape.
    #======================================================================================

    csr_shape_inner = csr_inner.shape       
    stack_list = []

    for req_row in row_selector_inner:
        row = csr_inner.getrow(req_row)
        stack_list.append(row)
    
    stack_csr = vstack(stack_list)
    stack_csr.eliminate_zeros() 

    return csr_to_bytes_string_new(stack_csr, row_selector_inner, csr_shape_inner), stack_csr.getnnz()
#############################################
def replace_rows(coo_row, row_selector):
    #======================================================================================
    # Given a coo.row np array and a list of desired row indices, 
    # efficiently replaces values in coo_row with corresponding values
    # from row_selector.
    #====================================================================================== 
    num_rows = len(row_selector)
    rows_before = np.arange(num_rows, dtype=int)
    rows_after = np.array(list(row_selector.keys()))

    rows_out = np.empty(num_rows + 1, dtype=int)
    rows_out[rows_before] = rows_after
    return rows_out[coo_row]
#############################################
def csr_to_bytes_string_new(csr, row_selector, csr_shape):
    #=========================================================================================
    # Given a CSR matrix, list of required row indices and shape, converts to a
    # bytes string which can be appended to output_strs (and subsequently 
    # send_buffer).
    #=========================================================================================  
    coo = csr.tocoo()
    csr_nnz = csr.getnnz()
    coo.row = replace_rows(coo.row, row_selector)
   
    if USE_COMPRESSION == 1:
        # Need one column in output_np per nnz in CSR.
        if csr_nnz > 0:
            output_np = np.zeros((3,csr_nnz),dtype=np.float32)
            output_np[0,:] = coo.data
            output_np[1,:] = coo.row
            output_np[2,:] = coo.col
        # If no nonzeros, make minimal np zeros array to send zero CSR.
        else:
            output_np = np.zeros((3,1),dtype=np.float32)
            output_np[0,0] = 0
            output_np[1,0] = 0
            output_np[2,0] = 0
        
        np.set_printoptions(threshold=np.inf)
        compression_start_time = timeit.default_timer()
        
        output_np_comp = zlib.compress(output_np, level=COMPRESSION_LEVEL)
        final_str = str(output_np_comp)      
        process_timer(METRIC_ELAPSED_TIME_COMPRESSION, compression_start_time)

        return final_str
    
    else:
        print("In csr_to_bytes_string_new, sending a coo string, nnz of csr = ", str(csr.getnnz()),flush=True)
        d_str = np.array2string(coo.data, max_line_width=MAX_MSG_LINE_WIDTH, threshold=np.inf)
        r_str = np.array2string(coo.row, max_line_width=MAX_MSG_LINE_WIDTH, threshold=np.inf)
        c_str = np.array2string(coo.col, max_line_width=MAX_MSG_LINE_WIDTH, threshold=np.inf)
        shp_str = str(csr_shape)
        
        return d_str + "|" + r_str + "|" + c_str + "|" + shp_str + "|"
#############################################
def process_timer(metric, start_timer):
    end_timer = timeit.default_timer()
    duration = end_timer - start_timer
    update_stats(metric, duration)
#############################################
def debug_timer(reference_time, message):
    pass
    # current_time = timeit.default_timer()
    # print("[DEBUG TIMER] " , message , "            Elapsed: ", str(current_time - reference_time))
#############################################
def db_write_norow_targets(dyn, layer, targets):
  
    table = dyn.Table(DYNAMODB_TABLE_NAME)

    try:
        with table.batch_writer() as batch:

            for target in targets:
                layertarget = "L" + str(layer) + "T" + str(target)
                batch.put_item(Item={"LayerTarget": layertarget, "Source": id })
                
            update_stats(METRIC_DYNAMODB_ITEMS_WRITTEN, len(targets))

    except ClientError as err:
        print("Failure in DynamoDB batch_write!r")
        print(err)
        raise
#############################################
def db_read_norow_sources(dyn, layer, sources):

    # Retrieve all items (rows) for this layer, for this worker
    layertarget = "L" + str(layer) + "T" + str(id)

    table = dyn.Table(DYNAMODB_TABLE_NAME)

    response = table.query(
    KeyConditionExpression=Key('LayerTarget').eq(layertarget)
    )
    
    total_item_count = 0
    unique_item_count = 0
    
    for i in response['Items']:
        src = i['Source']
        total_item_count += 1
        if src in sources:
            print("[VALIDATION: RECEIVED DYNAMO] Layer" , str(layer) , " ,received DynamoDB flag for source ", str(src))
            sources.remove(src)
            unique_item_count += 1
            
    update_stats(METRIC_DYNAMODB_TOTAL_ITEMS_READ, total_item_count)
    update_stats(METRIC_DYNAMODB_UNIQUE_ITEMS_READ, unique_item_count)

    return sources
#############################################
def SpFF(model_params_in, infDataCSR_in, sns, sqs, invoc_params_in, metrics_loc):
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
    
    global stats, stats_allworkers
    stats = np.zeros(NUM_METRICS, dtype=np.float32)
    stats_allworkers = np.zeros(NUM_METRICS, dtype=np.float32)

    # LAYERSTATS
    oLS = LayerStats(id,metrics_loc)
    
    sources, targets = [], []
    
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
    sources_targets_start_time = timeit.default_timer()
    
    # For all layers, populate sources array
    for layer in range(numLayers):
        
        hsend_layer = Hsend[layer]
        hrecv_layer = Hrecv[layer]
        
        # For current layer
        targets_layer, sources_layer = [], []
        
        # Iterate through columns in Hsend[layer]/Hrecv[layer].
        # Get column from CSC matrix. 
        # If nnz != 0, we are sending/receiving at least one row to/from this worker.
        # So, append to targets[layer]/sources[layer].
        for hsendColIndex in range(hsend_layer.get_shape()[1]):
            hsend_layer_col = hsend_layer.getcol(hsendColIndex)
            if hsend_layer_col.getnnz() != 0:
                targets_layer.append(hsendColIndex)
                
        for hrecvColIndex in range(hrecv_layer.get_shape()[1]):
            hrecv_layer_col = hrecv_layer.getcol(hrecvColIndex)
            if hrecv_layer_col.getnnz() != 0:
                sources_layer.append(hrecvColIndex)
        
        # Append lists for current layer to master lists.                
        targets.append(targets_layer)
        sources.append(sources_layer)
        
    process_timer(METRIC_ELAPSED_TIME_COMMUNICATION, sources_targets_start_time)

    # We may receive data during sync phase at end of batch.
    # This buffer captures this data for later consolidation.
    recv_buffer_safety = []
    for l in range(numLayers):
        recv_buffer_safety.append({})
        
    multichunk_dict_safety = {}
    
    #===========================================================================
    # [PHASE 2] Preparation of inference data for each batch (loop)
    #===========================================================================

    # Start looping through Inference data examples
    batchCount = 0
    for batchStart in range(0,numDataToUse,batchSize):
        
        # Adjust batchsize to cater for non-divisible batchsize with numData
        if batchCount == (numBatches - 1):
            batchSize = numDataToUse - (batchCount*batchSize)
        
        print("batchStart: ", str(batchStart))
        print("batchSize : ", str(batchSize))
        
        all_max_rows = np.zeros(batchSize,dtype=int)
        all_max_vals = np.zeros(batchSize,dtype=np.float32)
        all_max_worker_ids = np.zeros(batchSize,dtype=int)

        coordination_ids = []
        for w in range(total_nworkers):
            coordination_ids.append(w)
        
        print("**********************************")
        print("Starting inference batch: ", str(batchCount), " from example " , str(batchStart), " to ", str(batchStart+batchSize - 1))
        print("**********************************")
        
        # Create empty recv_buffer list to store (numLayers) dictionaries
        recv_buffer = recv_buffer_safety.copy()
        recv_buffer_safety = []
        for l in range(numLayers):
            recv_buffer_safety.append({})
            
        multichunk_dict = multichunk_dict_safety.copy() 
        multichunk_dict_safety = {}
        
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
            
            global MAX_NNZ_PER_MESSAGE
            if layer == 0:
                MAX_NNZ_PER_MESSAGE = 40000
            if layer == 25:
                MAX_NNZ_PER_MESSAGE = 120000
            elif layer == 35:
                MAX_NNZ_PER_MESSAGE = 180000
            
            layer_start_reference_time = timeit.default_timer()
            
            send_buffer = []
            
            if layer > 0:
                recv_buffer[layer - 1] = []
                recv_buffer_delete_end_time = timeit.default_timer()
                debug_timer(layer_start_reference_time, "Deleted recv_buffer[layer - 1]")

                W[layer - 1] = []
                w_delete_end_time = timeit.default_timer()
                debug_timer(recv_buffer_delete_end_time, "Deleted W[layer - 1]")

            #[START COMMUNICATION TIMER FOR DB WRITES AND MESSAGE SENDS]
            send_messages_start_time = timeit.default_timer()

            # Once per layer, build an NP array holding nnz count for each row in H[layer]
            layer_rownnz_counts = H.getnnz(axis = 1)

            # LAYERSTATS
            oLS.record_layer_start(layer=layer, targets=len(targets[layer]), nnz=H.getnnz())
 
            max_vals_layer = H.max(axis=0)  
            print("max_vals_layer nnz                      : ", max_vals_layer.getnnz())
            
            # Once per layer, build a list of dictionaries, each dict containing the row indices/NNZs required by each target
            send_row_indices_dicts = []
            for target in targets[layer]:
                row_list = Hsend[layer].getcol(target).indices.tolist()                                              # LIST OF ROW NUMBERS FOR 1 TARGET
                row_dict = {}
                for r in row_list:
                    row_dict[r] = layer_rownnz_counts[r]                                                            # A DICT CONTAINING {ROW,NNZS IN ROW}
                    update_stats(METRIC_MAX_ROW_NNZ,row_dict[r],"max")                                              # REMOVE THIS WHEN METRIC FINALISED
                send_row_indices_dicts.append( dict(sorted(row_dict.items(),key=itemgetter(1), reverse=True)) )     # SORTS row_dict IN DESCENDING ORDER OF NUM NNZs            

            # Now remove from the dicts all rows where nnz = 0    
            for dic in send_row_indices_dicts:
                items_to_remove = []
                for key, value in dic.items():
                    if value == 0:
                        items_to_remove.append(key)
                for i in items_to_remove:
                    del dic[i]

            # Now check for empty dicts. Create a new list of targets with no row indices to send, and a dictionary of targets with row indices to be retrieved
            targets_norows = []
            targets_rows = {}      
            for ind, dic in enumerate(send_row_indices_dicts):
                if len(dic) == 0:
                    targets_norows.append(targets[layer][ind])
                else:
                    targets_rows[ targets[layer][ind] ] = dic
            
            send_row_indices_prep_end_timer = timeit.default_timer()
            debug_timer(send_messages_start_time, "Prepared send_row_indices")

            # Call function to store the targets with no rows to the DynamoDB database
            if len(targets_norows) > 0:
                db_write_norow_targets(dynamodb, layer, targets_norows)

            # Carry out row extractions
            for target, dic in targets_rows.items():
                send_row_extractor_start_time = timeit.default_timer()
                send_buffer.append(send_row_extractor(H, dic, oLS, layer))
                send_row_extractor_end_time = timeit.default_timer()
                send_row_extractor_elapsed_time = send_row_extractor_end_time - send_row_extractor_start_time                
                if send_row_extractor_elapsed_time > 1:
                    print("[SEND ROW EXTRACTOR] ID: ", str(id), ", TARGET: ", str(target), " , Send Row Extractor time: ", str(send_row_extractor_elapsed_time))

            send_row_extractor_end_time = timeit.default_timer()
            debug_timer(send_row_indices_prep_end_timer, "Did send_row_extractor")
            
            # Prepare and send SNS/SQS message batches
            prepare_message_batches(list(targets_rows.keys()), send_buffer, sns, layer, oLS)
            
            prepare_message_batches_end_time = timeit.default_timer()
            debug_timer(send_row_extractor_end_time, "Sent messages")
                
            # [END COMMUNICATION TIMER FOR DB WRITES AND MESSAGE SENDS]
            process_timer(METRIC_ELAPSED_TIME_COMMUNICATION, send_messages_start_time)

            # [START COMPUTATION TIMER]
            first_multiply_start_time = timeit.default_timer()

            Z = (W[layer] * H)
            
            # [END COMPUTATION TIMER]
            process_timer(METRIC_ELAPSED_TIME_COMPUTATION, first_multiply_start_time)
            
            first_multiply_end_time = timeit.default_timer()
            debug_timer(prepare_message_batches_end_time, "Did first multiply")
            
            # Printing set of expected rows in current layer
            validation_expected_rows = {}
            
            for source in sources[layer]:
                row_list = Hrecv[layer].getcol(source).indices.tolist()
                validation_expected_rows[source] = set(row_list)
 
            # Receive messages for current layer, parse and perform second multiplies.
            # (updating Z: Z = Z + WHrecv)
            Z = process_messages(sources[layer], Z, W[layer], sqs, recv_buffer, layer, coordination_ids, all_max_rows, all_max_vals, all_max_worker_ids, multichunk_dict, batchSize, numNeurons)
            
            process_messages_end_time = timeit.default_timer()
            debug_timer(first_multiply_end_time, "Processed messages, did second multiplies")
            
            # [START COMPUTATION TIMER]
            multiply_post_process_start_time = timeit.default_timer()
            
            # Post processing (Bias, ReLU, Threshold)
            H = multiply_post_process(Z)

            # [END COMPUTATION TIMER]
            process_timer(METRIC_ELAPSED_TIME_COMPUTATION, multiply_post_process_start_time)
            
            postproc_end_time = timeit.default_timer()
            debug_timer(process_messages_end_time, "Did multiply post processing")
            
            # LAYERSTATS
            oLS.record_layer_end(layer)

            # End of layer
            
        # End of layer loop for current inference example

        # Get max rows and max vals for current inference batch. For max_rows, we can easily identify max_rows with argmax function, it returns row 0
        # if all rows = 0. For max vals, we first call max function to give a COO matrix of max values. We then pull the column indices and data,
        # and map over the np.zeros array my_max_vals which we declared at the start. This only updates the max for columns (i.e. samples in batch) 
        # where there is a non-zero value, and leaves others as 0.
        
        print("H shape: ", str(H.shape))
        
        my_max_rows = np.ravel(H.argmax(axis=0).flatten())                              # Dense 1D array. Entries are the ROW NUMBERS of with the highest value for each col)
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
        process_timer(METRIC_ELAPSED_TIME_BATCH, inf_batch_start_time)
        
        # Sync
        synchronize_workers(coordination_ids, sns, sqs, queue_url, recv_buffer_safety, batchCount, numBatches, my_max_rows, my_max_vals, all_max_rows,all_max_vals,all_max_worker_ids, multichunk_dict_safety, batchSize, numNeurons)
        
        if (id == 0):
            # Make coo matrix from all_max rows, all_max_vals and all_max_worker_ids. Worker_ids -> cols.
            master_results_coo = coo_matrix((all_max_vals, (all_max_rows, all_max_worker_ids)) , shape=(numNeurons, batchSize))
            master_results_coo.eliminate_zeros()
            
            # Print master results
            print("*************************************************")
            print("OVERALL INFERENCE RESULTS")
            print("*************************************************")
            print()
            print("type(master_results_coo)                    : ",type(master_results_coo))
            print("master_results_coo nnz                      : ", master_results_coo.getnnz())
            print(master_results_coo)
            print()
            print("*************************************************")
            print("END OF OVERALL INFERENCE RESULTS")
            print("*************************************************")
        
        batchCount += 1

        # LAYERSTATS
        oLS.write_metrics_to_s3()
        
        ## All batches done - out of inference sample loop
############### START OF PUB/SUB FUNCTIONS ###############
def sns_client():
    sns = boto3.client('sns', region_name='eu-west-1')
    """ :type : pyboto3.sns """
    return sns
#############################################
def sqs_client():
    sqs = boto3.client('sqs', region_name='eu-west-1')
    """ :type : pyboto3.sqs """
    return sqs
#############################################
def get_queue_url(sqs):
    queue_name = QUEUE_NAME_PREFIX + str(id)
    response = sqs.get_queue_url(
        QueueName=queue_name
    )
    return response['QueueUrl']
#############################################
def get_queue_arn(url, sqs):
# Function to get ARN of SQS queue
    response = sqs.get_queue_attributes(QueueUrl = url, AttributeNames = ['QueueArn'])
    return response['Attributes']['QueueArn']
#############################################
def publish_message_batch(topic_arn, ids, messages, sns, layer, chunk_count_dict, oLS):
    # Function to publish a batch of messages.
    # Receives topic_arn, a list of IDs to send to, and a list of messages
    # Create empty lists of batch request entries/resends
    batch_request_entries = []
    batch_request_entries_resend = []
    
    batch_compilation_reference_time = timeit.default_timer()
    # Iterate through provided IDs, construct batch entry using ID + corresponding message
    for i in range(len(ids)):
        id_str = str(ids[i]) + str(i)
        num_chunks_str = str(chunk_count_dict[ids[i]])

        # LAYERSTATS
        oLS.record_message(layer, utf8len(messages[i]))

        subj_str = "Message for worker " + str(ids[i])
        attrs = {
                    'targetWorkerID':
                        {
                            'DataType': 'String', 
                            'StringValue': str(ids[i])
                        },
                    'sourceWorkerID':
                        {
                            'DataType': 'String',
                            'StringValue': str(id)
                        },
                    'messageLayer':
                        {
                            'DataType': 'String',
                            'StringValue': str(layer)
                        },
                    'coordinationFlag':
                        {
                            'DataType': 'String',
                            'StringValue': str(0)
                        },
                    'numberOfChunks':
                        {
                            'DataType': 'String',
                            'StringValue': num_chunks_str
                        }
                }
        # Append constructed entry to batch entries list
        batch_request_entries.append(
            {
                'Id': id_str,
                'Message': messages[i],
                'Subject': subj_str,
                'MessageAttributes': attrs,
                'MessageStructure': 'string'
            }    
        )

    # Publish messages    
    publishing_ref_time = timeit.default_timer()

    retry = True
    retries = 0
    while(retry and (retries < MAX_SNS_PUBLISH_RETRIES)):

        try:
            response = sns.publish_batch(
                TopicArn = topic_arn,
                PublishBatchRequestEntries = batch_request_entries
            )
            update_stats(METRIC_PUBLISH_CALLS, 1)
        except ClientError as e:
            print("[ISSUE] SNS Publish Throttled, Retry Number: ", str(retries), flush=True)
            time.sleep((2**retries * 100) / 1000)
            retries += 1
            
            if retries > MAX_SNS_PUBLISH_RETRIES:
                print("[ERROR] SNS Exponential Retry Backoff Failed - Quitting!",flush=True)
                print(e.response, flush=True)
                os._exit(1)
            
            continue

        retry = False

    # Success
    if len(response['Failed']) == 0:
        return response
    
    # If any messages failed to send, retry once to same workers
    if (len(response['Failed'])) > 0:
        print("WORKER " , str(id), " MESSAGES FAILED TO SEND",flush=True)
        resend_ids, resend_msgs = [], []
        for failed_msg in response['Failed']:
            failed_worker_id = failed_msg['Id']
            failed_msg_content = str(failed_msg['Message'])
            resend_ids.append(failed_worker_id)
            resend_msgs.append(failed_msg_content)
            
        for j in range(len(resend_ids)):
            id_str_resend = str(resend_ids[j])
            subj_str_resend = "Retry Message for worker " + str(resend_ids[j])
            num_chunks_str_resend = str(chunk_count_dict[ids[i]])

            attrs_resend = {
                        'targetWorkerID':
                            {
                                'DataType': 'String', 
                                'StringValue': str(resend_ids[j])
                            },
                        'sourceWorkerID':
                            {
                                'DataType': 'String',
                                'StringValue': str(id)
                            },
                        'messageLayer':
                            {
                                'DataType': 'String',
                                'StringValue': str(layer)
                            },
                        'coordinationFlag':
                            {
                                'DataType': 'String',
                                'StringValue': str(0)
                            },
                        'numberOfChunks':
                            {
                                'DataType': 'String',
                                'StringValue': num_chunks_str_resend
                            }
                    }
            # Append constructed entry to batch entries list
            batch_request_entries_resend.append(
                {
                    'Id': id_str_resend,
                    'Message': resend_msgs[j],
                    'Subject': subj_str_resend,
                    'MessageAttributes': attrs_resend,
                    'MessageStructure': 'string'
                }    
            )
    
    # Publish retry messages
    retry_response = sns.publish_batch(
        TopicArn = topic_arn,
        PublishBatchRequestEntries = batch_request_entries_resend
    )
    update_stats(METRIC_PUBLISH_CALLS, 1)
    
    # If any messages fail again, give up and exit program
    if (len(retry_response['Failed'])) > 0:
        print("Retry messages failed - giving up. Number of failed resend messages: " , str(len(retry_response['Failed'])),flush=True)
        sys.exit(1)
    
    return retry_response
#############################################
def utf8len(s):
# Function to calculate size in bytes of string. 
    return len(s.encode('utf-8'))
#############################################
def get_topics(sns):
    return sns.list_topics()
#############################################
def get_sparse_dnn_topic_arns(sns, use_multitopic):
    topics = get_topics(sns)['Topics']
    global topic_arn, coordination_topic_arn
    
    # Multitopic: do mod 10 for data topic, coord topic as normal
    if use_multitopic:
        my_topic_suffix = int(id % 10)
        for topic in topics:
            # Data
            if TOPIC_NAME_PREFIX in topic['TopicArn']:
                if topic['TopicArn'].endswith(str(my_topic_suffix)):
                    topic_arn = topic['TopicArn']
            # Coord
            elif topic['TopicArn'].endswith(COORDINATION_TOPIC_NAME):
                coordination_topic_arn = topic['TopicArn']
    
    # Single topic: always 0 for data topic, coord topic as normal
    else:
        for topic in topics:
            # Data
            if TOPIC_NAME_PREFIX in topic['TopicArn']:
                if topic['TopicArn'].endswith("0"):
                    topic_arn = topic['TopicArn']
            # Coord
            elif topic['TopicArn'].endswith(COORDINATION_TOPIC_NAME):
                coordination_topic_arn = topic['TopicArn']

    return topic_arn, coordination_topic_arn
#############################################
def prepare_message_batches(targets_layer, send_buffer, sns, layer, oLS):

    # #========================================
    # # MULTIPROCESSING SECTION
    # #========================================

    with ThreadPoolExecutor(SNS_PUBLISH_THREADS) as publish_executor:
        print("CPU count         : ",str(os.cpu_count()),flush=True)
        print("Thread Pool Size  : ",str(publish_executor._max_workers),flush=True)
        # Submit tasks and collect futures
        publish_futures = []

        all_targets_publish_start_time = timeit.default_timer()
    
        msg_count = 0
        total_msg_size = 0
        ids_list, msg_list = [],[]
    
        chunk_count_dict = {}
    
        for target_id, byte_string_list in zip(targets_layer, send_buffer):
            
            print("In prepare_message_batches, target_id: ", str(target_id), " , len(byte_string_list): ", str(len(byte_string_list)))
            chunk_count_dict[target_id] = len(byte_string_list)
            byte_string_lengths = []
            publishes_for_current_target = 0
    
            for bytes_str in byte_string_list:
    
                utf8len_bytes_str = utf8len(bytes_str)
                byte_string_lengths.append(utf8len_bytes_str)
    
                # LAYERSTATS
                oLS.record_chunk(layer, utf8len_bytes_str)
    
                if utf8len_bytes_str > MAX_ALLOWED_BATCH_MESSAGE_SIZE:
                    print("BIG MESSAGE - QUITTING! SIZE: ", str(utf8len_bytes_str), ", TARGET: ", str(target_id))
                    sys.exit(1)
    
                update_stats(METRIC_MAX_MESSAGE_SIZE, utf8len_bytes_str, "max")
        
                if (msg_count == 10) or ((total_msg_size + utf8len_bytes_str) >= MAX_ALLOWED_BATCH_MESSAGE_SIZE):
                    # Send what we have currently
                    update_stats(METRIC_NUM_MESSAGES_SENT, msg_count)
                    update_stats(METRIC_SIZE_MESSAGES_SENT, total_msg_size)
                    update_stats(METRIC_MAX_BATCH_MSG_SIZE, total_msg_size, "max")
                        
                    # Only publish batch if msg_count >= 1
                    if msg_count >= 1:
                        oLS.record_publish(layer, total_msg_size)
                        publish_futures.append( publish_executor.submit(publish_message_batch, topic_arn=topic_arn, ids=ids_list.copy(), messages=msg_list.copy(), sns=sns, layer=layer, chunk_count_dict=chunk_count_dict, oLS=oLS) )
                        publishes_for_current_target += 1
                        
                    # Clear lists and zero counters
                    ids_list.clear()
                    msg_list.clear()
                    
                    # Append current item pair
                    ids_list.append(target_id)
                    msg_list.append(bytes_str)
                    msg_count = 1
                    total_msg_size = utf8len(bytes_str)
                else:
                    # Append current item pair
                    ids_list.append(target_id)
                    msg_list.append(bytes_str)
                    msg_count += 1    
                    total_msg_size += utf8len_bytes_str
        
        # If ids_list/msg_list non-empty after processing whole send_buffer, send remaining contents
        if msg_count > 0:
    
            final_publish_message_batch_start_time = timeit.default_timer()
            
            oLS.record_publish(layer, total_msg_size)
            publish_futures.append( publish_executor.submit(publish_message_batch, topic_arn=topic_arn, ids=ids_list, messages=msg_list, sns=sns, layer=layer, chunk_count_dict=chunk_count_dict, oLS=oLS) )
    
            update_stats(METRIC_NUM_MESSAGES_SENT, msg_count)
            update_stats(METRIC_SIZE_MESSAGES_SENT, total_msg_size)
            update_stats(METRIC_MAX_BATCH_MSG_SIZE, total_msg_size, "max")
    
        all_targets_publish_end_time = timeit.default_timer()
        all_targets_publish_elapsed_time = all_targets_publish_end_time - all_targets_publish_start_time
        print(flush=True)
        print("PUBLISH Elapsed Time : ", str(all_targets_publish_elapsed_time),flush=True)
        print(flush=True)
    
    
        for future in as_completed(publish_futures):
            pass
    
    # End of Threadpool Context Manager

#############################################
def process_messages(sources_layer, Z_layer, W_layer, sqs, recv_buffer, layer, coordination_ids, all_max_rows, all_max_vals, all_max_worker_ids, multichunk_dict, batchSize, numNeurons):
    
    sources_list = sources_layer.copy()
    delete_entries = []
    pre_received_check = True
    
    # [START COMMUNICATION TIMER]
    receive_messages_start_time = timeit.default_timer()
    
    polling_time_elapsed = 0

    while len(sources_list) > 0:
        
        if polling_time_elapsed > POLL_TIMEOUT_DURATION:
            print("****************")
            print("EXCEEDED POLL_TIMEOUT_DURATION: EXITING.")
            print("Outstanding sources: ", str(sources_list))
            print("****************")
            sys.exit(1)
 
        # Check for any messages for this layer. First time through is checking for pre-received message
        if len(recv_buffer[layer]) > 0:
        
            for source_id in recv_buffer[layer].keys():
            
                if (source_id in sources_list):

                    if multichunk_dict.get(layer) == None:
                        sources_list.remove(source_id)
                    elif source_id not in multichunk_dict[layer]:
                        sources_list.remove(source_id)

                elif source_id not in sources_layer:
                    print("[PROBLEM] I have a message in current recv_buffer layer that shouldn't be there! Source_id: ", str(source_id))
                    print("sources_list = " , sources_list)
                    print("sources_layer = ", sources_layer)
                    print("recv_buffer[layer] keys = ", recv_buffer[layer].keys())
                    sys.exit(1)
                    
        if pre_received_check:
            pre_received_check = False

        # Check DynamoDB for any expected sources with nothing to send. Remove them from sources_list
        sources_list = db_read_norow_sources(dynamodb, layer, sources_list)
        # If nothing left in sources_list, no need for SQS Poll
        if len(sources_list) == 0:
            continue
        
        # Poll queue
        polling_start_time = timeit.default_timer()
        response = sqs.receive_message(
            QueueUrl = queue_url,
            MessageAttributeNames = ['targetWorkerID', 'sourceWorkerID', 'messageLayer', 'coordinationFlag', 'numberOfChunks'],
            MaxNumberOfMessages = 10,
            VisibilityTimeout = VISIBILITY_TIMEOUT,
            WaitTimeSeconds = WAIT_TIME_SECS
        )

        update_stats(METRIC_RECEIVE_POLL_CALLS, 1)
        
        polling_end_time = timeit.default_timer()

        # If we receive no messages, add the polling duration to a timer maintained outside the loop.
        # We want max 60s of polling with no messages received. If a message is received, reset timer.
        if "Messages" not in response:
            polling_time_elapsed += (polling_end_time - polling_start_time)
            continue
        else:
            polling_time_elapsed = 0
        
        total_size = 0
        
        # Unpack and de-string received messages, if any
        if len(response["Messages"]) > 0:
            
            # print("Number of messages received in latest poll: ", len(response["Messages"]))
            
            update_stats(METRIC_NUM_MESSAGES_RECEIVED, len(response["Messages"]))
            
            for message in response["Messages"]:
                total_size += utf8len(message['Body'])
                
                # Extract source worker ID
                source_worker_id = int(message['MessageAttributes']['sourceWorkerID']['StringValue'])
                
                # If coordination flag is set
                if int(message['MessageAttributes']['coordinationFlag']['StringValue']) == 1:
                
                    # Update per batch list of IDs to check off
                    coordination_ids.remove(source_worker_id)
                    
                    if id == 0:
                        handle_stats_string(message['Body'], source_worker_id, all_max_rows, all_max_vals, all_max_worker_ids)
                
                # Data message
                else: 
                    # Parse message attributes - layer and chunks
                    messageLayer = int(message['MessageAttributes']['messageLayer']['StringValue'])
                    num_chunks = int(message['MessageAttributes']['numberOfChunks']['StringValue'])
                    
                    # 1 chunk - easy case. Convert bytes string to csr, add to recv_buffer.
                    if num_chunks == 1:
                        decompression_start_time = timeit.default_timer()
                        recv_csr = bytes_string_to_csr_new(message['Body'], batchSize, numNeurons)
                        
                        decompression_end_time = timeit.default_timer()
                        recv_buffer[messageLayer][source_worker_id] = recv_csr
                    
                    # Multiple chunks.
                    else:
                        handle_multichunk_message(messageLayer, num_chunks, message['Body'], recv_buffer, multichunk_dict, source_worker_id, batchSize, numNeurons)

                # Create dictionary of ID and receipt handle to delete.
                delete_entries.append({
                    'Id': message['MessageId'],
                    'ReceiptHandle': message['ReceiptHandle']
                })
                
        update_stats(METRIC_SIZE_MESSAGES_RECEIVED, total_size)
        
        # Delete any messages we just successfully parsed.
        response = sqs.delete_message_batch(
            QueueUrl = queue_url,
            Entries = delete_entries
        )
        
        # Clear delete entries list
        delete_entries.clear()
       
    # [END COMMUNICATION TIMER]
    process_timer(METRIC_ELAPSED_TIME_COMMUNICATION, receive_messages_start_time)
    
    # [START COMPUTATION TIMER]
    second_multiply_start_time = timeit.default_timer()

    for recv_csr in recv_buffer[layer].values():
        recv_csr.eliminate_zeros()
        if recv_csr.getnnz() > 0:
            Z_layer += (W_layer * recv_csr)
        
    
    # [END COMPUTATION TIMER]
    process_timer(METRIC_ELAPSED_TIME_COMPUTATION, second_multiply_start_time)

    return Z_layer
#############################################
def handle_multichunk_message(messageLayer, num_chunks, msg_body, recv_buffer, multichunk_dict, source_worker_id, batchSize, numNeurons):

    # If multichunk_dict[layer] exists
    if multichunk_dict.get(messageLayer) != None:

        # Case 1: source_worker_id not in multichunk_dict[layer] (FIRST MESSAGE FROM THIS ID)
        if source_worker_id not in multichunk_dict[messageLayer]:

            # Append (num_chunks - 1) copies of sourceWorkerID to multichunk_dict[layer]
            for i in range(num_chunks-1):
                multichunk_dict[messageLayer].append(source_worker_id)
                
            # bytes string to csr
            recv_csr = bytes_string_to_csr_new(msg_body, batchSize, numNeurons)

            # add recv_csr to buffer
            recv_buffer[messageLayer][source_worker_id] = recv_csr

        # Case 2: source_worker_id is in multichunk_list[layer] (subcase: it's the last occurence!)
        else:   

            # Remove 1 occurence (.remove() method removes first occurence from list)
            multichunk_dict[messageLayer].remove(source_worker_id)
            recv_csr = bytes_string_to_csr_new(msg_body, batchSize, numNeurons)
            recv_buffer[messageLayer][source_worker_id] += recv_csr


    # If multichunk_dict[messageLayer] doesn't exist, this is the moment where we create it.
    else:
        multichunk_dict[messageLayer] = []

        # Append (num_chunks - 1) copies of sourceWorkerID to multichunk_dict[messageLayer]
        for i in range(num_chunks-1):
            multichunk_dict[messageLayer].append(source_worker_id)

        # bytes string
        recv_csr = bytes_string_to_csr_new(msg_body, batchSize, numNeurons)

        # add recv_csr to buffer
        recv_buffer[messageLayer][source_worker_id] = recv_csr
#############################################
def coo_string_to_csr(coo_str):
    coo_str = coo_str.replace("\n", "")
    coo_str = coo_str.replace("|", "\n")
    parts = coo_str.splitlines()
    
    np_for_csr = np.zeros((3,len(parts[0])))
    shp = parts[3]
    shp = shp.replace("(","")
    shp = shp.replace(")","")
    shp = shp.split(",")

    rows = int(shp[0])
    cols = int(shp[1])
    csr_components = []
    
    for i in range(len(parts)-1):
        x = parts[i]
        x = x.replace("[","")
        x = x.replace("]","")
        
        if i == 0:
            y = list(map(np.float32,x.split()))
        else:
            y = list(map(int,x.split()))
            
        csr_components.append(y)
        
    return csr_matrix((csr_components[0], (csr_components[1], csr_components[2])), shape=(rows,cols))
#############################################
def bytes_string_to_csr_new(bytes_str, batchSize, numNeurons):
    
    if USE_COMPRESSION == 1:
        decompression_start_time = timeit.default_timer()
        
        # # Eval to get from string to bytes array
        bytes_eval = eval(bytes_str)
        
        # # Decompress
        decomp = zlib.decompress(bytes_eval)
        
    
        # Handle numpy array
        back_in_np = np.frombuffer(decomp,dtype=np.float32)
        
        nnz_in_csr = back_in_np.shape[0] / 3
        back_in_np = back_in_np.reshape(3,int(nnz_in_csr))
        
        process_timer(METRIC_ELAPSED_TIME_COMPRESSION, decompression_start_time)
        
        return csr_matrix((back_in_np[0,:], (back_in_np[1,:], back_in_np[2,:])), shape=(numNeurons, batchSize))
        
    else:
        return coo_string_to_csr(bytes_str)
#############################################
def compile_stats_string(batchCount, numBatches, my_max_rows, my_max_vals):
    str_out = ""
    
    # Always output this workers max rows/vals arrays
    str_out = str_out + np.array2string(my_max_rows, separator=",", threshold=np.inf) + "|"
    str_out = str_out + np.array2string(my_max_vals, separator=",", threshold=np.inf) + "|"
    
    
    if batchCount == (numBatches-1):
        lambda_handler_end_time = timeit.default_timer()
        lambda_handler_duration = lambda_handler_end_time - lambda_handler_start_time
        update_stats(METRIC_LAMBDA_RUNTIME, lambda_handler_duration)
        
        # If on final batch, also output metrics
        for j in range(NUM_METRICS):
            str_out = str_out + str(stats[j]) + "|"
            
    str_out_comp = zlib.compress(str_out.encode(), level=COMPRESSION_LEVEL)
    np.set_printoptions(threshold=np.inf)
    final_str = str(str_out_comp)

    return final_str
#############################################
def handle_stats_string(stats_str_in, source_worker_id, all_max_rows, all_max_vals, all_max_worker_ids):
    print("Handling stats_str from source worker ID: " , str(source_worker_id))
    
    # Eval, decompress and decode received bytes string 
    bytes_arr = eval(stats_str_in)
    decompressed = zlib.decompress(bytes_arr)
    stats_str = decompressed.decode()
    
    # Replace automatically inserted newline chars with nothing, then replace
    # pipes with newline to enable splitlines.
    stats_str = stats_str.replace("\n", "")
    stats_str = stats_str.replace("|", "\n")
    stats_list = stats_str.splitlines()
    
    # Create lists of max rows/vals from elements 0/1 of stats_list.
    maxrows_list = ast.literal_eval(str(stats_list[0]))
    maxvals_list = ast.literal_eval(str(stats_list[1]))
    
    # Convert above lists to np arrays.
    maxrows = np.array(maxrows_list).astype(int)
    maxvals = np.array(maxvals_list).astype(np.float32)
    
    # Update global max rows/vals/worker IDs.
    for i in range(maxrows.size):
        if maxvals[i] > all_max_vals[i]:
            all_max_rows[i] = maxrows[i]
            all_max_vals[i] = maxvals[i]
            all_max_worker_ids[i] = source_worker_id                
    
    # If long msg (i.e. if on final sample, also output stats)
    if len(stats_list) > 2:
        for i in range(2, len(stats_list)-3):
            stats_allworkers[i-2] += np.float32(stats_list[i])
            
        # Update max message size only if current max > global max.
        if np.float32(stats_list[METRIC_MAX_MESSAGE_SIZE + 2]) > stats_allworkers[METRIC_MAX_MESSAGE_SIZE]:
            stats_allworkers[METRIC_MAX_MESSAGE_SIZE] = np.float32(stats_list[METRIC_MAX_MESSAGE_SIZE + 2])
        
        # Update max batch message size only if current max > global max.    
        if np.float32(stats_list[METRIC_MAX_BATCH_MSG_SIZE + 2]) > stats_allworkers[METRIC_MAX_BATCH_MSG_SIZE]:
            stats_allworkers[METRIC_MAX_BATCH_MSG_SIZE] = np.float32(stats_list[METRIC_MAX_BATCH_MSG_SIZE + 2])
            
        # Update max row nnz only if current max > global max.    
        if np.float32(stats_list[METRIC_MAX_ROW_NNZ + 2]) > stats_allworkers[METRIC_MAX_ROW_NNZ]:
            stats_allworkers[METRIC_MAX_ROW_NNZ] = np.float32(stats_list[METRIC_MAX_ROW_NNZ + 2])
#############################################
def synchronize_workers(coordination_ids, sns, sqs, queue_url, recv_buffer_safety, batchCount, numBatches, my_max_rows, my_max_vals, all_max_rows,all_max_vals,all_max_worker_ids, multichunk_dict_safety, batchSize, numNeurons):
    delete_entries = []
    
    # Print out coordination_ids at this point (who we're waiting for)
    print("In synchronize_workers: initial coordination_ids: ", str(coordination_ids))
    
    # Send my message
    attrs = {
                    'targetWorkerID':
                        {
                            'DataType': 'String',
                            'StringValue': str(-1)
                        },
                    'sourceWorkerID':
                        {
                            'DataType': 'String',
                            'StringValue': str(id)
                        },
                    'messageLayer':
                        {
                            'DataType': 'String',
                            'StringValue': str(-1)
                        },
                    'coordinationFlag':
                        {
                            'DataType': 'String',
                            'StringValue': str(1)
                        },
                    'numberOfChunks':
                        {
                            'DataType': 'String',
                            'StringValue': str(-1)
                        }
                }
    
    msg = compile_stats_string(batchCount, numBatches, my_max_rows, my_max_vals)
    subj = 'Coordination message from worker ' + str(id)
    
    # Publish message to coordination topic
    coord_response = sns.publish(
        TopicArn=coordination_topic_arn,
        Subject=subj,
        Message=msg,
        MessageAttributes=attrs,
        MessageStructure = 'string'
    )
    
    # While coordination_ids not empty
    while len(coordination_ids) > 0:
    
        # Poll queue for synchronisation messages
        coord_poll_response = sqs.receive_message(
            QueueUrl = queue_url,
            MessageAttributeNames = ['targetWorkerID', 'sourceWorkerID', 'messageLayer', 'coordinationFlag', 'numberOfChunks'],
            MaxNumberOfMessages = 10,
            VisibilityTimeout = VISIBILITY_TIMEOUT,
            WaitTimeSeconds = WAIT_TIME_SECS
        )

        # If no messages received, skip to next iteration
        if "Messages" not in coord_poll_response:
            continue
        
        # Unpack and de-string received messages, if any
        if len(coord_poll_response["Messages"]) > 0:
            for message in coord_poll_response["Messages"]:
                
                # Extract source worker ID from MessageAttributes
                source_worker_id = int(message['MessageAttributes']['sourceWorkerID']['StringValue'])
                
                # If coordination flag is set
                if int(message['MessageAttributes']['coordinationFlag']['StringValue']) == 1:
                
                    # Update per-batch list of IDs to check off
                    coordination_ids.remove(source_worker_id)
                    
                    # Worker 0 collates metrics from all workers to display at end of run.
                    if id == 0:
                        handle_stats_string(message['Body'], source_worker_id, all_max_rows, all_max_vals, all_max_worker_ids)
                
                # Coordination flag not set -> received a data message for next batch, need to handle.    
                else:
                    print("Problem: Received a data message in synchronize_workers! Adding to recv_buffer_safety")

                    # Parse message attributes - layer and num_chunks
                    messageLayer = int(message['MessageAttributes']['messageLayer']['StringValue'])
                    num_chunks = int(message['MessageAttributes']['numberOfChunks']['StringValue'])
                    
                    # 1 chunk - easy case. Convert bytes string to csr, add to recv_buffer.
                    if num_chunks == 1:
                        recv_csr = bytes_string_to_csr_new(message['Body'], batchSize, numNeurons)
                        recv_buffer_safety[messageLayer][source_worker_id] = recv_csr
                    # Multiple chunks.
                    else:
                        handle_multichunk_message(messageLayer, num_chunks, message['Body'], recv_buffer_safety, multichunk_dict_safety, source_worker_id, batchSize, numNeurons)
                
                # Create dictionary of ID (use msg id) and receipt handle
                delete_entries.append({
                    'Id': message['MessageId'],
                    'ReceiptHandle': message['ReceiptHandle']
                })
                
        # Delete any messages we just successfully parsed.
        response = sqs.delete_message_batch(
            QueueUrl = queue_url,
            Entries = delete_entries
        )
        
        # Clear delete entries list
        delete_entries.clear()
        
        print("Outstanding coordination_ids = ", str(coordination_ids))
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
        output_list.append(str(stats_allworkers[i]))
    
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
    connectivity_loc = event["data_params"]["connectivity_loc"]
    metrics_loc = event["data_params"]["metrics_loc"]
    invoc_params = event["invoc_params"]
    parent_params = event["parent_params"]
    model_params = event["model_params"]

    ######################### CHILD INVOCATION #########################
    global level
    level = int(parent_params["p_level"]) + 1
    
    global id
    id = worker_invoke_children(weights_loc, inf_data_loc, connectivity_loc, metrics_loc, invoc_params, parent_params, model_params)
    print("I am id: ", str(id))
    print("Invoked children")
    
    ######################### PUB/SUB SETUP #########################
    sns = sns_client()
    sqs = sqs_client()
    
    global topic_arn, coordination_topic_arn, queue_url, queue_arn
    topic_arn, coordination_topic_arn = get_sparse_dnn_topic_arns(sns, USE_MULTITOPIC)
    print("topic_arn updated to: ", str(topic_arn))
    print("coordination_topic_arn updated to: ", str(coordination_topic_arn))
    queue_url = get_queue_url(sqs)
    queue_arn = get_queue_arn(queue_url, sqs)
    print("Set up pub/sub resources")
    
    ######################### READING FROM S3 #########################
    readDNNP(weights_loc, model_params)
    print("readDNNP done")
    
    readConnectivity(connectivity_loc, model_params, invoc_params)
    print("readConnectivity done")
    
    infDataCSR = readInferenceData(inf_data_loc, model_params)
    print("readInferenceData done")
    
    ######################### INFERENCE #########################
    print("Entering SpFF")
    SpFF(model_params, infDataCSR, sns, sqs, invoc_params, metrics_loc)
    print("Inference run complete!")
    
    ######################### METRICS #########################
    if (id == 0):
        
        numWorkers = invoc_params['total_nworkers']
        numBatches = model_params['numBatches']
        batchSize = model_params['batchSize']
        numLayers = model_params['numLayers']
        
        print()
        print("*********************************************************************************************")
        print("[AVERAGE TOTALS] Inference run metric totals, averaged across all workers:")
        print()
        print("METRIC_PUBLISH_CALLS              : " + str(stats_allworkers[METRIC_PUBLISH_CALLS] / numWorkers))
        print("METRIC_NUM_MESSAGES_SENT          : " + str(stats_allworkers[METRIC_NUM_MESSAGES_SENT] / numWorkers))
        print("METRIC_SIZE_MESSAGES_SENT         : " + str(stats_allworkers[METRIC_SIZE_MESSAGES_SENT] / numWorkers))
        print("METRIC_RECEIVE_POLL_CALLS         : " + str(stats_allworkers[METRIC_RECEIVE_POLL_CALLS] / numWorkers))
        print("METRIC_NUM_MESSAGES_RECEIVED      : " + str(stats_allworkers[METRIC_NUM_MESSAGES_RECEIVED] / numWorkers))
        print("METRIC_SIZE_MESSAGES_RECEIVED     : " + str(stats_allworkers[METRIC_SIZE_MESSAGES_RECEIVED] / numWorkers))
        print("METRIC_ELAPSED_TIME_TOTAL         : " + str(stats_allworkers[METRIC_ELAPSED_TIME_BATCH] / numWorkers))
        print("METRIC_ELAPSED_TIME_COMMUNICATION : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMMUNICATION] / numWorkers))
        print("METRIC_ELAPSED_TIME_COMPRESSION   : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMPRESSION] / numWorkers))
        print("METRIC_ELAPSED_TIME_COMPUTATION   : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMPUTATION] / numWorkers))
        print("*********************************************************************************************")
        print()
        print("*********************************************************************************************")
        print("[AVERAGE TOTALS PER BATCH]:")
        print()
        print("METRIC_PUBLISH_CALLS              : " + str(stats_allworkers[METRIC_PUBLISH_CALLS] / (numWorkers*numBatches)))
        print("METRIC_NUM_MESSAGES_SENT          : " + str(stats_allworkers[METRIC_NUM_MESSAGES_SENT] / (numWorkers*numBatches)))
        print("METRIC_SIZE_MESSAGES_SENT         : " + str(stats_allworkers[METRIC_SIZE_MESSAGES_SENT] / (numWorkers*numBatches)))
        print("METRIC_RECEIVE_POLL_CALLS         : " + str(stats_allworkers[METRIC_RECEIVE_POLL_CALLS] / (numWorkers*numBatches)))
        print("METRIC_NUM_MESSAGES_RECEIVED      : " + str(stats_allworkers[METRIC_NUM_MESSAGES_RECEIVED] / (numWorkers*numBatches)))
        print("METRIC_SIZE_MESSAGES_RECEIVED     : " + str(stats_allworkers[METRIC_SIZE_MESSAGES_RECEIVED] / (numWorkers*numBatches)))
        print("METRIC_ELAPSED_TIME_BATCH         : " + str(stats_allworkers[METRIC_ELAPSED_TIME_BATCH] / (numWorkers*numBatches)))
        print("METRIC_ELAPSED_TIME_COMMUNICATION : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMMUNICATION] / (numWorkers*numBatches)))
        print("METRIC_ELAPSED_TIME_COMPRESSION   : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMPRESSION] / (numWorkers*numBatches)))
        print("METRIC_ELAPSED_TIME_COMPUTATION   : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMPUTATION] / (numWorkers*numBatches)))
        print("*********************************************************************************************")
        print()
        print("*********************************************************************************************")
        print("[AVERAGE TOTALS PER SAMPLE]:")
        print()
        print("METRIC_PUBLISH_CALLS              : " + str(stats_allworkers[METRIC_PUBLISH_CALLS] / (numWorkers*numBatches*batchSize)))
        print("METRIC_NUM_MESSAGES_SENT          : " + str(stats_allworkers[METRIC_NUM_MESSAGES_SENT] / (numWorkers*numBatches*batchSize)))
        print("METRIC_SIZE_MESSAGES_SENT         : " + str(stats_allworkers[METRIC_SIZE_MESSAGES_SENT] / (numWorkers*numBatches*batchSize)))
        print("METRIC_RECEIVE_POLL_CALLS         : " + str(stats_allworkers[METRIC_RECEIVE_POLL_CALLS] / (numWorkers*numBatches*batchSize)))
        print("METRIC_NUM_MESSAGES_RECEIVED      : " + str(stats_allworkers[METRIC_NUM_MESSAGES_RECEIVED] / (numWorkers*numBatches*batchSize)))
        print("METRIC_SIZE_MESSAGES_RECEIVED     : " + str(stats_allworkers[METRIC_SIZE_MESSAGES_RECEIVED] / (numWorkers*numBatches*batchSize)))
        print("METRIC_ELAPSED_TIME_SAMPLE        : " + str(stats_allworkers[METRIC_ELAPSED_TIME_BATCH] / (numWorkers*numBatches*batchSize)))
        print("METRIC_ELAPSED_TIME_COMMUNICATION : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMMUNICATION] / (numWorkers*numBatches*batchSize)))
        print("METRIC_ELAPSED_TIME_COMPRESSION   : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMPRESSION] / (numWorkers*numBatches*batchSize)))
        print("METRIC_ELAPSED_TIME_COMPUTATION   : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMPUTATION] / (numWorkers*numBatches*batchSize)))
        print("*********************************************************************************************")
        print()
        print("*********************************************************************************************")
        print("[MISC METRICS]")
        print()
        print("% TIME COMMUNICATION              : " + str((stats_allworkers[METRIC_ELAPSED_TIME_COMMUNICATION] / stats_allworkers[METRIC_ELAPSED_TIME_BATCH]) * 100))
        print("    % TIME COMPRESSION            : " + str((stats_allworkers[METRIC_ELAPSED_TIME_COMPRESSION] / stats_allworkers[METRIC_ELAPSED_TIME_BATCH]) * 100))
        print("% TIME COMPUTATION                : " + str((stats_allworkers[METRIC_ELAPSED_TIME_COMPUTATION] / stats_allworkers[METRIC_ELAPSED_TIME_BATCH]) * 100))
        print("AVG PUBLISH CALLS PER WKR-LAYER   : " + str(stats_allworkers[METRIC_PUBLISH_CALLS] / (numWorkers*numBatches*numLayers)))
        print("AVG MSGS RECEIVED PER POLL CALL   : " + str(stats_allworkers[METRIC_NUM_MESSAGES_RECEIVED] / stats_allworkers[METRIC_RECEIVE_POLL_CALLS]))
        print("TOTAL DATA TRANSFER SNS-SQS       : " + str(stats_allworkers[METRIC_SIZE_MESSAGES_SENT]))
        print("TOTAL DATA TRANSFER SQS-LAMBDA    : " + str(stats_allworkers[METRIC_SIZE_MESSAGES_RECEIVED]))
        print("TOTAL DATA TRANSFER VOLUME (GB)   : " + str((stats_allworkers[METRIC_SIZE_MESSAGES_SENT] + stats_allworkers[METRIC_SIZE_MESSAGES_RECEIVED]) / 1000000000))
        print("TOTAL LAMBDA RUNTIME              : " + str(stats_allworkers[METRIC_LAMBDA_RUNTIME]))
        print("DYNAMODB ITEM WRITES              : " + str(stats_allworkers[METRIC_DYNAMODB_ITEMS_WRITTEN]))
        print("TOTAL DYNAMODB ITEM READS         : " + str(stats_allworkers[METRIC_DYNAMODB_TOTAL_ITEMS_READ]))
        print("UNIQUE DYNAMODB ITEM READS        : " + str(stats_allworkers[METRIC_DYNAMODB_UNIQUE_ITEMS_READ]))
        print("MAX ROW NNZ COUNT                 : " + str(stats_allworkers[METRIC_MAX_ROW_NNZ]))        
        print("MAX MESSAGE SIZE (B)              : " + str(stats_allworkers[METRIC_MAX_MESSAGE_SIZE]))
        print("MAX MESSAGE BATCH SIZE (B)        : " + str(stats_allworkers[METRIC_MAX_BATCH_MSG_SIZE]))
        print("*********************************************************************************************")
   
        output_metrics(s3, metrics_loc, invoc_params, model_params)
        
        
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Sparse DNN Worker Completed Successfully!",
        }),
    }  