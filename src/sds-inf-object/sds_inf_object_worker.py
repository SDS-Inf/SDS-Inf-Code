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
SPARSE_DNN_WORKER = "" # Replace with ARN of deployed sds_inf_object_worker Lambda function.
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
    LS_NUM_METRICS = 24
    
    LS_WORKER_ID = 0
    LS_LAYER = 1
    LS_LAYER_START_TIME = 2
    LS_FILES_WRITTEN =  3
    LS_FILES_WRITTEN_TOTAL_SIZE = 4
    LS_FILES_WRITTEN_MAX_SIZE = 5
    LS_FILES_WRITTEN_MIN_SIZE = 6
    LS_FILES_WRITTEN_AVG_SIZE = 7
    LS_FILES_READ = 8
    LS_FILES_READ_TOTAL_SIZE = 9
    LS_FILES_READ_MAX_SIZE = 10
    LS_FILES_READ_MIN_SIZE = 11
    LS_FILES_READ_AVG_SIZE = 12
    LS_NULL_FILES_WRITTEN = 13
    LS_NULL_FILES_OBSERVED = 14
    LS_NUM_FOLDER_SCANS = 15
    LS_NUM_TARGETS = 16
    LS_NNZ = 17 # NNZ in H
    LS_NNZ_SENT = 18
    LS_NNZ_PER_TARGET = 19 # Average
    LS_ROWS_SENT = 20
    LS_ROWS_PER_TARGET = 21 # Average
    LS_LAYER_END_TIME = 22
    LS_LAYER_DURATION = 23

    CURRENT_LAYER = -1

    INTEGER_METRICS = [0,1,3,8,13,14,15,16,17,18,20]
 
    METRICS = ["WORKER_ID",  "LAYER", "LAYER_START_TIME",       "FILES_WRITTEN",       "FILES_WRITTEN_TOTAL_SIZE",   "FILES_WRITTEN_MAX_SIZE",       "FILES_WRITTEN_MIN_SIZE",    "FILES_WRITTEN_AVG_SIZE", 
               "FILES_READ",          "FILES_READ_TOTAL_SIZE",  "FILES_READ_MAX_SIZE", "FILES_READ_MIN_SIZE",        "FILES_READ_AVG_SIZE",          "NULL_FILES_WRITTEN",        "NULL_FILES_OBSERVED",
               "NUM_FOLDER_SCANS",    "NUM_TARGETS",            "NNZ",     "NNZ_SENT", "NNZ_PER_TARGET",             "ROWS_SENT", "ROWS_PER_TARGET", "LAYER_END_TIME",            "LAYER_DURATION"] 

    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    # Constructor
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
    def __init__(self, worker_id=0, metrics_loc=""):
        self.worker_id = worker_id
        self.metrics_loc = metrics_loc
        self.data = np.zeros([LayerStats.LS_NUM_LAYERS,LayerStats.LS_NUM_METRICS],dtype=np.float32)
        self.s3 = boto3.client('s3')

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    # Private Methods
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    def _metric_increment(self,layer,metric):
        if metric in [LayerStats.LS_FILES_WRITTEN, LayerStats.LS_FILES_READ, LayerStats.LS_NULL_FILES_WRITTEN, LayerStats.LS_NULL_FILES_OBSERVED, LayerStats.LS_NUM_FOLDER_SCANS,
                      LayerStats.LS_NNZ, LayerStats.LS_NNZ_SENT, LayerStats.LS_ROWS_SENT]:
            self.data[layer,metric] += 1
        else:
            print("Layerstats : _metric_increment >> Invalid metric passed - ", str(metric), flush=True)

    def _metric_add(self,layer,metric,value):
        if metric in [LayerStats.LS_FILES_WRITTEN, LayerStats.LS_FILES_WRITTEN_TOTAL_SIZE, LayerStats.LS_FILES_READ, 
                      LayerStats.LS_FILES_READ_TOTAL_SIZE, LayerStats.LS_NULL_FILES_WRITTEN, LayerStats.LS_NULL_FILES_OBSERVED, LayerStats.LS_NUM_FOLDER_SCANS,
                      LayerStats.LS_NNZ, LayerStats.LS_NNZ_SENT, LayerStats.LS_ROWS_SENT]:
            self.data[layer,metric] = self.data[layer,metric] + value
        else:
            print("Layerstats : _metric_add >> Invalid metric passed - ", str(metric), flush=True)

    def _metric_max(self,layer,metric,value):
        if metric in [LayerStats.LS_FILES_WRITTEN_MAX_SIZE, LayerStats.LS_FILES_READ_MAX_SIZE]:
            if value > self.data[layer,metric]:
                self.data[layer,metric] = value
        else:
            print("Layerstats : _metric_max >> Invalid metric passed - ", str(metric), flush=True)            

    def _metric_min(self,layer,metric,value):      
        if metric in [LayerStats.LS_FILES_WRITTEN_MIN_SIZE, LayerStats.LS_FILES_READ_MIN_SIZE]:        
            if value < self.data[layer,metric] or self.data[layer,metric] == 0:
                self.data[layer,metric] = value
        else:
            print("Layerstats : _metric_min >> Invalid metric passed - ", str(metric), flush=True)            
        
    def _metric_avg(self,layer,metric):
        if metric == LayerStats.LS_FILES_WRITTEN_AVG_SIZE:
            self.data[layer,metric] = self.data[layer, LayerStats.LS_FILES_WRITTEN_TOTAL_SIZE] / LayerStats.one_if_zero(self.data[layer, LayerStats.LS_FILES_WRITTEN])
        elif metric == LayerStats.LS_FILES_READ_AVG_SIZE:
            self.data[layer,metric] = self.data[layer,LayerStats.LS_FILES_READ_TOTAL_SIZE] / LayerStats.one_if_zero(self.data[layer,LayerStats.LS_FILES_READ])
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
        LayerStats.CURRENT_LAYER = layer

    def record_layer_end(self, printstats=False):
        self.data[LayerStats.CURRENT_LAYER,LayerStats.LS_LAYER_END_TIME] = timeit.default_timer()
        self._metric_avg(LayerStats.CURRENT_LAYER, LayerStats.LS_FILES_WRITTEN_AVG_SIZE)
        self._metric_avg(LayerStats.CURRENT_LAYER, LayerStats.LS_FILES_READ_AVG_SIZE)
        self._metric_avg(LayerStats.CURRENT_LAYER, LayerStats.LS_NNZ_PER_TARGET)        
        self._metric_avg(LayerStats.CURRENT_LAYER, LayerStats.LS_ROWS_PER_TARGET)  
        self.record_layer_duration()              
        if printstats:
            self.print_layer(LayerStats.CURRENT_LAYER)
    
    # FILES WRITTEN
    def record_file_write(self,size):
        self._metric_increment(LayerStats.CURRENT_LAYER, LayerStats.LS_FILES_WRITTEN)
        self._metric_add(LayerStats.CURRENT_LAYER, LayerStats.LS_FILES_WRITTEN_TOTAL_SIZE,size)
        self._metric_max(LayerStats.CURRENT_LAYER, LayerStats.LS_FILES_WRITTEN_MAX_SIZE,size)
        self._metric_min(LayerStats.CURRENT_LAYER, LayerStats.LS_FILES_WRITTEN_MIN_SIZE,size)

    def record_null_file_write(self):
        self._metric_increment(LayerStats.CURRENT_LAYER, LayerStats.LS_NULL_FILES_WRITTEN)

    def record_null_file_writes(self,number):
        self._metric_add(LayerStats.CURRENT_LAYER, LayerStats.LS_NULL_FILES_WRITTEN, number)        

    # FILES READ
    def record_file_read(self,size):
        self._metric_increment(LayerStats.CURRENT_LAYER, LayerStats.LS_FILES_READ)
        self._metric_add(LayerStats.CURRENT_LAYER, LayerStats.LS_FILES_READ_TOTAL_SIZE,size)
        self._metric_max(LayerStats.CURRENT_LAYER, LayerStats.LS_FILES_READ_MAX_SIZE,size)
        self._metric_min(LayerStats.CURRENT_LAYER, LayerStats.LS_FILES_READ_MIN_SIZE,size)

    def record_null_file_observed(self):
        self._metric_increment(LayerStats.CURRENT_LAYER, LayerStats.LS_NULL_FILES_OBSERVED)

    def record_null_files_observed(self,number):
        self._metric_add(LayerStats.CURRENT_LAYER, LayerStats.LS_NULL_FILES_OBSERVED, number)        

    # FOLDER SCANS
    def record_folder_scan(self):
        self._metric_increment(LayerStats.CURRENT_LAYER, LayerStats.LS_NUM_FOLDER_SCANS)

    # NNZ
    def record_nnz_sent(self,number):
        self._metric_add(LayerStats.CURRENT_LAYER, LayerStats.LS_NNZ_SENT, number)

    # ROWS
    def record_rows_sent(self,number):
        self._metric_add(LayerStats.CURRENT_LAYER, LayerStats.LS_ROWS_SENT, number)

    # DURATION
    def record_layer_duration(self):
        self.data[LayerStats.CURRENT_LAYER,LayerStats.LS_LAYER_DURATION] = self.data[LayerStats.CURRENT_LAYER,LayerStats.LS_LAYER_END_TIME] - self.data[LayerStats.CURRENT_LAYER,LayerStats.LS_LAYER_START_TIME]
       

    def print_layer(self,printstats=False):
        rep = ""
        rep =   "LAYERSTATS : Layer-wise Metrics for Inferencing Run \n"
        rep +=  "=================================================== \n"

        np.set_printoptions(precision=2, suppress=True, linewidth=100)
        rep += ("LAYER : " + str(LayerStats.CURRENT_LAYER)).ljust(30) + "\n"
        for ind, metric_str in enumerate(LayerStats.METRICS):
            if ind in LayerStats.INTEGER_METRICS:
                val = int(self.data[LayerStats.CURRENT_LAYER,ind])
            else:
                val = self.data[LayerStats.CURRENT_LAYER,ind]
            rep += metric_str.ljust(30) + str(val)  + "\n"
        print(rep,flush=True)

    def write_metrics_to_s3(self):
        
        # Extract S3 location from metrics_loc
        bucket = str(self.metrics_loc["bucket"])
        subfolder = str(self.metrics_loc["subfolder"])
        expt_id = str(self.metrics_loc["expt_id"])

        # Currently just for 1 batch
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
def get_bucket_for_id(worker_id):
    return "-" + str(worker_id % S3_BUCKET_MODULO)
#############################################
def write_csr_to_s3(target, csr, hidden_loc, prefix, filename, oLS):
    #=========================================================================================
    # Writes a single csr file to s3. 
    # First converts csr to bytes string, optionally compressing it.
    # Prefix should include layer folder and end in "/"
    #=========================================================================================  
    
    hidden_layer_bucket = hidden_loc["bucket"] + get_bucket_for_id(target)
    hidden_layer_subfolder = hidden_loc["subfolder"] + "/"

    keyname = hidden_layer_subfolder + prefix + filename

    try:
        # Upload object to s3
        body = csr_to_bytes_string(csr)
        body_len = utf8len(body)
        
        result = s3.put_object(Bucket=hidden_layer_bucket, Key=keyname, Body=body)
        res = result.get('ResponseMetadata')
        status_code = res.get('HTTPStatusCode')

        if status_code == 200:
            update_stats(METRIC_S3_NUM_WRITES,1)
            update_stats(METRIC_S3_WRITE_SIZE,body_len)
            update_stats(METRIC_MAX_FILE_SIZE, body_len, "max")
            oLS.record_file_write(body_len)
            return True
        else:
            print("[ERROR]: Failed to upload csr file to s3 -> Prefix : ", prefix, " File Name : ", filename, " ERROR CODE : ", str(status_code), flush=True)
            os._exit(1)
    
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        print("[ERROR]: Failed to upload csr file to s3 -> Prefix : ", prefix, " File Name : ", filename, " ERROR CODE : ", str(error_code), flush=True)
        raise

#############################################
def write_worker_stats_to_s3(stats, hidden_loc, prefix, filename):
    #=========================================================================================
    # Writes a single worker stats file to s3
    # Called from synchronise_workers
    # Prefix should include layer folder and end in "/"
    #=========================================================================================  
    
    hidden_layer_bucket = hidden_loc["bucket"]
    hidden_layer_subfolder = hidden_loc["subfolder"] + "/"

    keyname = hidden_layer_subfolder + prefix + filename

    try:
        # Upload object to s3
        stats_len = utf8len(stats)

        result = s3.put_object(Bucket=hidden_layer_bucket, Key=keyname, Body=stats)
        res = result.get('ResponseMetadata')
        status_code = res.get('HTTPStatusCode')

        if status_code == 200:
            update_stats(METRIC_S3_NUM_WRITES,1)
            update_stats(METRIC_S3_WRITE_SIZE, stats_len)
            update_stats(METRIC_MAX_FILE_SIZE, stats_len, "max")
            return True
        else:
            print("[ERROR]: Failed to upload stats file to s3 -> Prefix : ", prefix, " File Name : ", filename, " ERROR CODE : ", str(status_code), flush=True)
            os._exit(1)

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        print("[ERROR]: Failed to upload stats file to s3 -> Prefix : ", prefix, " File Name : ", filename, " ERROR CODE : ", str(error_code), flush=True)
        raise

#############################################
def read_csr_from_s3(hidden_loc, prefix, filename, batchSize, numNeurons, oLS):
    #=========================================================================================
    # Reads a single csr file from s3. 
    # Then converts back to csr, optionally decompressing it.
    # Prefix should include layer folder and end in "/"
    #========================================================================================= 
    
    hidden_layer_bucket = hidden_loc["bucket"] + get_bucket_for_id(id)
    hidden_layer_subfolder = hidden_loc["subfolder"] + "/"

    keyname = hidden_layer_subfolder + prefix + filename

    try:
        # Download object from s3
        obj = s3.get_object(Bucket=hidden_layer_bucket, Key=keyname)
        csr_str = obj["Body"].read().decode('utf-8') 
        csr_str_len = utf8len(csr_str)
        update_stats(METRIC_S3_NUM_READS,1)
        update_stats(METRIC_S3_READ_SIZE, csr_str_len)
        update_stats(METRIC_MAX_FILE_SIZE, csr_str_len, "max")
        oLS.record_file_read(csr_str_len)
    
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        print("[ERROR]: Failed to upload file to s3 -> Prefix : ", prefix, " File Name : ", filename, " ERROR CODE : ", str(error_code), flush=True)
        raise

    # Convert back to csr
    return bytes_string_to_csr(csr_str, batchSize, numNeurons)

#############################################
def read_worker_stats_from_s3(hidden_loc, prefix, filename):
    #=========================================================================================
    # Reads a single worker stats file from s3. 
    # Called from synchronise_workers
    # Prefix should include layer folder and end in "/"
    #=========================================================================================  
    
    hidden_layer_bucket = hidden_loc["bucket"]
    hidden_layer_subfolder = hidden_loc["subfolder"] + "/"

    keyname = hidden_layer_subfolder + prefix + filename

    try:
        # Download object from s3        
        stats_str = s3.get_object(Bucket=hidden_layer_bucket, Key=keyname)['Body'].read().decode('utf-8')
        stats_str_len = utf8len(stats_str)
        update_stats(METRIC_S3_NUM_READS,1)
        update_stats(METRIC_S3_READ_SIZE, stats_str_len)     
        update_stats(METRIC_MAX_FILE_SIZE, stats_str_len, "max")
    
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        print("[ERROR]: Failed to upload file to s3 -> Prefix : ", prefix, " File Name : ", filename, " ERROR CODE : ", str(error_code), flush=True)
        raise

    # Convert back to csr
    return stats_str

#############################################
def s3_write_norow_targets(hidden_loc, prefix, targets, oLS): 
    #=========================================================================================
    # Writes a list of files to s3, for targets with no data to send.
    # Purpose is to flag this to the recipient workers
    # Prefix should include layer folder and end in "/"
    #=========================================================================================  
    prefix_in = prefix
    
    hidden_layer_bucket = hidden_loc["bucket"]
    hidden_layer_subfolder = hidden_loc["subfolder"] + "/"

    write_count = 0

    try:
        for target in targets:
            prefix = prefix_in + str(target) + "/"
            target_source = str(target) + "_" + str(id) + ".nul"
            target_bucket = hidden_layer_bucket + get_bucket_for_id(target)
            keyname = hidden_layer_subfolder + prefix + target_source
            result = s3.put_object(Bucket=target_bucket, Key=keyname, Body="")
            res = result.get('ResponseMetadata')
            status_code = res.get('HTTPStatusCode')

            if status_code == 200:
                write_count += 1 
            else:
                print("[ERROR]: Failed to upload norow file to s3 -> Prefix : ", prefix, " File Name : ", target_source, " ERROR CODE : ", str(status_code), flush=True)
                os._exit(1)
    
        update_stats(METRIC_S3_NUM_WRITES, write_count)
        oLS.record_null_file_writes(write_count)
        return write_count

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        print("[ERROR]: Failed to upload norow file to s3 -> Prefix : ", prefix, " ERROR CODE : ", str(error_code), flush=True)
        raise

#############################################
def get_matching_s3_keys(hidden_loc_in, prefix, suffix="", sync=False, oLS=None):
    #=========================================================================================
    # Scans an s3 folder and returns a list of file keys found
    # Always used in conjunction with get_matching_s3_keys
    # param prefix: Only fetch objects whose key starts with this prefix (optional)
    # param suffix: Only fetch objects whose keys end with this prefix (optional)
    # Courtesy @alexwlchan https://alexwlchan.net/2019/07/listing-s3-keys/
    #
    # NB if prefix provided it is APPENDED TO the hidden_loc subfolder
    #=========================================================================================    
    if sync:
        hidden_layer_bucket = hidden_loc_in["bucket"]
    else:
        hidden_layer_bucket = hidden_loc_in["bucket"] + get_bucket_for_id(id)
    
    hidden_layer_subfolder = hidden_loc_in["subfolder"] + "/"   

    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {'Bucket': hidden_layer_bucket}
    matches = []

    # We can pass the prefix directly to the S3 API.  If the user has passed
    # a tuple or list of prefixes, we go through them one by one.
    if isinstance(prefix, str):
        prefixes = (prefix, )
    else:
        prefixes = prefix

    for key_prefix in prefixes:
        kwargs["Prefix"] = hidden_layer_subfolder + key_prefix

        for page in paginator.paginate(**kwargs):
            try:
                contents = page["Contents"]
            except KeyError:
                break

            for obj in contents:
                key = obj["Key"]
                if key.endswith(suffix):
                    filename = obj["Key"].split('/')[-1]
                    matches.append(filename)
        update_stats(METRIC_S3_FOLDER_SCANS,1)
        oLS.record_folder_scan()

    return matches
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
def worker_invoke_children(weights_loc_in, inf_data_loc_in, connectivity_loc_in, hidden_loc_in, metrics_loc_in, invoc_params_in, parent_params_in, model_params_in):
    nlevels = int(invoc_params_in["nlevels"])
    p_id = int(parent_params_in["p_id"])
    p_iter = int(parent_params_in["p_iter"])
    p_js = int(parent_params_in["p_js"])
    bfr = int(invoc_params_in["bfr"])
    
    # Case 1: Internal node, not at penultimate layer. Invoke with jumps.
    if(nlevels - level) > 1:
        
        # Create a few vars from parent_params for convenience
        id = int(p_id + (p_iter * p_js) + 1)
        js = math.ceil((p_js - 1) / bfr)
        
        for i in range(bfr):
            payload = {
                "data_params":
                    {
                        "weights_loc": weights_loc_in,
                        "inf_data_loc": inf_data_loc_in,
                        "connectivity_loc": connectivity_loc_in,
                        "hidden_loc": hidden_loc_in, 
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
                        "hidden_loc": hidden_loc_in,
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
        # Should never reach this point - level has exceeded nlevels
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
    
    # Get nnp.ID from s3/mnist-radixnet-weights/1024
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
def readConnectivity(connectivity_loc_in, model_params_in, invoc_params_in):
    numLayers = int(model_params_in["numLayers"])
    connectivity_bucket = connectivity_loc_in["bucket"]
    connectivity_subfolder = connectivity_loc_in["subfolder"]
    numNeurons = int(model_params_in["numNeurons"])
    total_nworkers = int(invoc_params_in["total_nworkers"])
    
    global Hsend, Hrecv
    Hsend, Hrecv = [], []
    
    for i in range(numLayers):
    
        # Instantiate lists: sendRows, recvRows, targetIDs, sourceIDs, sendData, recvData
        sendRows, recvRows, targetIDs, sourceIDs, sendData, recvData = [], [], [], [], [], []
    
        # Construct string for current layer's conn file in S3
        connFile = connectivity_subfolder + "/conn." + str(i)
        
        # Call s3.get_object to get current layer's connectivity file
        result = s3.get_object(Bucket=connectivity_bucket, Key=connFile) 
        
        # Iterate through lines in connectivity file
        for line in result["Body"].read().splitlines():
            
            # Split line into list of elements. 
            # elements[0] is row, elements[1] is source, elements[2+] are targets
            line = line.decode('utf-8')
            elements = line.split()
            
            if int(elements[1]) == id:
                # i.e. if I'm the source
        
                # Iterate through remaining elements of list (loop syntax to go from 2 up to len(elements))
                for j in range(2, len(elements)):
                    # For each element index k , append elements[0] to sendRows, append elements[k] to targetIDs and 1 to sendData
                    sendRows.append(int(elements[0]))
                    targetIDs.append(int(elements[j]))
                    sendData.append(1)
                
            # Else (i.e. I'm not source, I might be a target)
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
        
        # Append to Hsend, Hrecv
        Hsend.append(layerSendCSC)
        Hrecv.append(layerRecvCSC)
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
def send_row_extractor(target, csr, row_selector, hidden_loc, prefix, oLS):
    #===========================================================================
    # Controlling function for row extraction
    #===========================================================================

    # Given a csr (H) and a list of rows to send to a single target, extracts required rows from
    # csr into a new csr. Converts this to bytes string and writes to s3 via call to write_csr_to_s3.
    #
    # NOTE: row_selector is now a dictionary containing (row_id: row_nnz) pairs

    total_nnz = 0
    
    # First, get total nnz for required rows
    for nnz in row_selector.values():
        total_nnz += nnz
    
    # LAYERSTATS
    oLS.record_rows_sent(len(row_selector))
    oLS.record_nnz_sent(total_nnz)

    stack_list = []
    for req_row in row_selector:

        row = csr.getrow(req_row)
        stack_list.append(row)
    
    stack_csr = vstack(stack_list)
    stack_csr.eliminate_zeros() 
    stack_coo = stack_csr.tocoo()
        
    # Replace rows with row selector
    stack_coo.row = replace_rows(stack_coo.row, row_selector)

    target_csr = csr_matrix((stack_coo.data, (stack_coo.row,stack_coo.col)),shape=csr.shape)

    # Write to s3
    fname = str(target) + "_" + str(id) + ".dat"
    success = write_csr_to_s3(target, target_csr, hidden_loc, prefix, fname, oLS)

    if success:
        return 1
    else:
        return 0

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
def csr_to_bytes_string(csr):
    #=========================================================================================
    # Convert csr to bytes string
    #=========================================================================================  
    coo = csr.tocoo()
    csr_nnz = csr.getnnz()
      
    if USE_COMPRESSION == 1:
        
        # Need one column in output_np per nnz in CSR.
        if csr_nnz > 0:
            output_np = np.zeros((3,csr_nnz))
            output_np[0,:] = coo.data
            output_np[1,:] = coo.row
            output_np[2,:] = coo.col
        # If no nonzeros, make minimal np zeros array to send zero CSR.
        else:
            output_np = np.zeros((3,1))
            output_np[0,0] = 0
            output_np[1,0] = 0
            output_np[2,0] = 0

        # Compress output_np
        compression_start_time = timeit.default_timer()
        np.set_printoptions(threshold=np.inf)
        output_np_comp = zlib.compress(output_np, level=COMPRESSION_LEVEL)
        output_str = str(output_np_comp) 
        process_timer(METRIC_ELAPSED_TIME_COMPRESSION, compression_start_time)

        return output_str
    
    else:
        print("In csr_to_bytes_string, sending a coo string, nnz of csr = ", str(csr.getnnz()),flush=True)
        d_str = np.array2string(coo.data, max_line_width=MAX_MSG_LINE_WIDTH, threshold=np.inf)
        r_str = np.array2string(coo.row, max_line_width=MAX_MSG_LINE_WIDTH, threshold=np.inf)
        c_str = np.array2string(coo.col, max_line_width=MAX_MSG_LINE_WIDTH, threshold=np.inf)
        shp_str = str(csr.shape)
        
        return d_str + "|" + r_str + "|" + c_str + "|" + shp_str + "|"
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
    
    global stats, stats_allworkers
    stats = np.zeros(NUM_METRICS, dtype=np.float32)
    stats_allworkers = np.zeros(NUM_METRICS, dtype=np.float32)

    # LAYERSTATS
    oLS = LayerStats(id,metrics_loc_in)
    
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
    
    # For all layers, populate sources array
    sources_targets_start_time = timeit.default_timer()
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
    
    print("Sources for all layers: ", sources)
    print("Targets for all layers: ", targets)
                
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
            
            #[START COMMUNICATION TIMER FOR WRITES AND MESSAGE SENDS]
            send_messages_start_time = timeit.default_timer()

            # Once per layer, build an NP array holding nnz count for each row in H[layer]
            layer_rownnz_counts = H.getnnz(axis = 1)

            # LAYERSTATS
            oLS.record_layer_start(layer=layer, targets=len(targets[layer]), nnz=H.getnnz())
            
            # Print nnz for max_vals in current layer
            max_vals_layer = H.max(axis=0) 
            print("max_vals_layer nnz                      : ", max_vals_layer.getnnz())
            
            # Once per layer, build a list of dictionaries, each dict containing the row indices/NNZs required by each target
            send_row_indices_dicts = []
            for target in targets[layer]:
                row_list = Hsend[layer].getcol(target).indices.tolist()                                             # LIST OF ROW NUMBERS FOR 1 TARGET
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

            # Call function to write a ".nul" file to s3 for the targets where no rows need to be sent
            if len(targets_norows) > 0:
                prefix = str(layer) + "/"
                num_null_files = s3_write_norow_targets(hidden_loc_in, prefix, targets_norows, oLS)
                if num_null_files != len(targets_norows):
                    print("[ERROR]: Number of norow targets [", str(targets_norows), "] does not match number of null files written to s3 [", str(num_null_files), "]" )
                    sys.exit(1)

            #========================================
            # MULTIPROCESSING SECTION 1
            #========================================
                       
            # Iterate through the targets sending each to send_row_extractor, which also calls the write_csr_to_s3 function
            send_row_extractor_start_time = timeit.default_timer()
            targets_processed = 0
            with ThreadPoolExecutor() as executor:

                print("Process Pool Process Count : ",str(executor._max_workers),flush=True)
                
                # Submit tasks and collect futures           
                futures = []
                print()

                for target, dic in targets_rows.items():
                    pref = str(layer) + "/" + str(target) + "/"
                    futures.append( executor.submit(send_row_extractor, target=target, csr=H, row_selector=dic, hidden_loc=hidden_loc_in, prefix=pref, oLS=oLS) )
                
                # Process task results as they are available
                for future in as_completed(futures):
                    targets_processed += 1

            send_row_extractor_end_time = timeit.default_timer()
            send_row_extractor_elapsed_time = send_row_extractor_end_time - send_row_extractor_start_time                

            # Check all targets successfully processed
            if targets_processed != len(targets_rows):
                print("[ERROR]: Number of row targets [", str(len(targets_rows), "] does not match number of data files written to s3 [", str(targets_processed)), "]" )
                sys.exit(1)

            debug_timer(send_row_extractor_end_time, "Sent messages")
            
            # [END COMMUNICATION TIMER FOR DB WRITES AND MESSAGE SENDS]
            process_timer(METRIC_ELAPSED_TIME_COMMUNICATION, send_messages_start_time)

            #==============================================================================
            # END OF MULTIPROCESSING SECTION 1
            #==============================================================================
            
            # [START COMPUTATION TIMER]
            first_multiply_start_time = timeit.default_timer()

            Z = W[0] * H
                        
            # [END COMPUTATION TIMER]
            process_timer(METRIC_ELAPSED_TIME_COMPUTATION, first_multiply_start_time)
            
            first_multiply_end_time = timeit.default_timer()
            debug_timer(first_multiply_start_time, "Did first multiply")

            # Receive csr files for current layer, parse and perform second multiplies.
            Z = process_messages(sources[layer], Z, W[0], layer, hidden_loc_in, batchSize, numNeurons, oLS)
            
            process_messages_end_time = timeit.default_timer()
            debug_timer(first_multiply_end_time, "Processed messages, did second multiplies")
            
            # [START POST PROCESS COMPUTATION TIMER]
            multiply_post_process_start_time = timeit.default_timer()
            
            # Post processing (Bias, ReLU, Threshold)
            H = multiply_post_process(Z)

            # [END POST PROCESS COMPUTATION TIMER]
            process_timer(METRIC_ELAPSED_TIME_COMPUTATION, multiply_post_process_start_time)
            
            postproc_end_time = timeit.default_timer()
            debug_timer(process_messages_end_time, "Did multiply post processing")

            # LAYERSTATS
            # oLS.record_layer_end(True)
            oLS.record_layer_end(False)

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
        process_timer(METRIC_ELAPSED_TIME_TOTAL, inf_batch_start_time)
        
        # Synchronise workers
        sync_start_time = timeit.default_timer()
        synchronize_workers(coordination_ids, batchCount, numBatches, my_max_rows, my_max_vals, all_max_rows,all_max_vals,all_max_worker_ids, batchSize, numNeurons, layer, hidden_loc_in, oLS)
        debug_timer(sync_start_time, "Synchonise Workers")
  

        if (id == 0):
            master_results_coo = coo_matrix((all_max_vals, (all_max_rows, all_max_worker_ids)) , shape=(numNeurons, batchSize))
            master_results_coo.eliminate_zeros()

        batchCount += 1

        # LAYERSTATS
        oLS.write_metrics_to_s3()
        
        ## All batches done - out of inference sample loop

#############################################
def process_messages(sources_layer, Z_layer, W_layer, layer, hidden_loc, batchSize, numNeurons, oLS):
    
    sources_list = sources_layer.copy()
    recv_buffer_layer = []

    # [START COMMUNICATION TIMER]
    receive_messages_start_time = timeit.default_timer()

    #========================================
    # MULTIPROCESSING SECTION 2
    #========================================
    #search_prefix = str(layer) + "/" + str(id) + "_"
    search_prefix = str(layer) + "/" + str(id) + "/" + str(id) + "_"
    file_prefix = str(layer) + "/" + str(id) + "/"

    with ThreadPoolExecutor() as executor:

        print("Process Pool Process Count : ", str(executor._max_workers),flush=True)
        futures = []
        print()
        num_null_sources = 0

        # Outer loop continues until all expected files received
        while len(sources_list) > 0:
                
            # Get "directory" listing from s3 to identify relevant files for this layer/worker
            fnames = get_matching_s3_keys(hidden_loc, prefix=search_prefix, sync=False, oLS=oLS)
            for fname in fnames:
                source = int(fname.split("_")[1].split(".")[0])
                fsuff = fname.split("_")[1].split(".")[1]
                
                if source in sources_list:
                    sources_list.remove(source)

                    # Only need to read .dat files - no further processing needed for .nul files
                    if fsuff == "dat":
                        futures.append( executor.submit(read_csr_from_s3, hidden_loc=hidden_loc, prefix=file_prefix, filename=fname, batchSize=batchSize, numNeurons=numNeurons, oLS=oLS) )
                    else:
                        num_null_sources += 1
                        oLS.record_null_file_observed()
            
            # Reduce frequency of s3 list requests
            time.sleep(S3_LIST_SLEEP_TIME)
            time_check = timeit.default_timer()
            if time_check - receive_messages_start_time > S3_POLLING_TIMEOUT:
                print("[ERROR] Exceeded S3 Polling Timeout, Exiting! Worker ", str(id), " in layer ", str(layer), " with number of remaining sources: ", str(len(sources_list)))
                sys.exit(1)
        
        # At this point, have issued "read" threads for all required files. Now need to collect the incoming csrs
        sllen = len(sources_layer)
        num_csrs = 0
        for future in as_completed(futures):
            num_csrs += 1
            recv_csr = future.result()
            futures.remove(future)
            process_timer(METRIC_ELAPSED_TIME_COMMUNICATION, receive_messages_start_time)
            second_multiply_start_time = timeit.default_timer()

            recv_csr.eliminate_zeros()
            if recv_csr.getnnz() > 0:
                Z_layer += (W_layer * recv_csr)

            process_timer(METRIC_ELAPSED_TIME_COMPUTATION, second_multiply_start_time)
            receive_messages_start_time = timeit.default_timer()

        if (num_csrs + num_null_sources) != sllen:
            print("[ERROR] : Number of CSRs read from s3 [", str(num_csrs),   "] does not equal expected number [", str(sllen), "]")
            sys.exit(0)

    # [END COMMUNICATION TIMER]
    process_timer(METRIC_ELAPSED_TIME_COMMUNICATION, receive_messages_start_time)        

    #========================================
    # END OF MULTIPROCESSING SECTION 2 
    #========================================

    # [START COMPUTATION TIMER]
    second_multiply_start_time = timeit.default_timer()
    
    return Z_layer
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
def bytes_string_to_csr(bytes_str, batchSize, numNeurons):
    
    if USE_COMPRESSION == 1:
        decompression_start_time = timeit.default_timer()
        
        # Eval to get from string to bytes array
        bytes_eval = eval(bytes_str)
        
        # Decompress
        decomp = zlib.decompress(bytes_eval)
        
        # Handle numpy array
        back_in_np = np.frombuffer(decomp)
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
        
        for j in range(NUM_METRICS):
            str_out = str_out + str(stats[j]) + "|"
            
    str_out_comp = zlib.compress(str_out.encode(), level=COMPRESSION_LEVEL)
    np.set_printoptions(threshold=np.inf)
    final_str = str(str_out_comp)

    return final_str
#############################################
def handle_stats_string(stats_str_in, source_worker_id, all_max_rows, all_max_vals, all_max_worker_ids):
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
        for i in range(2, len(stats_list)-2):
            stats_allworkers[i-2] += np.float32(stats_list[i])
            
        # Update max message size only if current max > global max.
        if np.float32(stats_list[METRIC_MAX_FILE_SIZE + 2]) > stats_allworkers[METRIC_MAX_FILE_SIZE]:
            stats_allworkers[METRIC_MAX_FILE_SIZE] = np.float32(stats_list[METRIC_MAX_FILE_SIZE + 2])
                   
        # Update max row nnz only if current max > global max.    
        if np.float32(stats_list[METRIC_MAX_ROW_NNZ + 2]) > stats_allworkers[METRIC_MAX_ROW_NNZ]:
            stats_allworkers[METRIC_MAX_ROW_NNZ] = np.float32(stats_list[METRIC_MAX_ROW_NNZ + 2])

#############################################
def synchronize_workers(coordination_ids, batchCount, numBatches, my_max_rows, my_max_vals, all_max_rows, all_max_vals, all_max_worker_ids, batchSize, numNeurons, layer, hidden_loc, oLS):
    
    # Print out coordination_ids at this point (who we're waiting for)
    print("In synchronize_workers: initial coordination_ids: ", str(coordination_ids))
    
    # Generate stats string for this worker
    my_stats = compile_stats_string(batchCount, numBatches, my_max_rows, my_max_vals)

    # Write stats string to s3
    pref = "Batch_" + str(batchCount) + "/"
    fname = str(id) + ".syn"

    success = write_worker_stats_to_s3(my_stats, hidden_loc, pref, fname)
    if not success:
        print("[ERROR]: In synchonise_workers - Failed to write stats to s3 for Worker ", str(id))
        sys.exit(1)

    sync_start_time = timeit.default_timer()

    # Start a theadpool context
    with ThreadPoolExecutor() as executor:

        print("Process Pool Process Count : ", str(executor._max_workers),flush=True)
        futures = []
        print()

        # Loop until all coordination files have been written by workers
        search_prefix = "Batch_" + str(batchCount) + "/"
        search_suffix = ".syn"
        while len(coordination_ids) > 0:
        
            fnames = get_matching_s3_keys(hidden_loc, prefix=search_prefix, suffix=search_suffix, sync=True, oLS=oLS)
            for fname in fnames:
                source = int(fname.split(".")[0])
                if source in coordination_ids:
                    coordination_ids.remove(source)

                    # Worker 0 also needs to read and process the file
                    if id == 0:
                        futures.append( executor.submit(read_worker_stats_from_s3, hidden_loc=hidden_loc, prefix=search_prefix, filename=fname) )
                        
            print("Outstanding coordination_ids = ", str(coordination_ids))

            # Reduce frequency of s3 list requests
            time.sleep(S3_LIST_SLEEP_TIME)
            time_check = timeit.default_timer()
            if time_check - sync_start_time > S3_POLLING_TIMEOUT:
                print("Sync S3 Polling Timeout")
                sys.exit(1)


        for future in as_completed(futures):
            handle_stats_string(future.result(), source, all_max_rows, all_max_vals, all_max_worker_ids)
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
    hidden_loc = event["data_params"]["hidden_loc"]
    metrics_loc = event["data_params"]["metrics_loc"]
    invoc_params = event["invoc_params"]
    parent_params = event["parent_params"]
    model_params = event["model_params"]
    
    ######################### CHILD INVOCATION #########################
    global level
    level = int(parent_params["p_level"]) + 1
    
    global id
    id = worker_invoke_children(weights_loc, inf_data_loc, connectivity_loc, hidden_loc, metrics_loc, invoc_params, parent_params, model_params)
    print("I am id: ", str(id))
    print("Invoked children")
    
    ######################### READING FROM S3 #########################
    readDNNP(weights_loc, model_params)
    print("readDNNP done")
    
    readConnectivity(connectivity_loc, model_params, invoc_params)
    print("readConnectivity done")
    
    infDataCSR = readInferenceData(inf_data_loc, model_params)
    print("readInferenceData done")
    
    ######################### INFERENCE #########################
    print("Entering SpFF")
    SpFF(model_params, infDataCSR, invoc_params, hidden_loc, metrics_loc)
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
        print("METRIC_S3_NUM_WRITES              : " + str(stats_allworkers[METRIC_S3_NUM_WRITES] / numWorkers))
        print("METRIC_S3_NUM_READS               : " + str(stats_allworkers[METRIC_S3_NUM_READS] / numWorkers))
        print("METRIC_S3_WRITE_SIZE              : " + str(stats_allworkers[METRIC_S3_WRITE_SIZE] / numWorkers))
        print("METRIC_S3_READ_SIZE               : " + str(stats_allworkers[METRIC_S3_READ_SIZE] / numWorkers))
        print("METRIC_S3_FOLDER SCANS            : " + str(stats_allworkers[METRIC_S3_FOLDER_SCANS] / numWorkers))
        print("METRIC_ELAPSED_TIME_TOTAL         : " + str(stats_allworkers[METRIC_ELAPSED_TIME_TOTAL] / numWorkers))
        print("METRIC_ELAPSED_TIME_COMMUNICATION : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMMUNICATION] / numWorkers))
        print("METRIC_ELAPSED_TIME_COMPUTATION   : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMPUTATION] / numWorkers))
        print("*********************************************************************************************")
        print()
        print("*********************************************************************************************")
        print("[AVERAGE TOTALS PER BATCH]:")
        print()
        print("METRIC_S3_NUM_WRITES              : " + str(stats_allworkers[METRIC_S3_NUM_WRITES] / (numWorkers*numBatches)))
        print("METRIC_S3_NUM_READS               : " + str(stats_allworkers[METRIC_S3_NUM_READS] / (numWorkers*numBatches)))
        print("METRIC_S3_WRITE_SIZE              : " + str(stats_allworkers[METRIC_S3_WRITE_SIZE] / (numWorkers*numBatches)))
        print("METRIC_S3_READ_SIZE               : " + str(stats_allworkers[METRIC_S3_READ_SIZE] / (numWorkers*numBatches)))
        print("METRIC_S3_FOLDER_SCANS            : " + str(stats_allworkers[METRIC_S3_FOLDER_SCANS] / (numWorkers*numBatches)))
        print("METRIC_ELAPSED_TIME_TOTAL         : " + str(stats_allworkers[METRIC_ELAPSED_TIME_TOTAL] / (numWorkers*numBatches)))
        print("METRIC_ELAPSED_TIME_COMMUNICATION : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMMUNICATION] / (numWorkers*numBatches)))
        print("METRIC_ELAPSED_TIME_COMPUTATION   : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMPUTATION] / (numWorkers*numBatches)))
        print("*********************************************************************************************")
        print()
        print("*********************************************************************************************")
        print("[AVERAGE TOTALS PER SAMPLE]:")
        print()
        print("METRIC_S3_NUM_WRITES              : " + str(stats_allworkers[METRIC_S3_NUM_WRITES] / (numWorkers*numBatches*batchSize)))
        print("METRIC_S3_NUM_READS               : " + str(stats_allworkers[METRIC_S3_NUM_READS] / (numWorkers*numBatches*batchSize)))
        print("METRIC_S3_WRITE_SIZE              : " + str(stats_allworkers[METRIC_S3_WRITE_SIZE] / (numWorkers*numBatches*batchSize)))
        print("METRIC_S3_READ_SIZE               : " + str(stats_allworkers[METRIC_S3_READ_SIZE] / (numWorkers*numBatches*batchSize)))
        print("METRIC_S3_FOLDER_SCANS            : " + str(stats_allworkers[METRIC_S3_FOLDER_SCANS] / (numWorkers*numBatches*batchSize)))
        print("METRIC_ELAPSED_TIME_TOTAL         : " + str(stats_allworkers[METRIC_ELAPSED_TIME_TOTAL] / (numWorkers*numBatches*batchSize)))
        print("METRIC_ELAPSED_TIME_COMMUNICATION : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMMUNICATION] / (numWorkers*numBatches*batchSize)))
        print("METRIC_ELAPSED_TIME_COMPUTATION   : " + str(stats_allworkers[METRIC_ELAPSED_TIME_COMPUTATION] / (numWorkers*numBatches*batchSize)))
        print("*********************************************************************************************")
        print()
        print("*********************************************************************************************")
        print("[MISC METRICS]")
        print()
        print("% TIME COMMUNICATION              : " + str((stats_allworkers[METRIC_ELAPSED_TIME_COMMUNICATION] / stats_allworkers[METRIC_ELAPSED_TIME_TOTAL]) * 100))
        print("% TIME COMPUTATION                : " + str((stats_allworkers[METRIC_ELAPSED_TIME_COMPUTATION] / stats_allworkers[METRIC_ELAPSED_TIME_TOTAL]) * 100))
        print("AVG WRITES PER WKR-LAYER          : " + str(stats_allworkers[METRIC_S3_NUM_WRITES] / (numWorkers*numBatches*numLayers)))
        print("TOTAL DATA TRANSFER VOLUME (GB)   : " + str((stats_allworkers[METRIC_S3_WRITE_SIZE] + stats_allworkers[METRIC_S3_READ_SIZE]) / 1000000000))
        print("TOTAL LAMBDA RUNTIME              : " + str(stats_allworkers[METRIC_LAMBDA_RUNTIME]))
        print("MAX MESSAGE FILE SIZE (B)         : " + str(stats_allworkers[METRIC_MAX_FILE_SIZE]))
        print("TOTAL S3 FOLDER SCANS ALL WORKERS : " + str(stats_allworkers[METRIC_S3_FOLDER_SCANS]))
        print("*********************************************************************************************")
   
        output_metrics(s3, metrics_loc, invoc_params, model_params)
        

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Sparse DNN Worker Completed Successfully!",
        }),
    }  