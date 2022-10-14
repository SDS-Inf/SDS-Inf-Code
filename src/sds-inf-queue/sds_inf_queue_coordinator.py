import json
import boto3
from io import BytesIO
########## CONSTANTS ##########
SPARSE_DNN_WORKER = "" # Replace with ARN of sds-inf-queue-worker Lambda function once deployed.
########## CLIENTS ##########
lambdaClient = boto3.client('lambda', region_name='eu-west-1')
########## KEY VARIABLES ##########
level = 0
id = -1
#################### FUNCTION DEFINITIONS ####################
def parse_coord_event(event):
	weights_loc = event['data_params']['weights_loc']
	inf_data_loc = event['data_params']['inf_data_loc']
	connectivity_loc = event['data_params']['connectivity_loc']
	metrics_loc = event['data_params']['metrics_loc']
	invoc_params = event['invoc_params']
	model_params = event['model_params']

	return weights_loc, inf_data_loc, connectivity_loc, metrics_loc, invoc_params, model_params
#############################################
def coord_invoke_children(weights_loc_in, inf_data_loc_in, connectivity_loc_in, metrics_loc_in, invoc_params_in, model_params_in, js_in, bfr_in):
    bfr = bfr_in
    nlevels = int(invoc_params_in["nlevels"])
    js = js_in
    global level
    
    print("In coord_invoke_children. nlevels: ", str(nlevels), " , bfr: ", str(bfr))
    if (nlevels == 1):
        print("Coordinator invoking children - 1 layer!")
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
                        "child_id": i,
                        "p_iter": i,
                        "p_js": js,
                        "p_id": id,
                        "p_level": level
                    },
                "model_params": model_params_in
            }
            response = lambdaClient.invoke(
                FunctionName = SPARSE_DNN_WORKER,
                InvocationType = 'Event',
                Payload = json.dumps(payload)
            )
            print(response)
    else:
        print("Coordinator invoking children - >1 layer")
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
                        "p_iter": i,
                        "p_js": js,
                        "p_id": id,
                        "p_level": level
                    },
                "model_params": model_params_in
            }
            print("About to invoke iteration ", str(i))
            response = lambdaClient.invoke(
                FunctionName = SPARSE_DNN_WORKER,
                InvocationType = 'Event',
                Payload = json.dumps(payload)
            )
            print(response)
#############################################
def lambda_handler(event, context):
    
    weights_loc = event["data_params"]["weights_loc"]
    print("Weights_loc in coordinator: ", weights_loc)
    inf_data_loc = event["data_params"]["inf_data_loc"]
    connectivity_loc = event["data_params"]["connectivity_loc"]
    metrics_loc = event["data_params"]["metrics_loc"]
    invoc_params = event["invoc_params"]
    model_params = event["model_params"]
    
    bfr = int(invoc_params["bfr"])
    total_nworkers = int(invoc_params["total_nworkers"])

    # Calculate coordinator jump size
    js = total_nworkers / bfr
    
    coord_invoke_children(weights_loc, inf_data_loc, connectivity_loc, metrics_loc, invoc_params, model_params, js, bfr)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Coordinator returning!"
        }),
    }
