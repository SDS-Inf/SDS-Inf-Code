# SDS-Inf
SDS-Inf (Serverless Distributed Sparse Inference) 

# Prerequisites
To run SDS-Inf, you will require:
- An AWS account
- The AWS CLI (for creation of communication resources)
- Access to the AWS SDK Boto3 for Python - automatically included if using AWS Cloud9

# Instructions
1. Create IAM roles with the following policies:
    - Role for all workers:
        - A policy to enable Lambda to write to S3
        - A policy to enable a given Lambda function to invoke other Lambda functions
        - SQSFullAccess
        - S3FullAccess
        - CloudWatchFullAccess
        - DynamoDBFullAccess
        - AWSLambdaBasicExecutionRole
        - AmazonSNSFullAccess
    - Role for all coordinators:
        - A policy to enable Lambda to write to S3
        - A policy to enable a given Lambda function to invoke other Lambda functions
        - S3FullAccess
        - AWSLambdaBasicExecutionRole

    Note that in a production environment, you should instead use more restrictive access policies.

2. Set up an IDE with AWS SDK: AWS Cloud9 is recommended.

3. Set up communication resources by running create_dynamodb_table.py and create_pub_sub_resources.py, both of which are in /scripts.

4. Create Lambda SAM applications with Python 3.8 for SDS-Inf-Queue worker/coordinator, SDS-Inf-Object worker/coordinator, and SDS-Inf-Serial worker.

5. Copy the relevant Python code into the Lambda application

6. Set up the AWS SAM template to refer to your code location as appropriate, and include the ARN of the IAM role. An example template.yaml is included in /resources.

7. Deploy the worker Lambda function (s). 

8. Update the constant SPARSE_DNN_WORKER with the ARN of the deployed worker function, in both the relevant worker and coordinator. 

9. Deploy coordinator and re-deploy worker.

10. Configure Lambda function memory, max runtime, max concurrency as desired. Under "Asynchronous Invocation", we recommend:

    - Maximum age of event: 1min
    - Retry attempts: 0
    - Dead-letter queue service: None

11. Create S3 buckets for inference, weights and connectivity information. The benchmark data we use in our experiments can be accessed at https://graphchallenge.mit.edu/data-sets. 

12. To begin SDS-Inf execution (either Queue or Object), invoke the relevant coordinator function. Example .JSON invocation payloads are given in /resources. Metric output files will be written to S3, and metrics can also be seen in the CloudWatch log of the worker functions. We provide example metrics and layerstats files in /resources (these are the median of 3 runs).



