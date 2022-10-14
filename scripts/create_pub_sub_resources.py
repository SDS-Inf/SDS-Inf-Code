# Imports
import boto3
import json
import scipy
from scipy.sparse import csr_matrix, csc_matrix
import numpy as np
import ast
import time

# Define constants and create clients
TOPIC_NAME_PREFIX = 'sparseDNNTopic'
COORDINATION_TOPIC_NAME = 'sparseDNNCoordinationTopic'
QUEUE_NAME_PREFIX = 'SMPI_WQ_DEMO'
QUEUE_ARN = ''
MAX_BATCH_MESSAGE_BYTES = 262144
NUM_DATA_TOPICS = 10

def sns_client():
    sns = boto3.client('sns', region_name='eu-west-1')
    """ :type : pyboto3.sns """
    return sns
    
def sqs_client():
    sqs = boto3.client('sqs', region_name='eu-west-1')
    """ :type : pyboto3.sqs """
    return sqs
    

# Function to create an SNS topic
def create_topic(id):
    try:
        response = sns.create_topic(
            Name=TOPIC_NAME_PREFIX + str(id)
        )
    except:
        print("Failure in create_topic")
    return response['TopicArn']

# Function to create an SNS topic
def create_topic_old():
    try:
        response = sns.create_topic(
            Name=TOPIC_NAME_PREFIX
        )
    except:
        print("Failure in create_topic")
    return response['TopicArn']
    
def create_coordination_topic():
    try:
        response = sns.create_topic(
            Name=COORDINATION_TOPIC_NAME
        )
    except:
        print("Failure in create_topic")
    return response['TopicArn']


# Function to create an SQS queue
def create_sqs_queue(id):
    queue_name = QUEUE_NAME_PREFIX + str(id)
    response =  sqs.create_queue(
        QueueName=queue_name,
        Attributes = {
            'VisibilityTimeout': '0'
        }
    )
    return response['QueueUrl']

# Function to get ARN of SQS queue
def get_queue_arn(url):
    response = sqs.get_queue_attributes(QueueUrl = url, AttributeNames = ['QueueArn'])
    return response['Attributes']['QueueArn']

# Function to subscribe a queue to a topic
def create_sqs_queue_subscription(topic_arn, queue_arn):
    response = sns.subscribe(
        TopicArn=topic_arn,
        Protocol='sqs',
        Endpoint=queue_arn
    )
    return response['SubscriptionArn']

# Function to add a filter to the subscription
def update_subscription_attributes(subscription_arn, id):
    
    policy = {
        "targetWorkerID": [str(id)]
    }
    sns.set_subscription_attributes(
        SubscriptionArn = subscription_arn,
        AttributeName = 'FilterPolicy',
        AttributeValue = json.dumps(policy)
    )
    sns.set_subscription_attributes(
        SubscriptionArn = subscription_arn,
        AttributeName = 'RawMessageDelivery',
        AttributeValue = 'true'
    )
    
    print("Added subscription filter for worker ID: " , id)
    
# Function to add a filter to the subscription
def update_subscription_attributes_coordination(subscription_arn):
    
    sns.set_subscription_attributes(
        SubscriptionArn = subscription_arn,
        AttributeName = 'RawMessageDelivery',
        AttributeValue = 'true'
    )
    
    print("Added coordinationFlag filter for worker ID: " , id)

# Function to update queue policy to allow message receives from SNS
def add_queue_iam_policy(queue_url, queue_arn, topic_arn, id):
    
    policyJson = {
      "Version": "2008-10-17",
      "Statement": [
        {
          "Sid": "Allow-SNS-SendMessage",
          "Effect": "Allow",
          "Principal": {
            "Service": "sns.amazonaws.com"
          },
          "Action": "sqs:SendMessage",
          "Resource": queue_arn,
          "Condition": {
            "ArnLike": {
                "aws:SourceArn": "arn:aws:sns:eu-west-1:570608630185:DUB*"
            }
          }
        }
      ]
    }
    print("policyJson: " ,  policyJson)
    policy = json.dumps(policyJson)
    print("policy type: ", type(policy), " policy: " ,  policy)
    
    
    #   "aws:SourceArn": "arn:aws:sns:eu-west-2:570608630185:sparseDNN*"
        # "aws:SourceArn": "arn:aws:sns:eu-west-1:*"
    
    sqs.set_queue_attributes(
        QueueUrl = queue_url,
        Attributes = {
            'Policy': policy
        }
    )


# Function for experimenting with sending messages to topic
def publish_message(topic_arn, id):
    attrs = {'targetWorkerID': {'DataType': 'String', 'StringValue': str(id)}}
    msg = 'Sending a message to worker ' + str(id)
    subj = 'Message to worker ' + str(id)
    return sns.publish(
        TopicArn=topic_arn,
        Subject=subj,
        Message=msg,
        MessageAttributes=attrs,
        MessageStructure = 'string'
    )

# Function to create multiple queues plus subscription and filter
def create_all_queues(numWorkers, coordination_topic_arn):
    # Make all topics 
    topic_arns = []
    for i in range(NUM_DATA_TOPICS): # Topic loop
        topic_arns.append(create_topic(i)) # Example name: DUBsparseDNNTopic0

    for i in range(numWorkers): # Queue loop
        queue_url = create_sqs_queue(i)
        queue_arn = get_queue_arn(queue_url)

        subscription_arn_coord = create_sqs_queue_subscription(coordination_topic_arn,queue_arn)
        update_subscription_attributes_coordination(subscription_arn_coord)
 
        # Subscribe to all topics in loop and update
        for topic_arn in topic_arns:
            subscription_arn = create_sqs_queue_subscription(topic_arn, queue_arn)
            update_subscription_attributes(subscription_arn, i)

        add_queue_iam_policy(queue_url, queue_arn, topic_arn, i)

def delete_resources_old():
    # Get topic ARN
    sns_resource = boto3.resource('sns')
    topic_delete_arn = create_topic_old()
    coordination_topic_delete_arn = create_coordination_topic()
    
    subs = sns.list_subscriptions_by_topic(TopicArn=topic_delete_arn)
    for sub in subs['Subscriptions']:
        print("Deleting subscription: " , sub)
        subClass = sns_resource.Subscription(sub['SubscriptionArn'])
        subClass.delete()
        
    subs_coord = sns.list_subscriptions_by_topic(TopicArn=coordination_topic_delete_arn)
    for sub in subs_coord['Subscriptions']:
        print("Deleting subscription: " , sub)
        subClass = sns_resource.Subscription(sub['SubscriptionArn'])
        subClass.delete()
    
    # Get queue ARNs matching name (from SQS client)
    queues = sqs.list_queues(QueueNamePrefix=QUEUE_NAME_PREFIX)
    for queueUrl in queues['QueueUrls']:
        print("Deleting queue: " , queueUrl)
        sqs.delete_queue(QueueUrl = queueUrl)
    
    print("Deleting topic: " , topic_delete_arn)
    sns.delete_topic(TopicArn = topic_delete_arn)
    sns.delete_topic(TopicArn = coordination_topic_delete_arn)


def delete_resources(numWorkers):
    sns_resource = boto3.resource('sns', region_name='eu-west-1')

    #--------------------------------------------------------------------------------------------------------
    # Get coordination topic and delete all its subscriptions, then delete coord topic
    coordination_topic_delete_arn = create_coordination_topic()

    subs_coord = sns.list_subscriptions_by_topic(TopicArn=coordination_topic_delete_arn)
    for sub in subs_coord['Subscriptions']:
        print("Deleting subscription: " , sub)
        subClass = sns_resource.Subscription(sub['SubscriptionArn'])
        subClass.delete()

    sns.delete_topic(TopicArn = coordination_topic_delete_arn)
    #--------------------------------------------------------------------------------------------------------
    # Iterate through all worker IDs, get topic, delete all subscriptions, then delete topic
    for i in range(NUM_DATA_TOPICS):
        topic_delete_arn = create_topic(i)
        
        subs = sns.list_subscriptions_by_topic(TopicArn=topic_delete_arn)
        for sub in subs['Subscriptions']:
            print("Deleting subscription: " , sub)
            subClass = sns_resource.Subscription(sub['SubscriptionArn'])
            subClass.delete()

        # Delete topic
        print("Deleting topic: " , topic_delete_arn)
        sns.delete_topic(TopicArn = topic_delete_arn)    
    #--------------------------------------------------------------------------------------------------------
    # Get queue ARNs matching name (from SQS client)
    queues = sqs.list_queues(QueueNamePrefix=QUEUE_NAME_PREFIX)
    for queueUrl in queues['QueueUrls']:
        print("Deleting queue: " , queueUrl)
        sqs.delete_queue(QueueUrl = queueUrl)


# Function to publish a batch of messages.
# Receives topic_arn, a list of IDs to send to, and a list of messages
def publish_message_batch(topic_arn, ids, messages):
    # Empty list of batch request entries
    batch_request_entries = []
    
    # Iterate through provided IDs, construct batch entry using ID + corresponding message
    for i in range(len(ids)):
        id_str = str(ids[i])
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
    
    response = sns.publish_batch(
        TopicArn = topic_arn,
        PublishBatchRequestEntries = batch_request_entries
    )
    
    if len(response['Failed']) == 0:
        return response
    
    # Check for failed messages, and attempt one re-try.
    retry_count = 0
    
    if (len(response['Failed'])) > 0 and (retry_count < 2):
        resend_ids, resend_msgs = [], []
        for failed_msg in response['Failed']:
            failed_worker_id = failed_msg['Id']
            failed_msg_content = str(failed_msg['Message'])
            resend_ids.append(failed_worker_id)
            resend_msgs.append(failed_msg_content)
        retry_response = publish_message_batch(topic_arn, resend_ids, resend_msgs)
        retry_count += 1
    
    if len(retry_response['Failed'] > 0):
        print("Retry failed!")
    
    return retry_response
    
# Function to calculate size in bytes of string. 
# When writing function to prepare send_buffer from Hsend and InfData, 
# call this function to get size of each csr message. Don't add to 
# current batch if it will make size exceed MAX_BATCH_MESSAGE_BYTES
def utf8len(s):
    return len(s.encode('utf-8'))
    
def get_queue_url(id):
    queue_name = QUEUE_NAME_PREFIX + str(id)
    response = sqs.get_queue_url(
        QueueName=queue_name
    )
    return response['QueueUrl']
   
def get_topics():
    return sns.list_topics()

def get_sparse_dnn_topic_arn():
    topics = get_topics()['Topics']
    for topic in topics:
        if topic['TopicArn'].endswith(TOPIC_NAME_PREFIX):
            return topic['TopicArn']

def purge_all_queues():
    for i in range(numWorkers):
        url = get_queue_url(i)
        sqs.purge_queue(url)

# Main
if __name__ == "__main__":
    sns = sns_client()
    sqs = sqs_client()
    
    id = 1
    numWorkers = 65

    # delete_resources(numWorkers)
    # time.sleep(65)
    
    coordination_topic_arn = create_coordination_topic()
    create_all_queues(numWorkers, coordination_topic_arn)
    
    
    
    
    
    

    
    
  