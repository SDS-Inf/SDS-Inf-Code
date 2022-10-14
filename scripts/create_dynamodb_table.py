import boto3
import time

DYNAMODB_TABLE_NAME = "SPARSEDNN_NULL_TARGETS"

dynamodb = boto3.resource('dynamodb', region_name='eu-west-1')
dynamodb_client = boto3.client('dynamodb', region_name='eu-west-1')

if DYNAMODB_TABLE_NAME in dynamodb_client.list_tables()['TableNames']:
    # # Get table
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    
    # Drop table
    response = table.delete()
    
    time.sleep(10)

while DYNAMODB_TABLE_NAME in dynamodb_client.list_tables()['TableNames']:

    time.sleep(5)

# NEED TO CREATE DYNAMODB TABLE. WILL ALSO NEED TO CLEAR IT OUT (OR MAYBE DROP/RECREATE IT, POSSIBLY IN CO-ORDINATOR?)
dynamoDBResponse = dynamodb.create_table(
    TableName=DYNAMODB_TABLE_NAME,
    KeySchema=[
        {
            'AttributeName': 'LayerTarget',
            'KeyType': 'HASH' # Partition key
        },
        {
            'AttributeName': 'Source',
            'KeyType': 'RANGE' # Sort key
            
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'LayerTarget',
            'AttributeType': 'S'
        },
        {
            'AttributeName': 'Source',
            'AttributeType': 'N'
        }
    ],
    BillingMode='PAY_PER_REQUEST'
)

print(dynamoDBResponse)

