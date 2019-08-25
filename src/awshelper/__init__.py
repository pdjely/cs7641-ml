"""
AWS Helper -- Automated deployment of EC2 workers

These methods all assume that credentials have been preloaded into
~/.aws.
"""

import boto3
import os
import uuid


def createWorker():
    client = boto3.client('ec2', region_name='us-west-1')
    worker = EC2Worker()
    return worker


class EC2Worker:
    def __init__(self):
        # https://blog.ipswitch.com/how-to-create-an-ec2-instance-with-python
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/collections.html
        self.ec2 = boto3.resource('ec2', region_name='us-west-1')
        self.keyname = 'ec2worker-' + str(uuid.uuid4())
        self.keypath = os.path.join(os.path.dirname(__file__), 'keys')
        self.keyfile = os.path.join(self.keypath, 'ec2key.pem')
        self.instance = None

        # If no key in the key directory, create one.
        if not(os.path.exists(self.keyfile)):
            print('EC2 key not found. Creating...')
            outfile = open(self.keyfile, 'w')
            keypair = self.ec2.create_key_pair(KeyName=self.keyname)

            keypairOut = str(keypair.key_material)
            print('Created new keypair: ', keypairOut)
            outfile.write(keypairOut)
            outfile.close()
            os.chmod(self.keyfile, 0o400)

    def __del__(self):
        if self.instance is not None:
            self.instance.terminate()

    def sendJob(self, learner):
        """
        Send a learning job to an ec2 instance

        :param learner:
        :return:
        """
        pass

    def startInstance(self, instanceOS='ubuntu', instanceType='t2.micro'):
        amiID = 'ami-08fd8ae3806f09a08'     # Ubuntu LTS 18
        instance = self.ec2.create_instances(
            ImageId=amiID,
            MinCount=1,
            MaxCount=1,
            InstanceType=instanceType,
            KeyName=self.keyname
        )

        self.instance = instance[0]

