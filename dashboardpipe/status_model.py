
from sklearn import tree
import pickle
import boto3
import boto3.session


class StatusModel():
    def __init__(self):
        pass

    def set_params(self, **params):
        pass

    def fit(self, x, y):
        pass

    def pred(self, x):
        pass

    def save_model(self, bucket, key):
        pass

    def load_model(self):
        pass

class SimpleTreeModel(StatusModel):
    def __init__(self):
        super().__init__()
        self.model = None
        self.params = None

    def set_params(self, **params):
        self.params = params
    
    def fit_model(self, x, y, **g):
        self.model = tree.DecisionTreeClassifier(**g)
        self.model.fit(x, y)

    def pred(self, x):
        return self.model.predict(x)

    def save_model(self, bucket, key):
        model_pkl_obj = pickle.dumps(self.model)
        session = boto3.Session()
        s3_resource = session.resource('s3')
        s3_resource.Object(bucket, key).put(Body=model_pkl_obj)
    
    def load_model(self, bucket, key):
        s3client = boto3.client('s3')
        pklobj = s3client.get_object(Bucket=bucket, Key=key)
        body = pklobj['Body'].read()
        self.model = pickle.loads(body)


    

