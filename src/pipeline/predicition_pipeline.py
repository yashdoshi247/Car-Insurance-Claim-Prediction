import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self,df):
        self.test_df = df
        self.policy_id = df['policy_id']

    def predict(self):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            test_df_preprocessed=preprocessor.transform(self.test_df)
            preds=model.predict(test_df_preprocessed)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def make_submission_df(self,predictions):
        submission_df = pd.DataFrame(self.policy_id,columns=['policy_id'])
        submission_df['is_claim']=predictions
        return submission_df
