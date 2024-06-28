import dagshub
dagshub.init(repo_owner='Darkbeast747474', repo_name='MLProject', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
  mlflow.autolog()