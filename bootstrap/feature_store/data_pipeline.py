import kfp
from kfp import dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)
from kfp import kubernetes

@component(packages_to_install=["psycopg2", "pandas"])
def extract_data(
    source_con_details: dict,
    data: Output[Dataset],
):
    import psycopg2
    import pandas as pd
    import os
    
    conn = psycopg2.connect(
        host=source_con_details['host'],
        database=os.getenv("DATABASE-NAME"),
        user=os.getenv("DATABASE-USER"),
        password=os.getenv("DATABASE-PASSWORD"),
    )
    query = f"SELECT * FROM {source_con_details['table']}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.to_csv(data, index=False)

@component(packages_to_install=["pandas"])
def transform_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
):
    import pandas as pd
    
    df = pd.read_csv(input_data)
    df.columns = map(str.lower, df.columns)
    df.to_csv(output_data, index=False)

@component(packages_to_install=["psycopg2", "pandas"])
def load_data(
    target_con_details: dict,
    data: Input[Dataset]
):
    import psycopg2
    import pandas as pd
    import os
    
    conn = psycopg2.connect(
        host=target_con_details['host'],
        database=os.getenv("DATABASE-NAME"),
        user=os.getenv("DATABASE-USER"),
        password=os.getenv("DATABASE-PASSWORD"),
    )
    df = pd.read_csv(data)
    df.to_sql(target_con_details['table'], conn, if_exists='replace', index=False)
    conn.close()


@dsl.pipeline(
  name='ETL Pipeline',
  description='Moves and transforms data from transactions data storage (postgresql) to feast storage (postgresql).'
)
def fraud_training_pipeline(source_con_details: dict, target_con_details: dict):
    extract_task = extract_data(source_con_details=source_con_details)
    kubernetes.use_secret_as_env(extract_task,
                                 secret_name="transactionsdb-info",
                                 secret_key_to_env={'database-name': 'DATABASE-NAME',
                                                    'database-user': 'DATABASE-USER',
                                                    'database-password': 'DATABASE-PASSWORD',
                                                    })
    transform_task = transform_data(input_data=extract_task.outputs["data"])
    load_task = load_data(target_con_details=target_con_details, 
                          data=transform_task.outputs["output_data"])
    kubernetes.use_secret_as_env(load_task,
                                 secret_name="feast",
                                 secret_key_to_env={'database-name': 'DATABASE-NAME',
                                                    'database-user': 'DATABASE-USER',
                                                    'database-password': 'DATABASE-PASSWORD',
                                                    })
    
    

if __name__ == '__main__':
    metadata = {
        "source_con_details": {
            "host": "transactionsdb.mlops-transactionsdb.svc.cluster.local",
            "table": "transactions.transactions",
        },
        "target_con_details": {
            "host": "feast.mlops-feature-store.svc.cluster.local",
            "table": "feast.feast",
        }
    }
        
    namespace_file_path =\
        '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    with open(namespace_file_path, 'r') as namespace_file:
        namespace = namespace_file.read()

    kubeflow_endpoint =\
        f'https://ds-pipeline-dspa.{namespace}.svc:8443'

    sa_token_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    with open(sa_token_file_path, 'r') as token_file:
        bearer_token = token_file.read()

    ssl_ca_cert =\
        '/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt'

    print(f'Connecting to Data Science Pipelines: {kubeflow_endpoint}')
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert
    )

    client.create_run_from_pipeline_func(
        fraud_training_pipeline,
        arguments=metadata,
        experiment_name="fraud-training",
        namespace="mlops-dev-zone",
        enable_caching=True
    )
