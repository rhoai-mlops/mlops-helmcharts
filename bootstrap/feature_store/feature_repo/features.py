from datetime import timedelta

import pandas as pd

from feast import Entity, FeatureService, FeatureView, Field, PushSource, RequestSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.infra.offline_stores.file_source import FileSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64
from feast.data_format import ParquetFormat

transactions = Entity(name="transactions", join_keys=["transaction_id"])

transactions_source = PostgreSQLSource(
    name="transactions_sql",
    query="SELECT * FROM transactions.transactions",
    timestamp_field="event_timestamp",
)

# transactions_source = FileSource(
#     name="transactions_parquet",
#     path="card_transaction_data.parquet",
#     file_format=ParquetFormat(),
#     timestamp_field="event_timestamp",
# )

transactions_features = FeatureView(
    name = "transactions_features",
    entities = [transactions],
    schema = [
        Field(name="distance_from_last_transaction", dtype=Float32),
        Field(name="ratio_to_median_purchase_price", dtype=Float32),
        Field(name="repeat_retailer", dtype=Int64),
        Field(name="used_chip", dtype=Int64),
        Field(name="used_pin_number", dtype=Int64),
        Field(name="online_order", dtype=Int64),
        Field(name="fraud", dtype=Int64),
    ],
    source = transactions_source,
)