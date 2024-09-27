import argparse
import kserve
from kserve import logging

from .fraud_transformer import FraudTransformer

# DEFAULT_MODEL_NAME = "fraud"
# DEFAULT_PROTOCOL = "v1"

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
# parser.add_argument(
#     "--predictor_host", 
#     help="The URL for the model predict function", 
#     required=True
# )
# parser.add_argument(
#     "--protocol",
#     default=DEFAULT_PROTOCOL, 
#     help="The protocol for the predictor"
# )
# parser.add_argument(
#     "--model_name",
#     default=DEFAULT_MODEL_NAME,
#     help="The name that the model is served under.",
# )
parser.add_argument(
    "--feast_serving_url",
    type=str,
    default="",
    help="The url of the Feast feature server.",
    required=False,
)
parser.add_argument(
    "--entity_id_name",
    type=str,
    help="Entity id name to use as keys in the feature store.",
    required=True,
)
parser.add_argument(
    "--feature_refs",
    type=str,
    nargs="+",
    help="A list of features to retrieve from the feature store.",
    required=True,
)


args, _ = parser.parse_known_args()

if __name__ == "__main__":
    # if args.configure_logging:
    #     logging.configure_logging(args.log_config_file)
    transformer = FraudTransformer(
        name=args.model_name,
        predictor_host=args.predictor_host,
        protocol=args.protocol,
        feast_serving_url=args.feast_serving_url,
        entity_id_name=args.entity_id_name,
        feature_refs=args.feature_refs,
    )
    server = kserve.ModelServer()
    server.start(models=[transformer])