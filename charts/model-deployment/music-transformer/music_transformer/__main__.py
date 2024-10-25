import argparse
import kserve
from kserve import logging

from .music_transformer import MusicTransformer

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])

parser.add_argument(
    "--scaler_file_path", 
    help="The file path to the scaler in the sidecar", 
    required=True
)
parser.add_argument(
    "--encoder_file_path", 
    help="The firl path to the encoder in the sidecar", 
    required=True
)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    transformer = MusicTransformer(
        name=args.model_name,
        predictor_host=args.predictor_host,
        predictor_protocol=args.predictor_protocol,
        predictor_use_ssl=args.predictor_use_ssl,
        scaler_file_path=args.scaler_file_path,
        encoder_file_path=args.encoder_file_path,
    )
    server = kserve.ModelServer()
    server.start(models=[transformer])