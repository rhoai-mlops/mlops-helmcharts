import json
from typing import List, Dict, Union
import pickle

import requests
import numpy as np
import os
import pyarrow.fs
import pandas as pd

import kserve
from kserve import InferRequest, InferResponse, InferInput, InferOutput
from kserve.protocol.grpc import grpc_predict_v2_pb2 as pb
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferResponse
from kserve.logging import logger
from kserve.model import PredictorConfig, PredictorProtocol


class MusicTransformer(kserve.Model):
    def __init__(
        self,
        name: str,
        predictor_host: str,
        predictor_protocol: str,
        predictor_use_ssl: bool,
        scaler_file_path: str,
        encoder_file_path: str,
        feast_server_url: str,
        feature_service: str,
        entity_id_name: str,
    ):
        """Initialize the model name, predictor host, Feast serving URL,
           entity IDs, and feature references

        Args:
            name (str): Name of the model.
            predictor_host (str): The host in which the predictor runs.
            protocol (str): The protocol in which the predictor runs.
            scaler_file_path (str): Path to the scaler artifact
            encoder_file_path (str): Path to the label encoder artifact
            feast_serving_url (str): The Feast feature server URL, in the form
            of <host_name:port>
            feature_service (str): The Feast feature service that will be used
        """
        super().__init__(name, PredictorConfig(predictor_host, predictor_protocol, predictor_use_ssl))
        self.predictor_host = predictor_host
        self.predictor_protocol = predictor_protocol
        self.scaler_file_path = scaler_file_path
        self.encoder_file_path = encoder_file_path
        self.feast_server_url = feast_server_url
        self.feature_service = feature_service
        self.entity_id_name = entity_id_name
        logger.info("Model name = %s", name)
        logger.info("Protocol = %s", predictor_protocol)
        logger.info("Predictor host = %s", predictor_host)
        logger.info("Predictor use SSL = %s", predictor_use_ssl)
        logger.info("Scaler file path = %s", scaler_file_path)
        logger.info("Encoder file path = %s", encoder_file_path)
        logger.info("Feast Server URL = %s", feast_server_url)
        logger.info("Feature Service = %s", feature_service)
        logger.info("Entity ID Name = %s", entity_id_name)

        self.ready = True

    def load_scaler(self):
        with open(self.scaler_file_path, 'rb') as file:
            scaler = pickle.load(file)
        logger.info(f"Scaler loaded from {self.scaler_file_path}")
        return scaler
    
    def load_label_encoder(self):
        with open(self.encoder_file_path, 'rb') as file:
            label_encoder = pickle.load(file)
        logger.info(f"Label encoder loaded from {self.encoder_file_path}")
        return label_encoder
    
    def request_features(self, entities):
        entities = {
            self.entity_id_name: entities,
        }
        json_data = {
            "feature_service": self.feature_service,
            "entities": entities
        }
        response = requests.post(f"{self.feast_server_url}/get-online-features", json=json_data, verify=False)
        logger.info("feast response status is %s", response.status_code)
        logger.info("feast response headers %s", response.headers)
        response_dict = response.json()
        logger.info("feast response body %s", response_dict)
        return response_dict
    
    def create_feature_dict(self, feature_dict):
        """
        feature_dict has the following format:
        {
            metadata: {
                feature_names: [ 'feature names with same order as the results list' ]
            }
            results: [
                {
                    values: [ 'values for all entities requests' ]
                    statuses: ...
                    event_timestamps: ...
                },
                ...
            ]
        }
        We will simply merge the feature names with their values into a dictionary
        """
        features = {}
        for idx, name in enumerate(feature_dict["metadata"]["feature_names"]):
            features[name] = feature_dict["results"][idx]["values"]
        return features
    

    def preprocess(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferRequest]:

        self.scaler = self.load_scaler()

        logger.info("Incoming payload: %s", payload)

        if isinstance(payload, InferRequest):
            data = payload.inputs[0].data
        else:
            headers["request-type"] = "v1"
            data = [
                instance["data"]
                for instance in payload["instances"]
            ]

        logger.info("Data content: %s", data)

        feature_dict = self.request_features(entities=np.array(data).flatten().tolist())
        clean_features_dict = self.create_feature_dict(feature_dict)
        logger.info("Features: %s", clean_features_dict)
        features_df = pd.DataFrame(clean_features_dict)[self.scaler.feature_names_in_]
        scaled_df = pd.DataFrame(self.scaler.transform(features_df), columns=features_df.columns)
        data_dict = {name: scaled_df[[name]].to_numpy() for name in scaled_df.columns}

        if self.protocol == PredictorProtocol.REST_V1.value:
            inputs = [{"data": d.tolist()} for d in data]
            payload = {"instances": inputs}
            return payload
        else:
            infer_inputs = [
                InferInput(
                    name=name,
                    datatype="FP32",
                    shape=list(data_dict[name].shape),
                    data=data_dict[name].astype(np.float32),
                )
                for name in data_dict.keys()
            ] 
            infer_request = InferRequest(model_name=self.name, infer_inputs=infer_inputs)
            return infer_request

    def postprocess(
        self, infer_response: Union[Dict, InferResponse, ModelInferResponse], headers: Dict[str, str] = None,
    ) -> Union[Dict, InferResponse]:

        self.label_encoder = self.load_label_encoder()
        
        prediction = infer_response.outputs[0].as_numpy()
        logger.info("The output from model predict is %s", prediction)
        country_codes = self.label_encoder.classes_
        logger.info("Country code is %s", country_codes)

        # Note, we only handle postprocessing for V2 at the moment
        if "request-type" in headers and headers["request-type"] == "v1":
            if self.protocol == PredictorProtocol.REST_V1.value:
                return infer_response
            else:
                return {"predictions": infer_response.outputs[0].as_numpy().tolist()}
        else:
            infer_response.outputs.append(
                InferOutput(
                    name="country_codes",
                    datatype="BYTES",
                    shape=country_codes.shape,
                    data=country_codes.tolist(),
                )
            )
            return infer_response