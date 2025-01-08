import json
from typing import List, Dict, Union
import pickle

import requests
import numpy as np
import os
import pyarrow.fs

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
    ):
        """Initialize the model name, predictor host, Feast serving URL,
           entity IDs, and feature references

        Args:
            name (str): Name of the model.
            predictor_host (str): The host in which the predictor runs.
            protocol (str): The protocol in which the predictor runs.
        """
        super().__init__(name, PredictorConfig(predictor_host, predictor_protocol, predictor_use_ssl))
        self.predictor_host = predictor_host
        self.predictor_protocol = predictor_protocol
        self.scaler_file_path = scaler_file_path
        self.encoder_file_path = encoder_file_path
        logger.info("Model name = %s", name)
        logger.info("Protocol = %s", predictor_protocol)
        logger.info("Predictor host = %s", predictor_host)
        logger.info("Predictor use SSL = %s", predictor_use_ssl)
        logger.info("Scaler file path = %s", scaler_file_path)
        logger.info("Encoder file path = %s", encoder_file_path)

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

    def preprocess(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferRequest]:

        self.scaler = self.load_scaler()

        logger.info("Incoming payload: %s", payload)

        if isinstance(payload, InferRequest):
            data = self.scaler.transform(payload.inputs[0].data)
        else:
            headers["request-type"] = "v1"
            data = [
                self.scaler.transform(instance["data"])
                for instance in payload["instances"]
            ]

        data = np.asarray(data, dtype=np.float32).reshape(payload.inputs[0].shape)
        logger.info("Data dtype: %s, shape: %s", data.dtype, data.shape)
        logger.info("Data content: %s", data)

        if self.protocol == PredictorProtocol.REST_V1.value:
            inputs = [{"data": d.tolist()} for d in data]
            payload = {"instances": inputs}
            return payload
        else:
            infer_inputs = [
                InferInput(
                    name=payload.inputs[0].name,
                    datatype=payload.inputs[0].datatype,
                    shape=list(data.shape),
                    data=data,
                )
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