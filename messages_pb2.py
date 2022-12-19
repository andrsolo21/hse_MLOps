# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: messages.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0emessages.proto\x12\nhelloworld\"\x1c\n\x0cHelloRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\"\x1f\n\tModelName\x12\x12\n\nmodel_name\x18\x01 \x01(\t\"4\n\x13\x41vailableModelTypes\x12\x1d\n\x15\x61vailable_model_types\x18\x01 \x03(\t\"Q\n\x0cMLModelsList\x12\x16\n\x0eml_models_list\x18\x01 \x03(\t\x12\x12\n\nerror_code\x18\x02 \x01(\x05\x12\x15\n\rerror_message\x18\x03 \x01(\t\"@\n\x14\x43reateModelRequest_2\x12\x12\n\nmodel_type\x18\x01 \x01(\t\x12\x14\n\x0cmodel_params\x18\x02 \x01(\t\">\n\x14ResponseCreatedModel\x12\x12\n\nmodel_path\x18\x01 \x01(\t\x12\x12\n\nmodel_type\x18\x02 \x01(\t\"K\n\x0b\x46itMetaData\x12\x11\n\textension\x18\x01 \x01(\t\x12\x12\n\nmodel_name\x18\x02 \x01(\t\x12\x15\n\rtarget_column\x18\x03 \x01(\t\"8\n\x0fPredictMetaData\x12\x11\n\textension\x18\x01 \x01(\t\x12\x12\n\nmodel_name\x18\x02 \x01(\t\"\x95\x01\n\x08SendFile\x12/\n\x0c\x66it_metadata\x18\x01 \x01(\x0b\x32\x17.helloworld.FitMetaDataH\x00\x12\x14\n\nchunk_data\x18\x02 \x01(\x0cH\x00\x12\x37\n\x10predict_metadata\x18\x03 \x01(\x0b\x32\x1b.helloworld.PredictMetaDataH\x00\x42\t\n\x07request\"Q\n\x10ResponseFitModel\x12\x12\n\nis_trained\x18\x01 \x01(\x08\x12\x12\n\nerror_code\x18\x02 \x01(\x05\x12\x15\n\rerror_message\x18\x03 \x01(\t\"Q\n\x13ResponsePredictData\x12\x0f\n\x07predict\x18\x01 \x03(\x02\x12\x12\n\nerror_code\x18\x02 \x01(\x05\x12\x15\n\rerror_message\x18\x03 \x01(\t\"T\n\x11ResponseModelInfo\x12\x14\n\x0cmodel_params\x18\x01 \x01(\t\x12\x12\n\nerror_code\x18\x02 \x01(\x05\x12\x15\n\rerror_message\x18\x03 \x01(\t2\x96\x04\n\x07Greeter\x12U\n\x16GetAvailableModelTypes\x12\x18.helloworld.HelloRequest\x1a\x1f.helloworld.AvailableModelTypes\"\x00\x12G\n\x0fGetMLModelsList\x12\x18.helloworld.HelloRequest\x1a\x18.helloworld.MLModelsList\"\x00\x12S\n\x0b\x43reateModel\x12 .helloworld.CreateModelRequest_2\x1a .helloworld.ResponseCreatedModel\"\x00\x12\x42\n\x08\x46itModel\x12\x14.helloworld.SendFile\x1a\x1c.helloworld.ResponseFitModel\"\x00(\x01\x12H\n\x0bPredictData\x12\x14.helloworld.SendFile\x1a\x1f.helloworld.ResponsePredictData\"\x00(\x01\x12\x46\n\x0cGetModelInfo\x12\x15.helloworld.ModelName\x1a\x1d.helloworld.ResponseModelInfo\"\x00\x12@\n\x0b\x44\x65leteModel\x12\x15.helloworld.ModelName\x1a\x18.helloworld.MLModelsList\"\x00\x42\x36\n\x1bio.grpc.examples.helloworldB\x0fHelloWorldProtoP\x01\xa2\x02\x03HLWb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'messages_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\033io.grpc.examples.helloworldB\017HelloWorldProtoP\001\242\002\003HLW'
  _HELLOREQUEST._serialized_start=30
  _HELLOREQUEST._serialized_end=58
  _MODELNAME._serialized_start=60
  _MODELNAME._serialized_end=91
  _AVAILABLEMODELTYPES._serialized_start=93
  _AVAILABLEMODELTYPES._serialized_end=145
  _MLMODELSLIST._serialized_start=147
  _MLMODELSLIST._serialized_end=228
  _CREATEMODELREQUEST_2._serialized_start=230
  _CREATEMODELREQUEST_2._serialized_end=294
  _RESPONSECREATEDMODEL._serialized_start=296
  _RESPONSECREATEDMODEL._serialized_end=358
  _FITMETADATA._serialized_start=360
  _FITMETADATA._serialized_end=435
  _PREDICTMETADATA._serialized_start=437
  _PREDICTMETADATA._serialized_end=493
  _SENDFILE._serialized_start=496
  _SENDFILE._serialized_end=645
  _RESPONSEFITMODEL._serialized_start=647
  _RESPONSEFITMODEL._serialized_end=728
  _RESPONSEPREDICTDATA._serialized_start=730
  _RESPONSEPREDICTDATA._serialized_end=811
  _RESPONSEMODELINFO._serialized_start=813
  _RESPONSEMODELINFO._serialized_end=897
  _GREETER._serialized_start=900
  _GREETER._serialized_end=1434
# @@protoc_insertion_point(module_scope)