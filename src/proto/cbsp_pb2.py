# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/cbsp.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10proto/cbsp.proto\x12\x04\x63\x62sp\"\x07\n\x05\x45mpty\"/\n\x0eParameterBytes\x12\x0e\n\x06tensor\x18\x01 \x01(\x0c\x12\r\n\x05shape\x18\x02 \x03(\x05\"L\n\x11PytorchParameters\x12(\n\nparameters\x18\x01 \x03(\x0b\x32\x14.cbsp.ParameterBytes\x12\r\n\x05\x64type\x18\x02 \x01(\t\"m\n\x08\x43onstant\x12\x10\n\x06\x64ouble\x18\x01 \x01(\x01H\x00\x12\x10\n\x06sint64\x18\x02 \x01(\x12H\x00\x12\x0e\n\x04\x62ool\x18\x03 \x01(\x08H\x00\x12\x10\n\x06string\x18\x04 \x01(\tH\x00\x12\x0f\n\x05\x62ytes\x18\x05 \x01(\x0cH\x00\x42\n\n\x08\x63onstant\"\xf0\t\n\rClientMessage\x12;\n\x0eget_parameters\x18\x01 \x01(\x0b\x32!.cbsp.ClientMessage.GetParametersH\x00\x12\x33\n\nget_config\x18\x02 \x01(\x0b\x32\x1d.cbsp.ClientMessage.GetConfigH\x00\x12=\n\x0fsend_parameters\x18\x03 \x01(\x0b\x32\".cbsp.ClientMessage.SendParametersH\x00\x12\x37\n\x0csend_results\x18\x04 \x01(\x0b\x32\x1f.cbsp.ClientMessage.SendResultsH\x00\x1a\xb7\x01\n\rGetParameters\x12.\n\x04type\x18\x01 \x01(\x0e\x32 .cbsp.ClientMessage.MESSAGE_TYPE\x12\x39\n\x04info\x18\x02 \x03(\x0b\x32+.cbsp.ClientMessage.GetParameters.InfoEntry\x1a;\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1d\n\x05value\x18\x02 \x01(\x0b\x32\x0e.cbsp.Constant:\x02\x38\x01\x1a\xaf\x01\n\tGetConfig\x12.\n\x04type\x18\x01 \x01(\x0e\x32 .cbsp.ClientMessage.MESSAGE_TYPE\x12\x35\n\x04info\x18\x02 \x03(\x0b\x32\'.cbsp.ClientMessage.GetConfig.InfoEntry\x1a;\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1d\n\x05value\x18\x02 \x01(\x0b\x32\x0e.cbsp.Constant:\x02\x38\x01\x1a\xe6\x01\n\x0eSendParameters\x12.\n\x04type\x18\x01 \x01(\x0e\x32 .cbsp.ClientMessage.MESSAGE_TYPE\x12:\n\x04info\x18\x02 \x03(\x0b\x32,.cbsp.ClientMessage.SendParameters.InfoEntry\x12+\n\nparameters\x18\x03 \x01(\x0b\x32\x17.cbsp.PytorchParameters\x1a;\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1d\n\x05value\x18\x02 \x01(\x0b\x32\x0e.cbsp.Constant:\x02\x38\x01\x1a\xb2\x02\n\x0bSendResults\x12.\n\x04type\x18\x01 \x01(\x0e\x32 .cbsp.ClientMessage.MESSAGE_TYPE\x12\x37\n\x04info\x18\x02 \x03(\x0b\x32).cbsp.ClientMessage.SendResults.InfoEntry\x12=\n\x07results\x18\x03 \x03(\x0b\x32,.cbsp.ClientMessage.SendResults.ResultsEntry\x1a;\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1d\n\x05value\x18\x02 \x01(\x0b\x32\x0e.cbsp.Constant:\x02\x38\x01\x1a>\n\x0cResultsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1d\n\x05value\x18\x02 \x01(\x0b\x32\x0e.cbsp.Constant:\x02\x38\x01\"Y\n\x0cMESSAGE_TYPE\x12\x12\n\x0eGET_PARAMETERS\x10\x00\x12\x0e\n\nGET_CONFIG\x10\x01\x12\x13\n\x0fSEND_PARAMETERS\x10\x02\x12\x10\n\x0cSEND_RESULTS\x10\x03\x42\x10\n\x0e\x63lient_message\"\xa4\x07\n\rServerMessage\x12<\n\x0eget_parameters\x18\x01 \x01(\x0b\x32\".cbsp.ServerMessage.SendParametersH\x00\x12\x35\n\x0bsend_config\x18\x02 \x01(\x0b\x32\x1e.cbsp.ServerMessage.SendConfigH\x00\x12=\n\x0fnormal_response\x18\x03 \x01(\x0b\x32\".cbsp.ServerMessage.NormalResponseH\x00\x12\x17\n\rtest_response\x18\x04 \x01(\tH\x00\x1a\xe6\x01\n\x0eSendParameters\x12.\n\x04type\x18\x01 \x01(\x0e\x32 .cbsp.ServerMessage.MESSAGE_TYPE\x12:\n\x04info\x18\x02 \x03(\x0b\x32,.cbsp.ServerMessage.SendParameters.InfoEntry\x12+\n\nparameters\x18\x03 \x01(\x0b\x32\x17.cbsp.PytorchParameters\x1a;\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1d\n\x05value\x18\x02 \x01(\x0b\x32\x0e.cbsp.Constant:\x02\x38\x01\x1a\xb1\x01\n\nSendConfig\x12.\n\x04type\x18\x01 \x01(\x0e\x32 .cbsp.ServerMessage.MESSAGE_TYPE\x12\x36\n\x04info\x18\x02 \x03(\x0b\x32(.cbsp.ServerMessage.SendConfig.InfoEntry\x1a;\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1d\n\x05value\x18\x02 \x01(\x0b\x32\x0e.cbsp.Constant:\x02\x38\x01\x1a\xcb\x01\n\x0eNormalResponse\x12.\n\x04type\x18\x01 \x01(\x0e\x32 .cbsp.ServerMessage.MESSAGE_TYPE\x12:\n\x04info\x18\x02 \x03(\x0b\x32,.cbsp.ServerMessage.NormalResponse.InfoEntry\x12\x10\n\x08response\x18\x03 \x01(\t\x1a;\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1d\n\x05value\x18\x02 \x01(\x0b\x32\x0e.cbsp.Constant:\x02\x38\x01\"I\n\x0cMESSAGE_TYPE\x12\x13\n\x0fSEND_PARAMETERS\x10\x00\x12\x0f\n\x0bSEND_CONFIG\x10\x01\x12\x13\n\x0fNORMAL_RESPONSE\x10\x02\x42\x10\n\x0eserver_message2W\n\x14\x43ommunicationService\x12?\n\x13\x42idirectionalStream\x12\x13.cbsp.ClientMessage\x1a\x13.cbsp.ServerMessageb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.cbsp_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_CLIENTMESSAGE_GETPARAMETERS_INFOENTRY']._options = None
  _globals['_CLIENTMESSAGE_GETPARAMETERS_INFOENTRY']._serialized_options = b'8\001'
  _globals['_CLIENTMESSAGE_GETCONFIG_INFOENTRY']._options = None
  _globals['_CLIENTMESSAGE_GETCONFIG_INFOENTRY']._serialized_options = b'8\001'
  _globals['_CLIENTMESSAGE_SENDPARAMETERS_INFOENTRY']._options = None
  _globals['_CLIENTMESSAGE_SENDPARAMETERS_INFOENTRY']._serialized_options = b'8\001'
  _globals['_CLIENTMESSAGE_SENDRESULTS_INFOENTRY']._options = None
  _globals['_CLIENTMESSAGE_SENDRESULTS_INFOENTRY']._serialized_options = b'8\001'
  _globals['_CLIENTMESSAGE_SENDRESULTS_RESULTSENTRY']._options = None
  _globals['_CLIENTMESSAGE_SENDRESULTS_RESULTSENTRY']._serialized_options = b'8\001'
  _globals['_SERVERMESSAGE_SENDPARAMETERS_INFOENTRY']._options = None
  _globals['_SERVERMESSAGE_SENDPARAMETERS_INFOENTRY']._serialized_options = b'8\001'
  _globals['_SERVERMESSAGE_SENDCONFIG_INFOENTRY']._options = None
  _globals['_SERVERMESSAGE_SENDCONFIG_INFOENTRY']._serialized_options = b'8\001'
  _globals['_SERVERMESSAGE_NORMALRESPONSE_INFOENTRY']._options = None
  _globals['_SERVERMESSAGE_NORMALRESPONSE_INFOENTRY']._serialized_options = b'8\001'
  _globals['_EMPTY']._serialized_start=26
  _globals['_EMPTY']._serialized_end=33
  _globals['_PARAMETERBYTES']._serialized_start=35
  _globals['_PARAMETERBYTES']._serialized_end=82
  _globals['_PYTORCHPARAMETERS']._serialized_start=84
  _globals['_PYTORCHPARAMETERS']._serialized_end=160
  _globals['_CONSTANT']._serialized_start=162
  _globals['_CONSTANT']._serialized_end=271
  _globals['_CLIENTMESSAGE']._serialized_start=274
  _globals['_CLIENTMESSAGE']._serialized_end=1538
  _globals['_CLIENTMESSAGE_GETPARAMETERS']._serialized_start=526
  _globals['_CLIENTMESSAGE_GETPARAMETERS']._serialized_end=709
  _globals['_CLIENTMESSAGE_GETPARAMETERS_INFOENTRY']._serialized_start=650
  _globals['_CLIENTMESSAGE_GETPARAMETERS_INFOENTRY']._serialized_end=709
  _globals['_CLIENTMESSAGE_GETCONFIG']._serialized_start=712
  _globals['_CLIENTMESSAGE_GETCONFIG']._serialized_end=887
  _globals['_CLIENTMESSAGE_GETCONFIG_INFOENTRY']._serialized_start=650
  _globals['_CLIENTMESSAGE_GETCONFIG_INFOENTRY']._serialized_end=709
  _globals['_CLIENTMESSAGE_SENDPARAMETERS']._serialized_start=890
  _globals['_CLIENTMESSAGE_SENDPARAMETERS']._serialized_end=1120
  _globals['_CLIENTMESSAGE_SENDPARAMETERS_INFOENTRY']._serialized_start=650
  _globals['_CLIENTMESSAGE_SENDPARAMETERS_INFOENTRY']._serialized_end=709
  _globals['_CLIENTMESSAGE_SENDRESULTS']._serialized_start=1123
  _globals['_CLIENTMESSAGE_SENDRESULTS']._serialized_end=1429
  _globals['_CLIENTMESSAGE_SENDRESULTS_INFOENTRY']._serialized_start=650
  _globals['_CLIENTMESSAGE_SENDRESULTS_INFOENTRY']._serialized_end=709
  _globals['_CLIENTMESSAGE_SENDRESULTS_RESULTSENTRY']._serialized_start=1367
  _globals['_CLIENTMESSAGE_SENDRESULTS_RESULTSENTRY']._serialized_end=1429
  _globals['_CLIENTMESSAGE_MESSAGE_TYPE']._serialized_start=1431
  _globals['_CLIENTMESSAGE_MESSAGE_TYPE']._serialized_end=1520
  _globals['_SERVERMESSAGE']._serialized_start=1541
  _globals['_SERVERMESSAGE']._serialized_end=2473
  _globals['_SERVERMESSAGE_SENDPARAMETERS']._serialized_start=1764
  _globals['_SERVERMESSAGE_SENDPARAMETERS']._serialized_end=1994
  _globals['_SERVERMESSAGE_SENDPARAMETERS_INFOENTRY']._serialized_start=650
  _globals['_SERVERMESSAGE_SENDPARAMETERS_INFOENTRY']._serialized_end=709
  _globals['_SERVERMESSAGE_SENDCONFIG']._serialized_start=1997
  _globals['_SERVERMESSAGE_SENDCONFIG']._serialized_end=2174
  _globals['_SERVERMESSAGE_SENDCONFIG_INFOENTRY']._serialized_start=650
  _globals['_SERVERMESSAGE_SENDCONFIG_INFOENTRY']._serialized_end=709
  _globals['_SERVERMESSAGE_NORMALRESPONSE']._serialized_start=2177
  _globals['_SERVERMESSAGE_NORMALRESPONSE']._serialized_end=2380
  _globals['_SERVERMESSAGE_NORMALRESPONSE_INFOENTRY']._serialized_start=650
  _globals['_SERVERMESSAGE_NORMALRESPONSE_INFOENTRY']._serialized_end=709
  _globals['_SERVERMESSAGE_MESSAGE_TYPE']._serialized_start=2382
  _globals['_SERVERMESSAGE_MESSAGE_TYPE']._serialized_end=2455
  _globals['_COMMUNICATIONSERVICE']._serialized_start=2475
  _globals['_COMMUNICATIONSERVICE']._serialized_end=2562
# @@protoc_insertion_point(module_scope)
