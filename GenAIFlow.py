import json
import os

import google_crc32c
from google.cloud import secretmanager
from metaflow import FlowSpec, Parameter
import argparse

# Internal prefix for storing Metaflow parmeters
MF_ARG_PREFIX = "mf_arg_"
MF_ARG_MULTI_PREFIX = "mf_multiarg_"

"""
This file has utility functions to make it easy to convert a command line-training job 
into a Metaflow job. A flow should subclass GenAIFlow.

Note that this utility will be moved to a common repo at some point.

"""

def is_true_flag_action(action):
    return action.const is True and action.default is False and action.type == None

def is_optional_int_value(action):
    return action.type == int and action.default is None

class GenAIFlow(FlowSpec):

    @classmethod
    def import_argparse_to_params(cls, parser):
        """
        Given an argument parser class using the standard python argparse library, construct
        similar Metaflow parameters for the flow that can be triggered in the Metaflow command line
        or a provider UI (Such as Outerbounds). Arguments are have arg_ prefix added to them in the
        metaflow UI.

        Multiple input arguments must be submitted as comma separated strings

        Supports most typical arguments and true flags (default if not set).
        """
        for action in parser._actions:
            if action.default != argparse.SUPPRESS:
                if is_true_flag_action(action):
                    setattr(cls, MF_ARG_PREFIX + action.dest, Parameter("arg_" + action.dest,
                                                                        type=bool,
                                                                        help=f"{action.help}",
                                                                        default=False))
                elif is_optional_int_value(action):
                    setattr(cls, MF_ARG_PREFIX + action.dest, Parameter("arg_" + action.dest,
                                                                        type=bool,
                                                                        help=f"{action.help}",
                                                                        default=False))
                elif action.nargs in ["*", "+"]:
                    setattr(cls, MF_ARG_MULTI_PREFIX + action.dest, Parameter("arg_" + action.dest,
                                                                    type=action.type or str,
                                                                    help=f"{action.help} {(','.join(action.choices)) if action.choices is not None else ''}",
                                                                    default=action.default))
                else:
                    setattr(cls, MF_ARG_PREFIX + action.dest, Parameter("arg_" + action.dest,
                                                                    type=action.type or str,
                                                                    help=f"{action.help} {(','.join(action.choices)) if action.choices is not None else ''}",
                                                                    default=action.default))

    def params_to_args(self):
        """
        Converts the metaflow arguments (passed to a metaflow job) to commandline arguments
        to be interpreted by legacy argparse command line code.

        This should be called at the class level. Note that '-' arg names aren't supported
        in Metaflow and are automatically converted to '_' in the Metaflow UI. They are
        converted back to '-' here.

        Also note that 0 value params are omitted, so items with non-zero defaults are not
        fully supported in some situations.
        """
        args = []
        for name, value in self.__class__.__dict__.items():
            if name.startswith(MF_ARG_MULTI_PREFIX):
                vv = getattr(self, name)
                if vv not in (None, 0, '', False):
                    args.append(f"--{name[len(MF_ARG_MULTI_PREFIX):].replace('_', '-')}")
                    items = str(vv).split(",")
                    for item in items:
                        args.append(item)
            if name.startswith(MF_ARG_PREFIX):
                vv = getattr(self, name)
                if vv not in (None, 0, '', False):
                    args.append(f"--{name[len(MF_ARG_PREFIX):].replace('_', '-')}")
                    if str(vv) != 'True':
                        args.append(str(vv))
        return args

    def get_project_id(self):
        """
        Returns what GCP project we are running on
        """
        return "moz-fx-mozsoc-ml-nonprod"

    def load_secret(self, secret_id: str) -> str:
        """
        Utility function to load a secret from GCP
        """
        client = secretmanager.SecretManagerServiceClient()
        secret_path = client.secret_version_path(self.get_project_id(), secret_id, "latest")
        response = client.access_secret_version(request={"name": secret_path})
        crc32c = google_crc32c.Checksum()
        crc32c.update(response.payload.data)
        if response.payload.data_crc32c != int(crc32c.hexdigest(), 16):
            raise Exception(f"Secret CRC Corrupted in project {self.get_project_id()} and path {secret_path}")
        return response.payload.data.decode("UTF-8")

    def load_remote_env(self):
        """"
        Load secrets for low access service accounts. This provides secrets to use OpenAI
        and write W&B artifacts
        """
        print("Loading common secrets from GCP")
        json_secrets = ['metaflow-job-secrets']
        for secret_id in json_secrets:
            raw_env = self.load_secret(secret_id)
            envs = json.loads(raw_env)
            for k, v in envs.items():
                os.environ[k.upper()] = v
