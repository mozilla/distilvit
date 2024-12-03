from metaflow import FlowSpec, step, kubernetes, environment, Parameter

from GenAIFlow import GenAIFlow
from distilvit.train import train, parse_args, environ_dict, get_arg_parser
from types import SimpleNamespace
class DistilVitFlow(GenAIFlow):
    GenAIFlow.import_argparse_to_params(get_arg_parser())
    @kubernetes(
        image="us-docker.pkg.dev/moz-fx-mozsoc-ml-nonprod/metaflow-dockers/metaflow_gpu:rolf-distilvit-build-test",
        gpu=1,
        disk=5000
    )
    @environment(
        vars=environ_dict
    )
    @step
    def start(self):
        args = self.params_to_args()

        print(f"Parsed args are as follows:{args}")

        train(parse_args(args))
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    DistilVitFlow()
