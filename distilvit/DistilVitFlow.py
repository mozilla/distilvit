from metaflow import FlowSpec, step, kubernetes, environment
from train import train, parse_args, environ_dict


class DistilVitFlow(FlowSpec):
    @kubernetes(
        image="us-docker.pkg.dev/moz-fx-mozsoc-ml-nonprod/metaflow-dockers/metaflow_gpu:rolf-distilvit-build-test",
        gpu=1)
    @environment(
        vars=environ_dict
    )
    @step
    def start(self):
        args = parse_args([])
        train(args)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    DistilVitFlow()
