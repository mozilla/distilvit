from metaflow import FlowSpec, Parameter

MF_ARG_PREFIX = "mf_arg_"

class GenAIFlow(FlowSpec):
    @classmethod
    def import_argparse_to_params(cls, parser):
        for action in parser._actions:
            if action.dest != "help":
                setattr(cls, MF_ARG_PREFIX + action.dest, Parameter("arg_" + action.dest,
                                                                type=action.type,
                                                                help=f"{action.help} {(','.join(action.choices)) if action.choices is not None else ''}",
                                                                default=action.default))

    def params_to_args(self):
        args = []
        for name, value in self.__class__.__dict__.items():
            if name.startswith(MF_ARG_PREFIX):
                vv = getattr(self, name)
                if (vv is not None and vv != 0):
                    args.append(f"--{name[len(MF_ARG_PREFIX):]}")
                    args.append(str(vv))
        return args
