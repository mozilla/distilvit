from metaflow import FlowSpec, Parameter
import argparse

MF_ARG_PREFIX = "mf_arg_"


def is_true_flag_action(action):
    return action.const is True and action.default is False and action.type == None


class GenAIFlow(FlowSpec):

    @classmethod
    def import_argparse_to_params(cls, parser):
        for action in parser._actions:
            if action.default != argparse.SUPPRESS:
                if is_true_flag_action(action):
                    setattr(cls, MF_ARG_PREFIX + action.dest, Parameter("arg_" + action.dest,
                                                                        type=bool,
                                                                        help=f"{action.help}",
                                                                        default=False))
                else:
                    setattr(cls, MF_ARG_PREFIX + action.dest, Parameter("arg_" + action.dest,
                                                                    type=action.type or str,
                                                                    help=f"{action.help} {(','.join(action.choices)) if action.choices is not None else ''}",
                                                                    default=action.default))

    def params_to_args(self):
        args = []
        for name, value in self.__class__.__dict__.items():
            if name.startswith(MF_ARG_PREFIX):
                vv = getattr(self, name)
                if vv not in (None, 0, '', False):
                    args.append(f"--{name[len(MF_ARG_PREFIX):].replace('_', '-')}")
                    args.append(str(vv))
        return args
