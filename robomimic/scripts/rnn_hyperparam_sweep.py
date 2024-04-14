import argparse

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils


def make_generator(config_file, script_file):
    """
    Implement this function to setup your own hyperparameter scan!
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file
    )

    # use RNN with horizon 10
    generator.add_param(
        key="algo.rnn.enabled",
        name="", 
        group=0, 
        values=[True],
    )
    generator.add_param(
        key="train.seq_length", 
        name="", 
        group=0, 
        values=[10], 
)
    generator.add_param(
        key="algo.rnn.horizon",
        name="", 
        group=0, 
        values=[10], 
    )

    # LR - 1e-3, 1e-4
    generator.add_param(
        key="algo.optim_params.policy.learning_rate.initial", 
        name="plr", 
        group=1, 
        values=[1e-3, 1e-4], 
    )

    # GMM y / n
    generator.add_param(
        key="algo.gmm.enabled", 
        name="gmm", 
        group=2, 
        values=[True, False], 
        value_names=["t", "f"],
    )

    # RNN dim 400 + MLP dims (1024, 1024) vs. RNN dim 1000 + empty MLP dims ()
    generator.add_param(
        key="algo.rnn.hidden_dim", 
        name="rnnd", 
        group=3, 
        values=[
            400, 
            1000,
        ], 
    )
    generator.add_param(
        key="algo.actor_layer_dims", 
        name="mlp", 
        group=3, 
        values=[
            [1024, 1024], 
            [],
        ], 
        value_names=["1024", "0"],
    )

    # datasets 
    generator.add_param(
        key="train.data",
        name="data",
        group=4,
        values=[
            [{"path": "/home/krishnans/lustre/datasets/bidex_scissors/rollouts_1000.json"}],
            [{"path": "/home/krishnans/lustre/datasets/bidex_switch/rollouts_1000.json"}],
            [{"path": "/home/krishnans/lustre/datasets/bidex_scissors/rollouts_1000.json"}, {"path": "/home/krishnans/lustre/datasets/bidex_switch/rollouts_1000.json"}]
        ],
        value_names=["scissors_1000", "switch_1000", "scissors_switch_1000"]
    )
    return generator


def main(args):

    # make config generator
    generator = make_generator(config_file=args.config, script_file=args.script)

    # generate jsons and script
    generator.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to base json config - will override any defaults.
    parser.add_argument(
        "--config",
        type=str,
        help="path to base config json that will be modified to generate jsons. The jsons will\
            be generated in the same folder as this file.",
    )

    # Script name to generate - will override any defaults
    parser.add_argument(
        "--script",
        type=str,
        help="path to output script that contains commands to run the generated training runs",
    )

    args = parser.parse_args()
    main(args)
