import argparse
import os
import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils
from pathlib import Path


def make_generator(args):
    """
    Implement this function to setup your own hyperparameter scan!
    """
    script_file = args.script
    rm_base = os.path.split(robomimic.__file__)[0]

    if args.use_r3m:
        config_file = os.path.join(rm_base, "exps/templates/bc_r3m.json")
    else:
        config_file = os.path.join(rm_base, "exps/templates/bc_articulate.json")
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file,
        wandb_proj_name="multi_task_experts",
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
        values=[10]
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
        values=[True], 
        value_names=["t"],
    )

    # RNN dim 400 + MLP dims (1024, 1024) vs. RNN dim 1000 + empty MLP dims ()

    generator.add_param(
        key="algo.actor_layer_dims", 
        name="mlp", 
        group=3, 
        values=[[1024, 1024]], 
        value_names=["1024"],
    )

    if args.use_r3m:
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_kwargs.freeze",
            name="freeze_r3m",
            group=4,
            values=[True],
            value_names=["t"]
        )

    # datasets 
    if args.env.startswith("bidex"):
        generator.add_param(
            key="train.data",
            name="data",
            group=5,
            values=[
                [{"path": "/home/krishnans/lustre/datasets/bidex_scissors/rollouts_1000.hdf5"}],
                [{"path": "/home/krishnans/lustre/datasets/bidex_switch/rollouts_1000.hdf5"}],
                [{"path": "/home/krishnans/lustre/datasets/bidex_scissors/rollouts_1000.hdf5", "weight": 0.5}, 
                 {"path": "/home/krishnans/lustre/datasets/bidex_switch/rollouts_1000.hdf5", "weight": 0.5}]
            ],
            value_names=["scissors_switch_1k", "scissors_1k", "switch_1k"]
        )
    elif args.env.startswith("myo"):
        generator.add_param(key="observation.modalities.obs.low_dim", values=[["vec_obs"]], group=-1, name="")
        generator.add_param(key="observation.modalities.obs.rgb", values=[["fixed_camera"]], group=-1, name="")
        datasets_path = os.environ["MYO_DATASET_PATH"]
        if args.mod == "im":
            values = ["all"]
        else:
            values = ["low_dim"]
        generator.add_param(key="train.hdf5_cache_mode", name="", group=-1, values=values)
        generator.add_param(
            key="train.data",
            name="ds",
            group=5,
            values=[
                [
                    {"path": str(p)}
                    for p in list((Path(datasets_path) / args.env).rglob("*.hdf5"))
                ],
            ],
            value_names=["myo"],
        )

    if args.demos is not None:
        generator.add_param(key="train.hdf5_filter_key", name="demos", group=6, values=[f"{demo}_demos" for demo in args.demos])
    return generator


def main(args):

    # make config generator
    generator = make_generator(args)

    # generate jsons and script
    generator.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Script name to generate - will override any defaults
    parser.add_argument("--use_r3m", action="store_true", help="whether or not to use r3m backbone")
    parser.add_argument(
        "--script",
        type=str,
        help="path to output script that contains commands to run the generated training runs",
    )
    parser.add_argument(
        "--mod",
        type=str,
        default="im",
        choices=["im", "ld"],
        help="image or lowdim data",
    )
    parser.add_argument(
        "--env", type=str, default="myo10",
        choices=["bidex_scissors", "bidex_switch", "bidex_bottle", "bidex_mt1",
                 "myo10", "myo5-easy", "myo5-hard"],
        )

    parser.add_argument("--demos", nargs="+", type=int)

    args = parser.parse_args()
    main(args)
