import os

from robomimic.scripts.config_gen.helper import *
from pathlib import Path


MYO_TASK_SET = {
    "myo10": [
        "myo-key-turn",
        "myo-key-turn-hard",
        "myo-obj-hold",
        "myo-obj-hold-hard",
        "myo-pen-twirl",
        "myo-pen-twirl-hard",
        "myo-pose",
        "myo-pose-hard",
        "myo-reach",
        "myo-reach-hard",
    ],
    "myo5-easy": ["myo-key-turn", "myo-obj-hold", "myo-pen-twirl", "myo-pose", "myo-reach"],
    "myo5-hard": ["myo-key-turn-hard", "myo-obj-hold-hard", "myo-pen-twirl-hard", "myo-pose-hard", "myo-reach-hard"],
}

BIDEX_TASK_SET = {
    "switch": "/home/krishnans/lustre/datasets/bidex_switch/rollouts_1000.hdf5",
    "scissors": "/home/krishnans/lustre/datasets/bidex_scissors/rollouts_1000.hdf5",
}

def make_generator_helper(args):
    algo_name_short = "act"
    generator = get_generator(
        algo_name="act",
        config_file=os.path.join(base_path, "robomimic/exps/templates/act_myo_r3m.json"),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[1000],
    )

    generator.add_param(
        key="train.batch_size",
        name="",
        group=-1,
        values=[64],
    )

    generator.add_param(
        key="train.max_grad_norm",
        name="",
        group=-1,
        values=[100.0],
    )

    if args.env.startswith("myo"):
        datasets_path = os.environ["MYO_DATASET_PATH"]
        task_set = MYO_TASK_SET[args.env]
        if args.mod == "im":
            values = ["all"]
        else:
            values = ["low_dim"]
        generator.add_param(key="train.hdf5_cache_mode", name="", group=-1, values=values)
        generator.add_param(key="train.hdf5_filter_key", name="", group=-1, values=[args.nr])
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [
                    {"path": str(p)}
                    for p in list((Path(datasets_path) / args.env).rglob("*.hdf5"))
                ],
            ],
            value_names=["myo"],
        )
        if args.goal_mode is not None:
            generator.add_param(
                key="train.goal_mode",
                name="",
                group=-1,
                values=args.goal_mode,
            )
            if args.mod == "ld":
                generator.add_param(
                    key="observation.modalities.goal.low_dim",
                    name="",
                    group=-1,
                    values=[["vec_obs"]],
                )
            else: 
                generator.add_param(
                    key="observation.modalities.goal.rgb",
                    name="",
                    group=-1,
                    values=[["fixed_camera"]],
                )
        generator.add_param(
            key="train.action_keys",
            name="ac_keys",
            group=-1,
            values=[["action"]],
            value_names=[
                "ac",
            ],
        )
    elif args.env == "r2d2":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [{"path": p} for p in scan_datasets("~/Downloads/example_pen_in_cup", postfix="trajectory_im128.h5")],
            ],
            value_names=[
                "pen-in-cup",
            ],
        )
        generator.add_param(
            key="train.action_keys",
            name="ac_keys",
            group=-1,
            values=[
                [
                    "action/abs_pos",
                    "action/abs_rot_6d",
                    "action/gripper_position",
                ],
            ],
            value_names=[
                "abs",
            ],
        )
    elif args.env == "kitchen":
        raise NotImplementedError
    elif args.env == "square":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [
                    {"path": "TODO.hdf5"},  # replace with your own path
                ],
            ],
            value_names=[
                "square",
            ],
        )

        # update env config to use absolute action control
        generator.add_param(
            key="experiment.env_meta_update_dict",
            name="",
            group=-1,
            values=[{"env_kwargs": {"controller_configs": {"control_delta": False}}}],
        )

        generator.add_param(
            key="train.action_keys",
            name="ac_keys",
            group=-1,
            values=[
                [
                    "action_dict/abs_pos",
                    "action_dict/abs_rot_6d",
                    "action_dict/gripper",
                    # "actions",
                ],
            ],
            value_names=[
                "abs",
            ],
        )

    else:
        raise ValueError

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "../{env}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    return generator


if __name__ == "__main__":
    parser = get_argparser()
    parser.add_argument("--goal_mode", nargs="+", choices=["random"])

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
