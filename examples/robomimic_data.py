import h5py, yaml, json
import numpy as np
import glob


TASK_SETS = {
    "allegro": ["spray_bottle", "scissors", "stapler"],
    "bidex": ["scissors", "switch", "bottle"],
    "myodex": ["reach", "push", "pick_and_place", "stack"],
}



def merge_hdf5_files(source_files, target_file):
    """
    Merge multiple HDF5 files into a single file, ensuring that "data/demo_X" keys
    are unique across the merged file.

    Args:
        source_files (list of str): List of paths to source HDF5 files to merge.
        target_file (str): Path to the target HDF5 file to create.
    """
    with h5py.File(target_file, "w") as target_h5:
        if "data" not in target_h5.keys():
            target_h5.create_group("data")  # Ensure the 'data' group exists in the target file
        for source_file in source_files:
            with h5py.File(source_file, "r") as source_h5:
                # Iterate over each "demo_X" group in the source file
                for key in source_h5["data"].keys():
                    # Extract the demo number from the key and increment it by the current offset
                    new_demo_key = f'demo_{len(target_h5["data"].keys())}'

                    # Copy the group to the new file with the updated demo key
                    source_h5.copy(f"data/{key}", target_h5["data"], new_demo_key)


def convert_camera_obs_hwc(source_file, target_file):
    """
    Convert all observations with the name "camera" in the key from (c, h, w) to (h, w, c) shape ordering,
    and change dtype to np.uint8 for a single source file, and map it to a target file.

    Args:
        source_file (str): Path to the source HDF5 file.
        target_file (str): Path to the target HDF5 file to create.
    """
    with h5py.File(source_file, "r") as source_h5:
        with h5py.File(target_file, "w") as target_h5:
            # Iterate over each "demo_X" group in the source file
            for demo_key in source_h5["data"].keys():
                demo_group = source_h5["data"][demo_key]
                target_demo_group = target_h5.require_group(f"data/{demo_key}")

                # Check if the "obs" group exists in the demo group
                for key in demo_group:
                    if key == "obs":
                        obs_group = demo_group["obs"]
                        # Add "obs" group to the target_demo_group if it doesn't already exist
                        target_demo_group.create_group("obs")
                        # Iterate over each item in the "obs" group
                        for item_key in obs_group.keys():
                            item = obs_group[item_key]

                            # Check if the item is an observation with "camera" in the key
                            if "camera" in item_key:
                                if len(item.shape) == 4 and item.shape[-1] != 3:  # Assuming shape is (timesteps, c, h, w)
                                    # Convert (c, h, w) to (h, w, c)
                                    item = item[:].transpose(0, 2, 3, 1)
                                # change dtype to np.uint8
                                item = item.astype(np.uint8)
                                target_demo_group["obs"].create_dataset(item_key, data=item)
                            else:
                                # For other items within "obs", just copy them as they are
                                obs_group.copy(item_key, target_demo_group["obs"])
                    else:
                        demo_group.copy(f"{key}", target_demo_group, key)
                        # target_demo_group.create_dataset(key, data=demo_group[key])

def convert_articulate_to_robosuite(dataset_path, next_obs=False, config_path=None):
    config = {}
    if config_path is not None:
        config = yaml.safe_load(open(config_path, "r"))

    with h5py.File(dataset_path, mode="a") as data:
        data.attrs["env_args"] = json.dumps(config)
        for ep in data["data"].keys():
            data["data/{}".format(ep)].attrs["num_samples"] = data["data/{}".format(ep)]["actions"].shape[0]
        # Create next_obs group for every demo in data, and copy obs/{key}.data[1:] to this group
        for demo_key in data["data"].keys():
            demo_group = data["data"][demo_key]
            # skip if "next_obs" in demo_group
            if "next_obs" in demo_group and not next_obs:
                continue
            next_obs_group = demo_group.create_group("next_obs")
            
            for obs_key in demo_group["obs"].keys():
                obs_data = demo_group["obs"][obs_key]
                next_obs_group.create_dataset(obs_key, data=obs_data[1:])


def add_mask(dataset_paths):
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "a") as data:
            success_demos = []
            for ep in data["data"].keys():
                if data[f"data/{ep}"].attrs['num_samples'] > 0:
                    success_demos.append(ep.encode('utf-8'))  # HDF5 requires bytes for strings

            if "traj_success" not in data["mask"].keys():
                data.create_group("mask")
                data["mask"].create_dataset("traj_success", data=np.array(success_demos, dtype=str))
            else:
                print(f"skipping mask for {dataset_path}, traj_success already exists")


def add_task_id(dataset_paths, task_names, task_set='bidex'):
    for dataset_path, task_name in zip(dataset_paths, task_names):
        task_id = TASK_SETS[task_set].index(task_name)
        with h5py.File(dataset_path, "a") as data:
            for ep in data["data"].keys():
                task_idx = np.ones(data["data/{}".format(ep)].attrs["num_samples"]) * task_id
                if "task_id" not in data["data/{}/obs".format(ep)].keys():
                    data["data/{}/obs".format(ep)].create_dataset("task_id", data=task_idx)
                else:
                    print(f"skipping task_id for {dataset_path}, ep {ep}")
                    assert data["data/{}/obs".format(ep)]["task_id"][:][0] == task_id, (
                        f"expected task_id for {dataset_path} to be {task_id}, "
                        f"but got {data['data/{}/obs'.format(ep)]['task_id'][:][0]}"
                    )


def main(args):
    if "*" in args.dataset_path:
        dataset_paths = glob.glob(args.dataset_path)
    else:
        dataset_paths = [x.strip() for x in args.dataset_path.split(",")]
    if args.merge:
        # check if wildcard in datasetpath, and if so, evaluate and pass in a list of datasets
        merge_hdf5_files(dataset_paths, args.target_path)
    if args.add_task_id:
        task_names = []
        for p in dataset_paths:
            if "scissors" in p:
                task_names.append("scissors")
            elif "switch" in p:
                task_names.append("switch")
            elif "bottle" in p:
                task_names.append("bottle")
        add_task_id(dataset_paths, task_names)
    if args.convert:
        convert_articulate_to_robosuite(args.dataset_path, args.next_obs, args.config_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--target_path", type=str, default="")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--add_task_id", action="store_true")
    parser.add_argument("--next-obs", action="store_true")
    main(parser.parse_args())
