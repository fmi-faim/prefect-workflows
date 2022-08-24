import json
import random
from glob import glob
from os.path import join

import numpy as np
import prefect
from n2v_tasks.prefect_task.environment_utils import save_conda_env_task, \
    save_system_information_task, save_prefect_context_task, \
    get_prefect_context_task, save_slurm_job_info_task, \
    get_slurm_job_info_task, \
    add_to_slurm_flow_run_table_task
from n2v_tasks.prefect_task.generate_train_data import extract_patches_task, \
    save_train_val_data_task
from n2v_tasks.prefect_task.path_utils import create_output_dir_task
from prefect import task
from prefect.client import Secret
from prefect.core import Flow, Parameter
from prefect.executors import DaskExecutor
from prefect.run_configs import LocalRun
from prefect.storage import GitHub
from prefect.tasks.secrets import PrefectSecret
from tifffile import imread


@task()
def load_imgs_from_directory(data_dir: str,
                             filter: str):
    logger = prefect.context.get("logger")

    files = glob(join(data_dir, filter))

    imgs = []
    for f in files:
        img = imread(f).astype(np.float32)
        logger.info(f"Loaded {f} with shape = {img.shape}.")
        for s in img:
            imgs.append(s[np.newaxis, :, :, np.newaxis])

    logger.info(f"Loaded {len(imgs)} images for N2V training.")
    if len(imgs) < 2:
        logger.error("At least two images are required for training.")

    random.Random(42).shuffle(imgs)

    return imgs


with Flow("N2V Data Generation [2D+T - 2D]",
          run_config=LocalRun(labels=["SLURM"],
                              working_dir=Secret(
                                  "prefect-slurm-logs").get())) as flow:
    data_dir = Parameter("data_dir", default="/path/to/data")
    name = Parameter("name", default="run-name")
    filter = Parameter("filter", default="*.tif")
    patch_shape = Parameter("patch_shape", default=[96, 96])
    num_patches_per_img = Parameter("num_patches_per_img", default=8)
    save_data_path = Parameter("save_data_path", default="/path/to/save")
    prefix = Parameter("prefix", default="prefix")
    group = Parameter("group", default="gmicro")
    user = Parameter("user", default="buchtimo")

    imgs = load_imgs_from_directory(data_dir=data_dir,
                                    filter=filter)

    X, X_val = extract_patches_task(imgs=imgs,
                                    num_patches_per_img=num_patches_per_img,
                                    patch_shape=patch_shape)

    output_dir = create_output_dir_task(save_data_path=save_data_path,
                                        group=group,
                                        user=user,
                                        name=name)

    save_train_val_data_task(output_dir=output_dir,
                             prefix=prefix,
                             X=X,
                             X_val=X_val)

    save_conda_env_task(output_dir=output_dir)
    save_system_information_task(output_dir=output_dir)
    context_dict = get_prefect_context_task()
    save_prefect_context_task(output_dir=output_dir, context_dict=context_dict)
    slurm_info = get_slurm_job_info_task()
    save_slurm_job_info_task(output_dir=output_dir,
                             slurm_info_dict=slurm_info)
    row = add_to_slurm_flow_run_table_task(prefect_context=context_dict,
                                           slurm_info_dict=slurm_info,
                                           base_key=PrefectSecret(
                                               "airtable_prefect_base_id"),
                                           table_name=PrefectSecret(
                                               "airtable-prefect-SLURM-Flow-Runs"),
                                           api_key=PrefectSecret(
                                               "airtable-api-key-buchtimo"))

flow.storage = GitHub(
    repo="fmi-faim/prefect-workflows",
    path="n2v_flows/2D/generate_train_data_n2v_2D+T_to_2D.py",
    ref="n2v_flows-v0.1.1",
    access_token_secret="github-access-token_buchtimo"
)

slurm_config_path = join(Secret("prefect-slurm-configs").get(),
                         "prefect_slurm_tiny-cpu.json")
with open(slurm_config_path, "r") as f:
    config = json.load(f)

flow.executor = DaskExecutor(
    cluster_class="dask_jobqueue.SLURMCluster",
    cluster_kwargs=config,
)
