import json
from os.path import join, basename

import numpy as np
from csbdeep.io import save_tiff_imagej_compatible
from n2v_tasks.prefect_task.environment_utils import \
    add_to_slurm_flow_run_table_task, save_slurm_job_info_task, \
    get_slurm_job_info_task, save_prefect_context_task, \
    get_prefect_context_task, save_system_information_task, save_conda_env_task
from n2v_tasks.prefect_task.path_utils import create_output_dir_task
from n2v_tasks.prefect_task.predict import create_save_dir_task, get_files_task
from n2v_tasks.task.predict import load_model
from prefect import Flow, Parameter, task, unmapped
from prefect.client import Secret
from prefect.executors import DaskExecutor
from prefect.run_configs import LocalRun
from prefect.storage import GitHub
from prefect.tasks.secrets import PrefectSecret
from tifffile import imread


@task(log_stdout=True)
def predict(model_dir, model_name, file, n_tiles, save_dir):
    model = load_model(model_dir=model_dir, model_name=model_name)
    img = imread(file)
    dtype = img.dtype
    iinfo = np.iinfo(dtype)
    pred = np.clip(model.predict(img, axes="YX", n_tiles=n_tiles),
                   a_min=iinfo.min,
                   a_max=iinfo.max).astype(dtype)
    save_tiff_imagej_compatible(join(save_dir, basename(file)), pred,
                                axes="YX")


with Flow("Predict N2V [2D]",
          run_config=LocalRun(labels=["SLURM"],
                              working_dir=Secret(
                                  "prefect-slurm-logs").get())) as \
        flow:
    model_name = Parameter("model_name",
                           default="model_name")
    model_dir = Parameter("model_dir", default="/path/to/model/dir")
    input_dir = Parameter("input_dir", default="/path/to/input/data")
    filter = Parameter("filter", default="*.tif")
    save_data_path = Parameter("save_data_path",
                               default="/path/to/save/results")
    n_tiles = Parameter("n_tiles", default=[1, 1])
    group = Parameter("group", default="gmicro")
    user = Parameter("user", default="buchtimo")
    name = Parameter("name", default="run-name")

    files = get_files_task(input_dir=input_dir,
                           filter=filter)

    output_dir = create_output_dir_task(save_data_path=save_data_path,
                                        group=group,
                                        user=user,
                                        name=name)

    save_dir = create_save_dir_task(output_dir=output_dir,
                                    model_name=model_name)

    predict.map(model_dir=unmapped(model_dir),
                model_name=unmapped(model_name),
                file=files,
                n_tiles=unmapped(n_tiles),
                save_dir=unmapped(save_dir))

    save_conda_env_task(output_dir=output_dir)
    save_system_information_task(output_dir=output_dir)
    context_dict = get_prefect_context_task()
    save_prefect_context_task(output_dir=output_dir, context_dict=context_dict)
    slurm_info = get_slurm_job_info_task()
    save_slurm_job_info_task(output_dir=output_dir,
                             slurm_info_dict=slurm_info)
    response = add_to_slurm_flow_run_table_task(prefect_context=context_dict,
                                                slurm_info_dict=slurm_info,
                                                base_key=PrefectSecret(
                                                    "airtable_prefect_base_id"),
                                                table_name=PrefectSecret(
                                                    "airtable-prefect-SLURM-Flow-Runs"),
                                                api_key=PrefectSecret(
                                                    "airtable-api-key-buchtimo"))

flow.storage = GitHub(
    repo="fmi-faim/prefect-workflows",
    path="n2v_flows/2D/predict2D_n2v_2D.py",
    ref="n2v_flows-v0.1.0",
    access_token_secret="github-access-token_buchtimo"
)

slurm_config_path = join(Secret("prefect-slurm-configs").get(),
                         "prefect_slurm_gpu.json")
with open(slurm_config_path, "r") as f:
    config = json.load(f)

config["n_workers"] = 1

flow.executor = DaskExecutor(
    cluster_class="dask_jobqueue.SLURMCluster",
    cluster_kwargs=config,
)
