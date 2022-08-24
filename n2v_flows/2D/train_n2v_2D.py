import json
from os.path import join

from n2v_tasks.prefect_task.environment_utils import \
    save_system_information_task, save_conda_env_task, \
    get_prefect_context_task, save_prefect_context_task, \
    get_slurm_job_info_task, save_slurm_job_info_task, \
    add_to_slurm_flow_run_table_task
from n2v_tasks.prefect_task.path_utils import create_output_dir_task
from n2v_tasks.prefect_task.train import build_model_task, \
    load_train_data_task, train_model_task
from prefect import Flow, Parameter
from prefect.client import Secret
from prefect.executors import DaskExecutor
from prefect.run_configs import LocalRun
from prefect.storage import GitHub
from prefect.tasks.secrets import PrefectSecret

with Flow("Train N2V [2D]",
          run_config=LocalRun(labels=["SLURM"],
                              working_dir=Secret("prefect-slurm-logs").get(),
                              env={"WANDB_CACHE_DIR": Secret(
                                  "wandb-cache-dir").get()})) as flow:
    train_data = Parameter("train_data",
                           default="/path/to/n2v_train.npz")
    val_data = Parameter("val_data",
                         default="/path/to/n2v_val.npz")
    model_name = Parameter("model_name",
                           default="model_name")
    save_data_path = Parameter("save_data_path", default="/path/to/save")
    epochs = Parameter("epochs", default=200)
    batch_size = Parameter("batch_size", default=128)
    group = Parameter("group", default="gmicro")
    user = Parameter("user", default="buchtimo")
    name = Parameter("name", default="run-name")
    wandb_project = Parameter("wandb_project", default="faim-n2v_flows")
    wandb_entity = Parameter("wandb_entity", default="entity")

    output_dir = create_output_dir_task(save_data_path=save_data_path,
                                        group=group,
                                        user=user,
                                        name=name)

    X, X_val = load_train_data_task(train_data=train_data,
                                    val_data=val_data)

    model = build_model_task(output_dir=output_dir,
                             model_name=model_name,
                             X=X,
                             epochs=epochs,
                             batch_size=batch_size,
                             group=group,
                             user=user,
                             name=name,
                             wandb_project=wandb_project,
                             wandb_entity=wandb_entity)

    train_model_task(model=model,
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
    path="n2v_flows/2D/train_n2v_2D.py",
    ref="n2v_flows-v0.1.1",
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
