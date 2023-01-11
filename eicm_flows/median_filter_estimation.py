from datetime import datetime
from os.path import splitext, join, basename, dirname
from typing import Dict

import numpy as np
import pkg_resources
from cpr.Serializer import cpr_serializer
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from eicm.estimator.utils import normalize_matrix
from faim_prefect.prefect import get_prefect_context
from prefect import task, flow
from prefect.context import get_run_context
from prefect_dask import DaskTaskRunner
from scipy.ndimage import median_filter

from eicm_flows.fit_gaussian_estimation import load_tiff


@task(cache_key_fn=task_input_hash)
def median_filter_task(shading_reference: str, filter_size: int = 3):
    n, ext = splitext(basename(shading_reference))
    save_path = join(dirname(shading_reference), f"{n}_median-filtered{ext}")

    resolution, metadata, data = load_tiff(path=shading_reference)

    matrix = ImageTarget.from_path(save_path,
                                   metadata=metadata,
                                   resolution=resolution)

    matrix.set_data(normalize_matrix(median_filter(data,
                                                   size=filter_size)).astype(
        np.float32))

    return matrix


@task(cache_key_fn=task_input_hash)
def write_median_filter_info_md(matrix: ImageTarget,
                                name: str,
                                shading_reference: str,
                                filter_size: int,
                                context: Dict):
    date = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
    eicm_version = pkg_resources.get_distribution("eicm").version
    flow_repo = "https://github.com/fmi-faim/prefect-workflows/blob/main/eicm_flows"

    file_name = basename(matrix.get_path())
    save_path = splitext(matrix.get_path())[0] + ".md"

    text = f"# {name}\n" \
           f"Source: [{flow_repo}]({flow_repo})\n" \
           f"Date: {date}\n" \
           f"\n" \
           f"`{name}` is a service provided by the Facility for Advanced " \
           f"Imaging and Microscopy (FAIM) at FMI for biomedical research. " \
           f"Consult with FAIM on appropriate usage.\n" \
           f"\n" \
           f"## Summary\n" \
           f"The computed illumination matrix ({file_name}) is the " \
           f"normalized (to max) median filtered shading reference.\n" \
           f"\n" \
           f"## Parameters\n" \
           f"* `shading_reference`: {shading_reference}\n" \
           f"* `filter_size`: {filter_size}\n" \
           f"\n" \
           f"## Packages\n" \
           f"* [https://github.com/fmi-faim/eicm](" \
           f"https://github.com/fmi-faim/eicm): v{eicm_version}\n" \
           f"\n" \
           f"## Prefect Context\n" \
           f"{str(context)}"

    with open(save_path, "w") as f:
        f.write(text)


@flow(
    name="EICM with Median Filter",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    result_storage="local-file-system/eicm",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "queue": "cpu_long",
            "cores": 1,
            "processes": 1,
            "memory": "4 GB",
            "walltime": "1:00:00",
            "job_extra_directives": [
                "--ntasks=1",
                "--output=/tungstenfs/scratch/gmicro_share/_prefect/slurm/output/%j.out",
            ],
            "worker_extra_args": [
                "--lifetime",
                "60m",
                "--lifetime-stagger",
                "10m",
            ],
            "job_script_prologue": [
                "conda run -p /tungstenfs/scratch/gmicro_share/_prefect/miniconda3/envs/airtable python /tungstenfs/scratch/gmicro_share/_prefect/airtable/log-slurm-job.py --config /tungstenfs/scratch/gmicro/_prefect/airtable/slurm-job-log.ini"
            ],
        },
        adapt_kwargs={
            "minimum": 1,
            "maximum": 2,
        },
    )
)
def eicm_median_filter(
        shading_reference: str = "/path/to/shading_reference",
        filter_size: int = 3,
):
    matrix = median_filter_task.submit(shading_reference=shading_reference,
                                       filter_size=filter_size).result()

    write_median_filter_info_md.submit(matrix=matrix,
                                       name=get_run_context().flow.name,
                                       shading_reference=shading_reference,
                                       filter_size=filter_size,
                                       context=get_prefect_context(
                                           get_run_context())
                                       )
