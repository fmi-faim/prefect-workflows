from datetime import datetime
from os.path import splitext, join, dirname, basename
from pathlib import Path
from typing import Dict, List

import numpy as np
import pkg_resources
from cpr.Serializer import cpr_serializer
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from eicm.estimator.polynomial_fit import polynomial_fit
from eicm.estimator.utils import normalize_matrix
from faim_prefect.prefect import get_prefect_context
from prefect import task, flow
from prefect.context import get_run_context
from prefect_dask import DaskTaskRunner

from eicm_flows.fit_gaussian_estimation import load_tiff


@task(cache_key_fn=task_input_hash)
def fit_polynomial(shading_reference: Path, polynomial_degree: int,
                                       order: int):

    n, ext = splitext(basename(shading_reference))
    save_path = join(dirname(shading_reference), f"{n}_poly-fit{ext}")

    resolution, metadata, data = load_tiff(path=shading_reference)

    matrix = ImageTarget.from_path(save_path,
                                   metadata=metadata,
                                   resolution=resolution)

    fit, _ = polynomial_fit(mip=data,
                                   polynomial_degree=polynomial_degree,
                                   order=order)

    matrix.set_data(normalize_matrix(fit).astype(np.float32))

    return matrix


@task(cache_key_fn=task_input_hash)
def write_poly_fit_info_md(matrix: ImageTarget,
                           name: str,
                           shading_reference: Path,
                           polynomial_degree: int,
                           order: int,
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
           f"The computed illumination matrix ({file_name}) is the best " \
           f"polynomial fit to the provided shading reference.\n" \
           f"\n" \
           f"## Parameters\n" \
           f"* `shading_reference`: {shading_reference}\n" \
           f"* `polynomial_degree`: {polynomial_degree}\n" \
           f"* `order`: {order}\n" \
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
    name="EICM with Polynomial Fit",
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
def eicm_polynomial_fit(
        shading_references: List[Path] = [Path("/path/to/shading_reference")],
        polynomial_degree: int = 4,
        order: int = 4,
):
    run_context = get_run_context()
    context = get_prefect_context(run_context)
    flow_name = run_context.flow.name
    for shading_reference in shading_references:
        matrix = fit_polynomial.submit(
            shading_reference=shading_reference,
            polynomial_degree=polynomial_degree,
            order=order)

        write_poly_fit_info_md.submit(matrix=matrix,
                                      name=flow_name,
                                      shading_reference=shading_reference,
                                      polynomial_degree=polynomial_degree,
                                      order=order,
                                      context=context)
