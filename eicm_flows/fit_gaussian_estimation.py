import json
from datetime import datetime
from os.path import splitext, join, basename, dirname
from typing import Dict, List

import numpy as np
import pkg_resources
from cpr.Serializer import cpr_serializer
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from eicm.estimator.gaussian2D_fit import get_coords, fit_gaussian_2d, \
    compute_fitted_matrix
from eicm.estimator.utils import normalize_matrix
from faim_prefect.prefect import get_prefect_context
from prefect import task, flow, get_run_logger
from prefect.context import get_run_context
from prefect_dask import DaskTaskRunner
from tifffile import TiffFile


def load_tiff(path: str):
    with TiffFile(path) as tiff:
        try:
            resolution = tiff.pages[0].resolution
        except Exception as e:
            get_run_logger().warning(f"Could not load resolution.\n{e}")
            resolution = [1.0, 1.0]

        try:
            metadata = json.loads(tiff.pages[0].description)
        except Exception as e:
            get_run_logger().warning(f"Could not load metadata.\n{e}")
            metadata = {'axes': 'YX'}

        data = tiff.asarray()
    return resolution, metadata, data


@task(cache_key_fn=task_input_hash)
def estimate_correction_matrix(shading_references: List[str]):
    matrices, popts = [], []

    for shading_reference in shading_references:
        n, ext = splitext(basename(shading_reference))
        save_path = join(dirname(shading_reference), f"{n}_gaussian-fit{ext}")

        resolution, metadata, data = load_tiff(path=shading_reference)

        matrix = ImageTarget.from_path(save_path,
                                       metadata=metadata,
                                       resolution=resolution)

        coords = get_coords(data)
        popt, pcov = fit_gaussian_2d(data, coords)

        matrix.set_data(normalize_matrix(compute_fitted_matrix(coords=coords,
                                                               ellipsoid_parameters=popt,
                                                               shape=data.shape)).astype(
            np.float32))

        matrices.append(matrix), popts.append(popt.tolist())

    return matrices, popts


@task(cache_key_fn=task_input_hash)
def write_gaussian_fit_info_md(matrices: List[ImageTarget],
                               name: str,
                               shading_references: List[str],
                               popts: List[List[float]],
                               context: Dict):
    date = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
    eicm_version = pkg_resources.get_distribution("eicm").version
    flow_repo = "https://github.com/fmi-faim/prefect-workflows/blob/main/eicm_flows"

    for matrix, shading_reference, popt in zip(matrices, shading_references,
                                               popts):
        amplitude = popt[0],
        background = popt[1],
        mu_x = popt[2],
        mu_y = popt[3]
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
               f"fit of a 2D Gaussian to the provided shading reference.\n" \
               f"\n" \
               f"### Fitted Gaussian\n" \
               f"* Amplitude: {amplitude}\n" \
               f"* Background: {background}\n" \
               f"* Centroid (X, Y): ({mu_x}, {mu_y})\n" \
               f"\n" \
               f"## Parameters\n" \
               f"* `shading_reference`: {shading_reference}\n" \
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
    name="EICM with Gaussian Fit",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    result_storage="local-file-system/eicm",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "queue": "cpu_long",
            "cores": 4,
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
def eicm_gaussian_fit(
        shading_references: List[str] = ["/path/to/shading_reference"],
):
    matrices, popts = estimate_correction_matrix.submit(
        shading_references=shading_references).result()

    write_gaussian_fit_info_md.submit(matrices=matrices,
                                      name=get_run_context().flow.name,
                                      shading_references=shading_references,
                                      popts=popts,
                                      context=get_prefect_context(
                                          get_run_context()))

    return [m.get_path() for m in matrices]
