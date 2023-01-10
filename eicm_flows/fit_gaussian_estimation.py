import json
import os
from datetime import datetime
from os.path import splitext, join, basename
from typing import Dict

import pkg_resources
from cpr.Serializer import cpr_serializer
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from eicm.estimator.gaussian2D_fit import get_coords, fit_gaussian_2d, \
    compute_fitted_matrix
from eicm.estimator.utils import normalize_matrix
from faim_prefect.block.choices import Choices
from faim_prefect.prefect import get_prefect_context
from prefect import task, flow
from prefect.blocks.system import String
from prefect.context import get_run_context
from prefect.filesystems import LocalFileSystem
from prefect_dask import DaskTaskRunner
from tifffile import TiffFile

from eicm_flows.shading_reference_yokogawa import Microscopes


def load_tiff(path: str):
    with TiffFile(path) as tiff:
        resolution = tiff.pages[0].resolution
        metadata = json.loads(tiff.pages[0].description)
        data = tiff.asarray()
    return resolution, metadata, data


@task(cache_key_fn=task_input_hash)
def estimate_correction_matrix(shading_reference: str, output_dir: str):
    n, ext = splitext(basename(shading_reference))
    save_path = join(output_dir, f"{n}_gaussian-fit{ext}")

    resolution, metadata, data = load_tiff(path=shading_reference)

    matrix = ImageTarget.from_path(save_path,
                                   metadata=metadata,
                                   resolution=resolution)

    coords = get_coords(data)
    popt, pcov = fit_gaussian_2d(data, coords)

    matrix.set_data(normalize_matrix(compute_fitted_matrix(coords=coords,
                                                           ellipsoid_parameters=popt,
                                                           shape=data.shape)))


    return matrix, popt


@task(cache_key_fn=task_input_hash)
def write_gaussian_fit_info_md(matrix: ImageTarget,
                               name: str,
                               shading_reference: str,
                               microscope: str,
                               group: str,
                               output_dir: str,
                               amplitude: float,
                               background: float,
                               mu_x: float,
                               mu_y: float,
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
           f"fit of a 2D Gaussian to the provided shading reference.\n" \
           f"\n" \
           f"### Fitted Gaussian\n" \
           f"* Amplitude: {amplitude}\n" \
           f"* Background: {background}\n" \
           f"* Centroid (X, Y): ({mu_x}, {mu_y})\n" \
           f"\n" \
           f"## Parameters\n" \
           f"* `shading_reference`: {shading_reference}\n" \
           f"* `microscope`: {microscope}\n" \
           f"* `group`: {group}\n" \
           f"* `output_dir`: {output_dir}\n" \
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
      result_storage="local-file-system/test-eicm",
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
                  String.load("log-slurm-job-to-airtable-cmd").value
              ],
          },
          adapt_kwargs={
              "minimum": 1,
              "maximum": 2,
          },
      )
)
def eicm_gaussian_fit(
        shading_reference: str = "/path/to/shading_reference",
        microscope: Microscopes = "CV7000",
        group: Choices.load("fmi-groups").get() = "gmicro",
        output_dir: str = LocalFileSystem.load("tungsten-gmicro-hcs").basepath
):
    output_dir = join(output_dir, group, microscope, "Maintenance",
                      "eicm")

    os.makedirs(output_dir, exist_ok=True)

    matrix, popt = estimate_correction_matrix.submit(mip_path=shading_reference,
                                                     output_dir=output_dir).result()

    write_gaussian_fit_info_md.submit(matrix=matrix,
                                      name=get_run_context().flow.name,
                                      shading_reference=shading_reference,
                                      microscope=microscope,
                                      group=group,
                                      output_dir=output_dir,
                                      amplitude=popt[0],
                                      background=popt[1],
                                      mu_x=popt[2],
                                      mu_y=popt[3],
                                      context=get_prefect_context(get_run_context()))


