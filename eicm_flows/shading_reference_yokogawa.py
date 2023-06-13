import os
from datetime import datetime
from enum import Enum
from glob import glob
from os.path import basename, join, splitext
from pathlib import Path
from typing import Literal, Tuple, Dict

import numpy as np
from cpr.Serializer import cpr_serializer
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from eicm.preprocessing.yokogawa import get_metadata, create_table, \
    build_field_stacks_for_channels, subtract_dark_images, \
    compute_median_projection, get_output_name
from faim_prefect.block.choices import Choices
from faim_prefect.prefect import get_prefect_context
from prefect import flow, task, get_run_logger
from prefect.context import get_run_context
from prefect.filesystems import LocalFileSystem
import pkg_resources
from prefect_dask import DaskTaskRunner

Microscopes = Literal[
    "CV7000",
    "CV8000"
]


@task(cache_key_fn=task_input_hash)
def create_shading_reference(input_dir: Path, z_plane: int, output_dir: Path):

    acq_date, px_size, px_unit, channels = get_metadata(input_dir=input_dir)
    plate_name = basename(input_dir)

    files = glob(join(input_dir, plate_name + "*.tif"))
    table = create_table(files=files, plate_name=plate_name)

    channel_stacks = build_field_stacks_for_channels(table=table,
                                                     z_plane=z_plane)

    dark_img_subtracted = subtract_dark_images(stacks=channel_stacks,
                                               channel_metadata=channels,
                                               input_dir=input_dir)

    projections = compute_median_projection(stacks=dark_img_subtracted)

    references = []
    for ch, projection in projections.items():
        out_name = get_output_name(acquistion_date=acq_date,
                                   channel=channels[str(int(ch[1:]))])
        final_out_dir = join(output_dir, acq_date)
        os.makedirs(final_out_dir, exist_ok=True)
        out_img = ImageTarget.from_path(join(final_out_dir, out_name),
                                        resolution=[1e4 / px_size,
                                                    1e4 / px_size],
                                        metadata={"axes": "YX",
                                                  "PhysicalSizeX": px_size,
                                                  "PhysicalSizeXUnit": px_unit,
                                                  "PhysicalSizeY": px_size,
                                                  "PhysicalSizeYUnit": px_unit,}
                                        )
        out_img.set_data(projection.astype(np.float32))
        references.append(out_img)

    return tuple(references)


@task(cache_key_fn=task_input_hash)
def write_info_md(references: Tuple[ImageTarget],
                  name: str,
                  input_dir: Path, z_plane: int, microscope: str, group: str,
                  output_dir: Path, context: Dict):
    date = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
    eicm_version = pkg_resources.get_distribution("eicm").version
    flow_repo = "https://github.com/fmi-faim/prefect-workflows/blob/main/eicm_flows"

    for reference in references:
        file_name = basename(reference.get_path())
        save_path = splitext(reference.get_path())[0] + ".md"

        text = f"# {name}\n" \
               f"Source: [{flow_repo}]({flow_repo})\n" \
               f"Date: {date}\n" \
               f"\n" \
               f"`{name}` is a service provided by the Facility for Advanced " \
               f"Imaging and Microscopy (FAIM) at FMI for biomedical research. " \
               f"Consult with FAIM on appropriate usage.\n" \
               f"\n" \
               f"## Summary\n" \
               f"The computed shading reference ({file_name}) is the median " \
               f"projection over n background (dark image) subtracted positions " \
               f"in the selected Z-plane.\n" \
               f"\n" \
               f"## Parameters\n" \
               f"* `input_dir`: {input_dir}\n" \
               f"* `microscope`: {microscope}\n" \
               f"* `z_plane`: {z_plane}\n" \
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

GROUPS = Choices.load("fmi-groups").get()

@flow(name="Create Shading Reference [Yokogawa]",
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
      ))
def create_shading_reference_yokogawa(input_dir: Path =
                                      Path("/tungstenfs/scratch/gmicro/reitsabi/CV7000/Flatfield_correction_tests/20221221-Field-illumination-QC_20221221_143935/Dyes_60xW_Cellvis/"),
                                      microscope: Microscopes = "CV7000",
                                      z_plane: int = 33,
                                      group: GROUPS = GROUPS.gmicro,
                                      output_dir: Path =
                                      Path(LocalFileSystem.load(
                                          "tungsten-gmicro-hcs").basepath)):
    output_dir_ = join(output_dir, group.value, microscope, "Maintenance",
                      "Shading_Reference")

    os.makedirs(output_dir, exist_ok=True)

    references = create_shading_reference.submit(
        input_dir=input_dir,
        z_plane=z_plane,
        output_dir=output_dir_)

    context = get_prefect_context(get_run_context())
    write_info_md.submit(references, name=get_run_context().flow.name,
                         input_dir=input_dir, z_plane=z_plane,
                         microscope=microscope, group=group.value,
                         output_dir=output_dir, context=context)

    reference_paths = [ref.get_path() for ref in references.result()]
    return reference_paths
