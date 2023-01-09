import os
from glob import glob
from os.path import basename, join
from typing import Literal

import numpy as np
from cpr.Serializer import cpr_serializer
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from eicm.preprocessing.yokogawa import get_metadata, create_table, \
    build_field_stacks_for_channels, subtract_dark_images, \
    compute_median_projection, get_output_name
from prefect import flow, task
from prefect_dask import DaskTaskRunner

Microscopes = Literal[
    "CV7000",
    "CV8000"
]


@task(cache_key_fn=task_input_hash)
def create_shading_reference(input_dir: str, z_plane: int, output_dir: str):

    if input_dir.endswith("/"):
        # Remove slash at the end
        input_dir = input_dir[:-1]

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
        out_img = ImageTarget.from_path(join(output_dir, out_name),
                                        resolution=[1 / px_size, 1 / px_size],
                                        metadata={"axes": "YX",
                                                  "unit": px_unit}
                                        )
        out_img.set_data(projection.astype(np.float32))
        references.append(out_img)

    return tuple(references)


@flow(name="Create Shading Reference [Yokogawa]",
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
                  "--output=/tungstenfs/scratch/gmicro_share/_prefect/slurm/gfriedri-em-alignment-flows/output/%j.out",
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
def create_shading_reference_yokogawa(input_dir: str =
                                      "/tungstenfs/scratch/gmicro/reitsabi/CV7000/Flatfield_correction_tests/20221221-Field-illumination-QC_20221221_143935/Dyes_60xW_Cellvis/",
                                      microscope: Microscopes = "CV7000",
                                      z_plane: int = 33,
                                      output_dir: str = "/tungstenfs/scratch/gmicro_hcs/gmicro/"):

    final_out_dir = join(output_dir, microscope, "Maintenance",
                         "Shading_Reference")

    os.makedirs(final_out_dir, exist_ok=True)

    references = create_shading_reference.submit(input_dir=input_dir,
                                                 z_plane=z_plane,
                                                 output_dir=final_out_dir)

    return references
