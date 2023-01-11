from faim_prefect.block.choices import Choices
from prefect import flow, get_run_logger
from prefect.filesystems import LocalFileSystem
from prefect_dask import DaskTaskRunner
from pydantic import BaseModel

from eicm_flows.fit_gaussian_estimation import eicm_gaussian_fit
from eicm_flows.fit_polynomial_estimation import eicm_polynomial_fit
from eicm_flows.median_filter_estimation import eicm_median_filter
from eicm_flows.shading_reference_yokogawa import Microscopes, \
    create_shading_reference_yokogawa

GROUPS = Choices.load("fmi-groups").get()

class RawData(BaseModel):
    input_dir: str = "/tungstenfs/scratch/gmicro/reitsabi/CV7000/Flatfield_correction_tests/20221221-Field-illumination-QC_20221221_143935/Dyes_60xW_Cellvis/"
    microscope: Microscopes = "CV7000"
    z_plane: int = 33
    group: GROUPS = GROUPS.gmicro
    output_dir: str = LocalFileSystem.load(
        "tungsten-gmicro-hcs").basepath


class MedianFilter(BaseModel):
    apply: bool = True
    filter_size: int = 3


class GaussianFit(BaseModel):
    apply: bool = True


class PolynomialFit(BaseModel):
    apply: bool = True
    polynomial_degree: int = 4
    order: int = 4


@flow(
    name="EICM All [Yokogawa]",
    cache_result_in_memory=False,
    persist_result=True,
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
def eicm_all_yokogawa(
        raw_data: RawData = RawData(),
        median_filter: MedianFilter = MedianFilter(),
        gaussian_fit: GaussianFit = GaussianFit(),
        polynomial_fit: PolynomialFit = PolynomialFit()
):
    references = create_shading_reference_yokogawa(
        input_dir=raw_data.input_dir,
        microscope=raw_data.microscope,
        z_plane=raw_data.z_plane,
        group=raw_data.group.value,
        output_dir=raw_data.output_dir
    )

    for reference in references:
        if median_filter.apply:
            eicm_median_filter(shading_reference=reference,
                               filter_size=median_filter.filter_size)

        if gaussian_fit.apply:
            eicm_gaussian_fit(shading_reference=reference)

        if polynomial_fit.apply:
            eicm_polynomial_fit(shading_reference=reference,
                                polynomial_degree=polynomial_fit.polynomial_degree,
                                order=polynomial_fit.order)
