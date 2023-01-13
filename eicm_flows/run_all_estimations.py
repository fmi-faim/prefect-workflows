from pathlib import Path
from typing import List

from prefect import flow
from prefect_dask import DaskTaskRunner
from pydantic import BaseModel

from eicm_flows.fit_gaussian_estimation import eicm_gaussian_fit
from eicm_flows.fit_polynomial_estimation import eicm_polynomial_fit
from eicm_flows.median_filter_estimation import eicm_median_filter


class RawData(BaseModel):
    shading_references: List[Path] = \
        [
            Path("/tungstenfs/scratch/gmicro/reitsabi/CV7000/Flatfield_correction_tests/20221221-Field-illumination-QC_20221221_143935/Dyes_60xW_Cellvis/")]


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
    name="EICM All",
    cache_result_in_memory=False,
    persist_result=True,
    result_storage="local-file-system/eicm",
    task_runner=DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "queue": "cpu_long",
            "cores": 2,
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
            "maximum": 3,
        },
    )
)
def eicm_all(
        raw_data: RawData = RawData(),
        median_filter: MedianFilter = MedianFilter(),
        gaussian_fit: GaussianFit = GaussianFit(),
        polynomial_fit: PolynomialFit = PolynomialFit()
):
    if median_filter.apply:
        eicm_median_filter(shading_references=raw_data.shading_references,
                           filter_size=median_filter.filter_size)

    if gaussian_fit.apply:
        eicm_gaussian_fit(shading_references=raw_data.shading_references)

    if polynomial_fit.apply:
        eicm_polynomial_fit(shading_references=raw_data.shading_references,
                            polynomial_degree=polynomial_fit.polynomial_degree,
                            order=polynomial_fit.order)
