import argparse
import os
from datetime import datetime
from os.path import splitext, join, dirname, basename
from typing import Literal

import numpy as np
import pkg_resources
from cpr.image.ImageTarget import ImageTarget
from eicm.estimator.gaussian2D_fit import get_coords, fit_gaussian_2d, \
    compute_fitted_matrix
from eicm.estimator.utils import normalize_matrix
from prefect import task, flow
from prefect.context import get_run_context
from tifffile import imread, imwrite


@task()
def load_img_task(path: str):
    return imread(path), dirname(path), basename(path)


@task()
def normalize_task(matrix):
    return normalize_matrix(matrix=matrix)


@task()
def save_matrix_task(matrix, save_dir, name, suffix):
    n, ext = splitext(name)

    save_path = join(save_dir, n + "_" + suffix + ext)
    imwrite(save_path, matrix.astype(np.float32), compression="zlib",
            resolutionunit="None")


@task()
def get_coords_task(img):
    return get_coords(img)


@task()
def fit_gaussian_2d_task(img, coords):
    return fit_gaussian_2d(data=img,
                           coords=coords)


@task()
def compute_fitted_matrix_task(coords, ellipsoid_parameters, mip):
    return compute_fitted_matrix(coords=coords,
                                 ellipsoid_parameters=ellipsoid_parameters,
                                 shape=mip.shape)


@task()
def estimate_correction_matrix(mip_path: str, output_dir: str):
    n, ext = splitext(basename(mip_path))
    save_path = join(output_dir, f"{n}_gaussian-fit{ext}")
    matrix = ImageTarget.from_path(save_path)

    mip = imread(mip_path)
    coords = get_coords(mip)
    popt, pcov = fit_gaussian_2d(mip, coords)

    matrix.set_data(normalize_matrix(compute_fitted_matrix(coords=coords,
                                                           ellipsoid_parameters=popt,
                                                           shape=mip.shape)))

    return matrix, popt


@task()
def info_txt(save_dir, matrix: ImageTarget, mip_path, amplitude,
             offset,
             mu_x,
             mu_y):
    save_path = join(save_dir, f"{matrix.get_name()}-{matrix.data_hash}.md")

    eicm_version = pkg_resources.get_distribution("eicm").version
    now = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")

    info = "# Estimate Illumination Correction Matrix (EICM) with Gaussian Fit\n" \
           f"Date: {now}\n\n" \
           "`EICM with Gaussian Fit` is a service provided by the Facility " \
           "for Advanced Imaging and Microscopy (FAIM) at FMI for " \
           "biomedical research. Consult with FAIM on appropriate usage. " \
           "\n\n" \
           "## Parameters:\n" \
           f"* `mip_path`: {mip_path}\n" \
           "\n" \
           "## Fitted Gaussian:\n" \
           f"* Amplitude: {amplitude}\n" \
           f"* Offset: {offset}\n" \
           f"* Centroid (X, Y): ({mu_x}, {mu_y})\n" \
           "\n" \
           "## Packages:\n" \
           "* [https://github.com/fmi-faim/eicm](" \
           f"https://github.com/fmi-faim/eicm): v{eicm_version}\n" \
           "\n"

    with open(save_path, "w") as f:
        f.write(info)


groups = Literal[
    "garber",
    "gbuehler",
    "gcaroni",
    "gchao",
    "gdiss",
    "gfelsenb",
    "gfriedri",
    "ggiorget",
    "ggrossha",
    "ginforma",
    "gkeller",
    "gliberal",
    "gluthi",
    "gmatthia",
    "gmicro",
    "gpeters",
    "grijli",
    "gschub",
    "gthoma",
    "gtsiairi",
    "gturco",
    "gzenke"
]


@flow(
    name="EICM with Gaussian Fit"
)
def eicm_gaussian_fit(
        mip_path: str = "/path/to/mip",
        group: str = groups,
):
    flow_run_name = get_run_context().flow_run.name

    output_dir = join("/tungstenfs/scratch/gmicro_prefect",
                      group,
                      "eicm",
                      flow_run_name)

    os.makedirs(output_dir, exist_ok=True)

    matrix, popt = estimate_correction_matrix.submit(mip_path=mip_path,
                                                     output_dir=output_dir)

    info_txt.submit(save_dir=output_dir,
                    matrix=matrix,
                    mip_path=mip_path,
                    amplitude=popt[0],
                    offset=popt[1],
                    mu_x=popt[2],
                    mu_y=popt[3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mip_path", type=str)
    parser.add_argument("--group", type=str)

    args = parser.parse_args()
    eicm_gaussian_fit(mip_path=args.mip_path, group=args.group)
