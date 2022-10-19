import argparse
from datetime import datetime
from os.path import splitext, join, exists, dirname, basename

import numpy as np
import pkg_resources
import prefect
from eicm.estimator.gaussian2D_fit import get_coords, fit_gaussian_2d, \
    compute_fitted_matrix
from eicm.estimator.utils import normalize_matrix
from futils.io import create_output_dir
from prefect import task, Flow
from prefect.core import Parameter
from prefect.run_configs import LocalRun
from prefect.storage import GitHub
from prefect.tasks.secrets import PrefectSecret
from tifffile import imread, imwrite


@task(nout=3)
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


@task(nout=2)
def fit_gaussian_2d_task(img, coords):
    return fit_gaussian_2d(data=img,
                           coords=coords)


@task()
def compute_fitted_matrix_task(coords, ellipsoid_parameters, mip):
    return compute_fitted_matrix(coords=coords,
                                 ellipsoid_parameters=ellipsoid_parameters,
                                 shape=mip.shape)


@task()
def info_txt(save_dir, name, mip_path, amplitude,
             offset,
             mu_x,
             mu_y,
             suffix):
    n, ext = splitext(name)
    tmp = join(save_dir, n + "_" + suffix)
    save_path = tmp
    c = 1
    while exists(save_path + ".md"):
        save_path = tmp + "_" + str(c)
        c += 1

    save_path = save_path + ".md"

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


with Flow(name="EICM with Gaussian Fit",
          run_config=LocalRun(labels=["CPU"])) as flow:
    mip_path = Parameter("mip_path", default="/path/to/mip")
    group = Parameter("group", default="gmicro")
    user = Parameter("user", default="buchtimo")
    output_root_dir = PrefectSecret("output-root-dir")

    logger = prefect.context.get("logger")

    mip, parent_dir, name = load_img_task(path=mip_path)

    coords = get_coords_task(mip)

    popt, pcov = fit_gaussian_2d_task(img=mip,
                                      coords=coords)
    logger.info(f"Found fit with popt = {popt} and corresponding pcov "
                f"= {pcov}.")

    matrix = compute_fitted_matrix_task(coords=coords,
                                        ellipsoid_parameters=popt,
                                        mip=mip)

    matrix = normalize_task(matrix=matrix)

    output_dir = create_output_dir(root_dir=output_root_dir,
                                   group=group,
                                   user=user,
                                   flow_name="eicm-gaussian-fit")

    save_matrix_task(matrix=matrix,
                     save_dir=output_dir,
                     name=name,
                     suffix="eicm-fit-gaussian")

    info_txt(save_dir=output_dir,
             name=name,
             mip_path=mip_path,
             amplitude=popt[0],
             offset=popt[1],
             mu_x=popt[2],
             mu_y=popt[3],
             suffix="eicm-fit-gaussian")

flow.storage = GitHub(
    repo="fmi-faim/prefect-workflows",
    path="eicm_flows/fit_gaussian_estimation.py",
    ref="eicm-v0.1.1",
    access_token_secret="github-access-token_buchtimo"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mip_path", type=str)
    parser.add_argument("--group", type=str)
    parser.add_argument("--user", type=str)

    args = parser.parse_args()
    flow.run(mip_path=args.mip_path)
