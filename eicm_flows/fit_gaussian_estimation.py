import argparse
from datetime import datetime
from os.path import dirname, basename, splitext, join, exists

import pkg_resources
import prefect
from eicm.estimator.gaussian2D_fit import get_coords, fit_gaussian_2d, \
    compute_fitted_matrix
from eicm.estimator.utils import normalize_matrix
from prefect import task, Flow, Parameter
from prefect.run_configs import LocalRun
from prefect.storage import GitHub
from tifffile import imread
from tifffile import imwrite


@task(nout=3)
def load_img_task(path: str):
    return imread(path), dirname(path), basename(path)


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
def normalize_task(matrix):
    return normalize_matrix(matrix=matrix)


@task()
def save_matrix_task(matrix, save_dir, name, suffix="eicm-fit"):
    n, ext = splitext(name)

    save_path = join(save_dir, n + "_" + suffix + ext)
    imwrite(save_path, matrix, compression="zlib")


@task()
def info_txt(save_dir, name, mip_path):
    n, ext = splitext(name)
    tmp = join(save_dir, n + "_" + "eicm-fit")
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
           "## Packages:\n" \
           "* [https://github.com/fmi-faim/eicm](" \
           f"https://github.com/fmi-faim/eicm): v{eicm_version}\n" \
           "\n"

    with open(save_path, "w") as f:
        f.write(info)


with Flow(name="EICM with Gaussian Fit",
          run_config=LocalRun(labels=["CPU"])) as flow:
    mip_path = Parameter("mip_path", default="/path/to/mip")

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

    save_matrix_task(matrix=matrix,
                     save_dir=parent_dir,
                     name=name)

    info_txt(save_dir=parent_dir,
             name=name,
             mip_path=mip_path)

flow.storage = GitHub(
    repo="fmi-faim/prefect-workflows",
    path="eicm_flows/fit_gaussian_estimation.py",
    ref="eicm-v0.1.0",
    access_token_secret="github-access-token_buchtimo"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mip_path", type=str)

    args = parser.parse_args()
    flow.run(mip_path=args.mip_path)
