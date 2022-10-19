import argparse
from datetime import datetime
from os.path import splitext, join, exists, dirname, basename

import numpy as np
import pkg_resources
import prefect
from eicm.estimator.polynomial_fit import polynomial_fit
from eicm.estimator.utils import normalize_matrix
from futils.io import create_output_dir
from prefect import task, Flow
from prefect.core import Parameter
from prefect.run_configs import LocalRun
from prefect.storage import GitHub
from prefect.tasks.secrets import PrefectSecret
from tifffile import imwrite, imread


@task(nout=3)
def load_img_task(path: str):
    return np.squeeze(imread(path)), dirname(path), basename(path)


@task()
def normalize_task(matrix):
    return normalize_matrix(matrix=matrix)


@task()
def save_matrix_task(matrix, save_dir, name, suffix):
    n, ext = splitext(name)

    save_path = join(save_dir, n + "_" + suffix + ext)
    imwrite(save_path, matrix.astype(np.float32), compression="zlib",
            resolutionunit="None")


@task(nout=2)
def fit_polynomial(mip, polynomial_degree, order):
    return polynomial_fit(mip=mip, polynomial_degree=polynomial_degree,
                          order=order)


@task()
def info_txt(save_dir, name, mip_path, polynomial_degree, order,
             poly_str, suffix):
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

    info = "# Estimate Illumination Correction Matrix (EICM) with Polynomial Fit\n" \
           f"Date: {now}\n\n" \
           "`EICM with Polynomial Fit` is a service provided by the Facility " \
           "for Advanced Imaging and Microscopy (FAIM) at FMI for " \
           "biomedical research. Consult with FAIM on appropriate usage. " \
           "\n\n" \
           "## Parameters:\n" \
           f"* `mip_path`: {mip_path}\n" \
           f"* `polynomial_degree`: {polynomial_degree}\n" \
           f"* `order`: {order}\n" \
           "\n" \
           "## Polynomial:\n" \
           f"f(X, Y) = {poly_str}\n" \
           "\n" \
           "## Packages:\n" \
           "* [https://github.com/fmi-faim/eicm](" \
           f"https://github.com/fmi-faim/eicm): v{eicm_version}\n" \
           "\n"

    with open(save_path, "w") as f:
        f.write(info)


with Flow(name="EICM with Polynomial Fit",
          run_config=LocalRun(labels=["CPU"])) as flow:
    mip_path = Parameter("mip_path", default="/path/to/mip")
    polynomial_degree = Parameter("polynomial_degree", default=4)
    order = Parameter("order", default=4)
    group = Parameter("group", default="gmicro")
    user = Parameter("user", default="buchtimo")
    output_root_dir = PrefectSecret("output-root-dir")

    logger = prefect.context.get("logger")

    mip, parent_dir, name = load_img_task(path=mip_path)

    matrix, poly_str = fit_polynomial(mip=mip,
                                      polynomial_degree=polynomial_degree,
                                      order=order)

    matrix = normalize_task(matrix=matrix)

    output_dir = create_output_dir(root_dir=output_root_dir,
                                   group=group,
                                   user=user,
                                   flow_name="eicm-polynomial-fit")

    save_matrix_task(matrix=matrix,
                     save_dir=output_dir,
                     name=name,
                     suffix="eicm-fit-polynomial")

    info_txt(save_dir=output_dir,
             name=name,
             mip_path=mip_path,
             polynomial_degree=polynomial_degree,
             order=order,
             poly_str=poly_str,
             suffix="eicm-fit-polynomial")

flow.storage = GitHub(
    repo="fmi-faim/prefect-workflows",
    path="eicm_flows/fit_polynomial_estimation.py",
    ref="eicm-v0.1.1",
    access_token_secret="github-access-token_buchtimo"
)

if __name__ == "__main__":
    def none_or_int(value):
        if value == 'None':
            return None
        return value


    parser = argparse.ArgumentParser()
    parser.add_argument("--mip_path", type=str)
    parser.add_argument("--polynomial_degree", type=int)
    parser.add_argument("--order", type=none_or_int)
    parser.add_argument("--group", type=str)
    parser.add_argument("--user", type=str)

    args = parser.parse_args()
    flow.run(mip_path=args.mip_path)
