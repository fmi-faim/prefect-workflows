import argparse
from datetime import datetime
from os.path import splitext, join, exists, dirname, basename

import pkg_resources
from eicm.estimator.gaussian_blur import create_blurred_illumination_matrix
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
def save_matrix_task(matrix, save_dir, name, suffix="eicm-blur"):
    n, ext = splitext(name)

    save_path = join(save_dir, n + "_" + suffix + ext)
    imwrite(save_path, matrix, compression="zlib", resolutionunit="None")


@task()
def create_blurred_illumination_matrix_task(img, sigma: float = 20.):
    return create_blurred_illumination_matrix(img=img, sigma=sigma)


@task()
def info_txt(save_dir, name, mip_path, sigma, suffix):
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

    info = "# Estimate Illumination Correction Matrix (EICM) with Gaussian " \
           "Blur\n" \
           f"Date: {now}\n\n" \
           "`EICM with Gaussian Blur` is a service provided by the Facility " \
           "for Advanced Imaging and Microscopy (FAIM) at FMI for " \
           "biomedical research. Consult with FAIM on appropriate usage. " \
           "\n\n" \
           "## Parameters:\n" \
           f"* `mip_path`: {mip_path}\n" \
           f"* `sigma`: {sigma}\n" \
           "\n" \
           "## Packages:\n" \
           "* [https://github.com/fmi-faim/eicm](" \
           f"https://github.com/fmi-faim/eicm): v{eicm_version}\n" \
           "\n"

    with open(save_path, "w") as f:
        f.write(info)


with Flow(name="EICM with Gaussian Blur",
          run_config=LocalRun(labels=["CPU"])) as flow:
    mip_path = Parameter("mip_path", default="/path/to/mip")
    sigma = Parameter("sigma", default=20.)
    group = Parameter("group", default="gmicro")
    user = Parameter("user", default="buchtimo")
    output_root_dir = PrefectSecret("output-root-dir")

    mip, parent_dir, name = load_img_task(path=mip_path)

    matrix = create_blurred_illumination_matrix_task(img=mip,
                                                     sigma=sigma)

    matrix = normalize_task(matrix=matrix)

    output_dir = create_output_dir(root_dir=output_root_dir,
                                   group=group,
                                   user=user,
                                   flow_name="eicm-gaussian-blur")

    save_matrix_task(matrix=matrix,
                     save_dir=output_dir,
                     name=name,
                     suffix="eicm-gaussian-blur")

    info_txt(save_dir=output_dir,
             name=name,
             mip_path=mip_path,
             sigma=sigma,
             suffix="eicm-gaussian-blur")

flow.storage = GitHub(
    repo="fmi-faim/prefect-workflows",
    path="eicm_flows/gaussian_blur_estimation.py",
    ref="eicm-v0.1.1",
    access_token_secret="github-access-token_buchtimo"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mip_path", type=str)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--group", type=str)
    parser.add_argument("--user", type=str)

    args = parser.parse_args()
    flow.run(mip_path=args.mip_path, sigma=args.sigma, group=args.group,
             user=args.user)
