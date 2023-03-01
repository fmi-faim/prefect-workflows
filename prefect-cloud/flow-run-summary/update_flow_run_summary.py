import argparse
import configparser
import json
import subprocess
from asyncio import sleep

import yaml
from prefect import task, get_client, flow, get_run_logger
from prefect.exceptions import ObjectNotFound
from prefect.orion.schemas.filters import FlowRunFilter, FlowRunFilterId
from prefect.orion.schemas.states import StateType
from prefect.task_runners import SequentialTaskRunner
from pyairtable import Api, Base


@task(retries=3)
def load_airtable_config(path):
    airtable_config = configparser.ConfigParser()
    airtable_config.read(path)
    return airtable_config


@task(retries=3)
def connect_to_base(airtable_config):
    api = Api(airtable_config['DEFAULT']['api_key'])

    return api.get_base(airtable_config['DEFAULT']['base_id'])


@task(retries=3)
def get_flow_run_log_records(flow_run_log):
    records = flow_run_log.all()
    return records


def get_job_info(job_ids=[60236]):
    ids = ",".join([str(j) for j in job_ids])
    cmd = subprocess.run(["sacct", "-j", ids, "--yaml"], capture_output=True)
    return yaml.safe_load(cmd.stdout)


@task(retries=3)
async def get_flow_run(record, client):
    flow_run = None
    try:
        flow_run_id = record["fields"]["flow-run-id"]
        flow_run = await client.read_flow_run(flow_run_id=flow_run_id)
    except ObjectNotFound as e:
        get_run_logger().warning(f"Record {flow_run_id} not found. "
                                 f"Might be from a different workspace.")

    return flow_run

@task(retries=3)
async def get_task_run_stats(record, client):
    flow_run_id = record["fields"]["flow-run-id"]

    total_tr = []
    offset = 0

    tr = await client.read_task_runs(
        flow_run_filter=FlowRunFilter(
            id=FlowRunFilterId(any_=[flow_run_id])),
        limit=200,
        offset=offset,
    )
    total_tr.extend(tr)

    while len(tr) > 0:
        offset += 200
        tr = await client.read_task_runs(
            flow_run_filter=FlowRunFilter(
                id=FlowRunFilterId(any_=[flow_run_id])),
            limit=200,
            offset=offset,
        )
        total_tr.extend(tr)

    completed_task_runs = list(
        filter(lambda t: t.state.type == StateType.COMPLETED, total_tr))
    failed_task_runs = list(
        filter(lambda t: t.state.type == StateType.FAILED, total_tr))
    crashed_task_runs = list(
        filter(lambda t: t.state.type == StateType.CRASHED, total_tr))
    cancelled_task_runs = list(
        filter(lambda t: t.state.type == StateType.CANCELLED, total_tr))

    return len(tr), len(completed_task_runs), len(cancelled_task_runs), len(failed_task_runs), len(crashed_task_runs)


def get_resource_summary(info):
    job_infos = info["jobs"]

    n_cpus = 0
    memory = 0
    gres = {}
    compute_time = 0
    for job_info in job_infos:
        gres_allocated = list(filter(lambda d: d["type"] == "gres",
                                     job_info["tres"]["allocated"]))
        if len(gres_allocated) > 0:
            true_gpus = list(
                filter(lambda d: d["name"] != "gpu", gres_allocated))
            for d in true_gpus:
                name = d["name"]
                count = d["count"]
                if name in gres.keys():
                    gres[name] += count
                else:
                    gres[name] = count

        memory += job_info["required"]["memory"]
        n_cpus += job_info["required"]["CPUs"]
        compute_time += job_info["time"]["elapsed"]

    return compute_time, n_cpus, memory, gres

def build_log_entry(record, flow_run, task_run_stats):
    row = None

    if flow_run.state.type in [StateType.CANCELLED, StateType.COMPLETED,
                               StateType.CRASHED, StateType.FAILED]:
        job_info = get_job_info([int(i) for i in record["fields"][
            "slurm-jobs"].split(",")])
        n_seconds, n_cpus, memory, gres = get_resource_summary(job_info)

        n_tr, completed_tr, cancelled_tr, failed_tr, crashed_tr = task_run_stats

        row = {}
        row["flow-run-id"] = record["fields"]["flow-run-id"]
        row["slurm-jobs"] = record["fields"]["slurm-jobs"]
        row["slurm-job-start"] = record["fields"]["date"]
        row["flow-created"] = str(flow_run.created)
        row["name"] = flow_run.name
        row["flow-id"] = str(flow_run.flow_id)
        row["deployment-id"] = str(flow_run.deployment_id)
        row["work-queue-name"] = flow_run.work_queue_name
        row["flow-version"] = flow_run.flow_version
        row["parameters"] = json.dumps(flow_run.parameters)
        row["tags"] = json.dumps(flow_run.tags)
        if flow_run.start_time is None:
            row["flow-start"] = record["fields"]["date"]
        else:
            row["flow-start"] = str(flow_run.start_time)
        if flow_run.end_time is None:
            row["flow-end"] = record["fields"]["date"]
        else:
            row["flow-end"] = str(flow_run.end_time)

        row["flow-run-time"] = flow_run.total_run_time.total_seconds()
        row["infrastructure-document-id"] = str(
            flow_run.infrastructure_document_id)
        row["created-by-user"] = flow_run.created_by.display_value
        row["final-flow-run-state"] = flow_run.state_name.upper()
        row["final-flow-run-message"] = flow_run.state.message
        row["task-runs"] = n_tr
        row["completed-task-runs"] = completed_tr
        row["failed-task-runs"] = failed_tr
        row["cancelled-task-runs"] = cancelled_tr
        row["crashed-task-runs"] = crashed_tr
        row["flow-compute-time"] = n_seconds
        row["cpus"] = n_cpus
        row["memory"] = memory
        row["gres"] = json.dumps(gres)
        row["gpus"] = sum(gres.values())
    return row


@task(retries=3)
def update_airtable(row, record, flow_run_summary, flow_run_log):
    flow_run_summary.create(row)
    flow_run_log.update(record["id"], fields={"processed": True})


@flow(
    name="Update flow-run-summary",
    task_runner=SequentialTaskRunner()
)
def add_flow_run_summary(airtable_config_path: str,
                         output_table_name: str):
    airtable_config = load_airtable_config(airtable_config_path)

    base = connect_to_base(airtable_config)

    flow_run_summary = base.get_table(output_table_name)
    flow_run_log = base.get_table("flow-run-log")
    flow_run_log_records = get_flow_run_log_records(flow_run_log)

    client = get_client()

    get_run_logger().info(f"Found {len(flow_run_log_records)} records.")

    for record in flow_run_log_records:
        if "processed" not in record["fields"].keys():
            flow_run = get_flow_run(record, client)
            task_run_stats = get_task_run_stats(record, client)
            if flow_run is not None:
                row = build_log_entry(record, flow_run, task_run_stats)

                if row is not None:
                    update_airtable(row, record, flow_run_summary, flow_run_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--airtable_config")
    parser.add_argument("--output_table_name")
    args = parser.parse_args()
    add_flow_run_summary(airtable_config_path=args.airtable_config,
                         output_table_name=args.output_table_name)
