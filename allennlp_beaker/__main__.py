from collections import deque
import os
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Iterable, Optional
import uuid

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
import click
import click_spinner
import yaml


DEFAULT_CLUSTER = "ai2/on-prem-ai2-server1"
DOCKERFILE = """
FROM python:3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /stage/allennlp

ENTRYPOINT ["allennlp"]

ARG ALLENNLP

RUN pip install --no-cache-dir ${ALLENNLP}
"""

DOCKERFILE_EXTRA_STEPS = """
# Ensure allennlp isn't re-installed when we install allennlp-models.
ENV ALLENNLP_VERSION_OVERRIDE allennlp

# To be compatible with older versions of allennlp-models.
ENV IGNORE_ALLENNLP_IN_SETUP true

ARG PACKAGES

RUN pip install --no-cache-dir ${PACKAGES}
"""


def echo_command_output(cmd: List[str]) -> None:
    for line in shell_out_command(cmd):
        click.echo(line, nl=True)


def shell_out_command(cmd: List[str]) -> Iterable[str]:
    try:
        for line in subprocess.Popen(  # type: ignore
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).stdout:
            yield line.rstrip()
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(click.style(exc.output, fg="red"))
    except FileNotFoundError as exc:
        raise click.ClickException(click.style(f"{exc.filename} not found", fg="red"))


def create_beaker_config(
    name: str = None,
    description: str = None,
    image: str = None,
    gpus: int = 0,
    cluster: str = DEFAULT_CLUSTER,
) -> Dict[str, Any]:
    return {
        "description": description,
        "tasks": [
            {
                "name": name,
                "spec": {
                    "image": image,
                    "resultPath": "/output",
                    "args": [
                        "train",
                        "config.jsonnet",
                        "-s",
                        "/output",
                        "--file-friendly-logging",
                    ],
                    "requirements": {"gpuCount": gpus},
                },
                "cluster": cluster,
            }
        ],
    }


def parse_version(ctx, param, version) -> str:
    if not version:
        return version
    if param.name == "allennlp_version":
        package = "allennlp"
    else:
        package = "allennlp-models"
    if version.startswith("git@"):
        git_url = f"https://github.com/allenai/{package}"
        if version == "git@master":
            # Get the latest commit from the git repo.
            latest_commits = list(
                shell_out_command(["git", "ls-remote", git_url + ".git"])
            )
            latest = latest_commits[0].split("\t")[0]
            version = f"git+{git_url}.git@{latest}"
        else:
            version = f"git+{git_url}.{version}"
    else:
        version = f"{package}=={version}"
    click.echo("Using " + click.style(f"{version}", fg="green"))
    return version


def check_for_beaker(ctx, param, value):
    # Print beaker version for debugging. If beaker is not installed, this will
    # exit with an error an notify the user.
    echo_command_output([value, "--version"])
    return value


def parse_gpus(ctx, param, value):
    if value is None:
        params_file = ctx.params["config"]
        gpus: int = 0
        params = Params.from_file(params_file).as_dict()
        if "distributed" in params:
            cuda_devices = params["distributed"].get("cuda_devices")
            if cuda_devices:
                gpus = len([d for d in cuda_devices if d >= 0])
        else:
            cuda_device = params.get("trainer", {}).get("cuda_device")
            if isinstance(cuda_device, int) and cuda_device >= 0:
                gpus = 1
        value = gpus
        click.echo("Config specifies " + click.style(f"{value}", fg="green") + " gpus")
    return value


@click.command()
@click.option(
    "--beaker",
    default="beaker",
    help="The beaker executable to use.",
    callback=check_for_beaker,
)
@click.option(
    "--name",
    prompt="What do you want to call your experiment?",
    help="The name of the experiment on beaker.",
)
@click.option(
    "--config",
    prompt="Where is your configuration file?",
    help="The path to your AllenNLP configuration file.",
    callback=lambda ctx, param, value: cached_path(value),
)
@click.option(
    "--allennlp-version",
    prompt="What version of AllenNLP do you want to use?",
    default="git@master",
    help="The PyPI version, branch, or commit SHA of AlleNLP to use. "
    "Git branches and commits should be prefixed with 'git@'. For example, "
    "'git@master'.",
    callback=parse_version,
)
@click.option(
    "--models-version",
    prompt="What version (if any) of AllenNLP Models do you want to use?",
    default="",
    help="The PyPI version, branch, or commit SHA of AllenNLP Models to use."
    "Git branches and commits should be prefixed with 'git@'. For example, "
    "'git@master'.",
    callback=parse_version,
)
@click.option(
    "--packages",
    prompt="What other Python packages does your experiment need?",
    help="Additional Python packages to install in the docker image. "
    "The value of this argument will be passed directly to `pip install`.",
    default="",
)
@click.option(
    "--gpus", default=None, callback=parse_gpus,
)
@click.option(
    "--workspace",
    default=None,
    help="The beaker workspace to submit the experiment to.",
)
def run(
    beaker: str,
    name: str,
    config: str,
    allennlp_version: str,
    models_version: str,
    packages: str,
    gpus: int,
    workspace: Optional[str],
):
    # We create a temp directory to use as context for the Docker build, and
    # also to create a  temporary beaker config file.
    with TemporaryDirectory() as context_dir:
        # Write the training config to the context directory.
        training_config_path = os.path.join(context_dir, "config.jsonnet")
        shutil.copyfile(config, training_config_path)

        image_id = str(uuid.uuid4())
        local_image_name = f"allennlp-beaker-{name}:{image_id}"
        beaker_image_name = f"allennlp-beaker-{name}-{image_id}"
        if models_version:
            packages = models_version + " " + packages
        packages = packages.strip()

        # Write the Dockefile to the context directory.
        dockerfile_path = os.path.join(context_dir, "Dockerfile")
        with open(dockerfile_path, "w") as dockerfile:
            dockerfile.write(DOCKERFILE)
            if packages:
                dockerfile.write(DOCKERFILE_EXTRA_STEPS)

        # Write the beaker config to the context directory.
        beaker_config_path = os.path.join(context_dir, "config.yml")
        with open(beaker_config_path, "w") as beaker_config:
            beaker_config.write(
                yaml.dump(
                    create_beaker_config(
                        name=name,
                        image=beaker_image_name,
                        gpus=gpus,
                        description=f"{allennlp_version} {packages}",
                    )
                )
            )

        # Build the Docker image.
        click.echo(
            "Building docker image with name "
            + click.style(local_image_name, fg="green")
            + "..."
        )
        build_args = [
            "docker",
            "build",
            "--build-arg",
            f"ALLENNLP={allennlp_version}",
        ]
        if packages:
            build_args.extend(["--build-arg", f"PACKAGES={packages}"])
        build_args.extend(["-t", local_image_name, context_dir])
        with click_spinner.spinner():
            deque(shell_out_command(build_args), maxlen=0)

        # Publish the image to beaker.
        click.echo("Publishing image to beaker...")
        echo_command_output(
            ["beaker", "image", "create", "-n", beaker_image_name, local_image_name]
        )

        # Submit the experiment to beaker.
        click.echo("Submitting experiment to beaker...")
        cmds = [
            "beaker",
            "experiment",
            "create",
            "--name",
            name,
            "-f",
            beaker_config_path,
        ]
        if workspace:
            cmds.extend(["--workspace", workspace])
        echo_command_output(cmds)


if __name__ == "__main__":
    run()