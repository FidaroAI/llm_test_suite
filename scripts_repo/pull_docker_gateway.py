#!/usr/bin/env python3
import argparse
import base64
import platform
import subprocess
import sys

import boto3


def run(cmd, input_text=None):
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def ecr_login(registry_id: str, region: str) -> str:
    client = boto3.client("ecr", region_name=region)
    resp = client.get_authorization_token(registryIds=[registry_id])
    auth_data = resp["authorizationData"][0]

    token = auth_data["authorizationToken"]
    proxy_endpoint = auth_data["proxyEndpoint"]

    decoded = base64.b64decode(token).decode("utf-8")
    username, password = decoded.split(":", 1)

    registry = proxy_endpoint.replace("https://", "").replace("http://", "")

    run(
        ["docker", "login", "--username", username, "--password-stdin", registry],
        input_text=password,
    )
    return registry


def get_most_recent_tag(registry_id: str, region: str, repository: str) -> str:
    client = boto3.client("ecr", region_name=region)
    paginator = client.get_paginator("describe_images")

    latest_detail = None

    for page in paginator.paginate(
        registryId=registry_id,
        repositoryName=repository,
        filter={"tagStatus": "TAGGED"},
    ):
        for detail in page.get("imageDetails", []):
            pushed_at = detail.get("imagePushedAt")
            tags = detail.get("imageTags", [])
            if not pushed_at or not tags:
                continue

            if latest_detail is None or pushed_at > latest_detail["imagePushedAt"]:
                latest_detail = detail

    if latest_detail is None:
        raise RuntimeError(f"No tagged images found in repository: {repository}")

    return sorted(latest_detail["imageTags"])[0]


def docker_pull(image: str):
    run(["docker", "pull", image])


def main():
    parser = argparse.ArgumentParser(
        description="Login to AWS ECR with boto3 and pull an image into local Docker."
    )
    parser.add_argument(
        "--registry-id",
        default="434247560915",
        help="AWS account ID / ECR registry ID",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for the ECR registry",
    )
    parser.add_argument(
        "--repository",
        default="fidaro/dev/cvm-llm-gateway",
        help="ECR repository name, e.g. my-service",
    )
    parser.add_argument(
        "--tag",
        help="Image tag to pull. If omitted, pull the most recently pushed tagged image.",
    )

    args = parser.parse_args()

    arch = platform.machine().lower()
    if arch.startswith("arm") or arch == "aarch64":
        raise RuntimeError(
            f"Unsupported architecture: {arch}. The ECR register only has x86 images. If you pull the image then you won't be able to use it locally. You will need to build from the secure-enclave repo."
        )

    registry = ecr_login(args.registry_id, args.region)

    tag = args.tag
    if not tag:
        tag = get_most_recent_tag(args.registry_id, args.region, args.repository)
        print(f"No tag provided, using most recently pushed tag: {tag}")

    image = f"{registry}/{args.repository}:{tag}"

    print(f"Pulling {image} ...")
    docker_pull(image)
    print(f"Done. Image is now available locally as: {image}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
