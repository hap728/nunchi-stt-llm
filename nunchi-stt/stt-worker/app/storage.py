from __future__ import annotations

from pathlib import Path

import boto3

from .config import Settings


def build_s3_client(settings: Settings):
    return boto3.client(
        "s3",
        endpoint_url=settings.ncp_endpoint,
        region_name=settings.ncp_region,
        aws_access_key_id=settings.ncp_access_key,
        aws_secret_access_key=settings.ncp_secret_key,
    )


def download_audio_file(settings: Settings, object_key: str, output_path: Path) -> Path:
    client = build_s3_client(settings)
    client.download_file(settings.ncp_bucket, object_key, str(output_path))
    return output_path
