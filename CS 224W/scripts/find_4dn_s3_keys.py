#!/usr/bin/env python3
"""Locate 4DN accessions in the public 4DN S3 bucket."""

from __future__ import annotations

import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


BUCKET_URL = "https://4dn-open-data-public.s3.amazonaws.com/"
NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_page(prefix: str, continuation_token: str | None) -> tuple[list[tuple[str, int]], str | None]:
    params = {"list-type": "2", "max-keys": "1000", "prefix": prefix}
    if continuation_token:
        params["continuation-token"] = continuation_token
    url = BUCKET_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        root = ET.fromstring(response.read())
    objects = []
    for item in root.findall("s3:Contents", NAMESPACE):
        key = item.findtext("s3:Key", namespaces=NAMESPACE)
        size_text = item.findtext("s3:Size", namespaces=NAMESPACE)
        if key is not None and size_text is not None:
            objects.append((key, int(size_text)))
    token = root.findtext("s3:NextContinuationToken", namespaces=NAMESPACE)
    return objects, token


def main() -> int:
    targets = set(sys.argv[1:])
    if not targets:
        raise SystemExit("Pass one or more 4DN accessions")
    found: dict[str, tuple[str, int]] = {}
    for prefix in ("fourfront-webprod/files/", "fourfront-webprod/wfoutput/"):
        token = None
        page = 0
        while True:
            page += 1
            objects, token = list_page(prefix, token)
            for key, size in objects:
                for accession in targets - found.keys():
                    if f"/{accession}." in key or key.endswith(f"/{accession}"):
                        found[accession] = (key, size)
                        print(f"FOUND\t{accession}\t{size}\t{key}", flush=True)
            if found.keys() == targets:
                break
            if page % 20 == 0:
                print(f"PROGRESS\t{prefix}\t{page * 1000}", flush=True)
            if not token:
                break
            time.sleep(0.02)
        if found.keys() == targets:
            break
    missing = targets - found.keys()
    for accession in sorted(missing):
        print(f"MISSING\t{accession}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
