#!/bin/bash

IP="${1:-127.0.0.1}"

poetry run python3 -B sambot.py --ip "$IP"