#!/usr/bin/env python
"""Test with debug logging"""

import asyncio
import logging
from src.main import fetch_data

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s'
)

# Run the fetch_data function
asyncio.run(fetch_data(3))