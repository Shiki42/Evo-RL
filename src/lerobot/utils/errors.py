# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class DeviceNotConnectedError(ConnectionError):
    """Exception raised when the device is not connected."""

    def __init__(self, message="This device is not connected. Try calling `connect()` first."):
        self.message = message
        super().__init__(self.message)


class DeviceAlreadyConnectedError(ConnectionError):
    """Exception raised when the device is already connected."""

    def __init__(
        self,
        message="This device is already connected. Try not calling `connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


def _serial_from_port(port: str) -> str:
    port_name = port.rsplit("/", 1)[-1]
    if "-if" not in port_name or "_" not in port_name:
        return port

    return port_name.split("-if", 1)[0].rsplit("_", 1)[-1]


class DeviceDroppedConnectionError(ConnectionError):
    """Exception raised when an already-connected device stops responding."""

    def __init__(
        self,
        device_label: str,
        port: str,
        operation: str,
        cause: ConnectionError,
    ):
        serial = _serial_from_port(port)
        self.message = (
            f"{device_label} dropped offline while {operation} "
            f"(serial={serial}, port={port}): {cause}"
        )
        super().__init__(self.message)
