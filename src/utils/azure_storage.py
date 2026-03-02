"""
Azure Data Lake Storage Gen2 Client
Handles upload/download of data files to/from ADLS
Works with Azure Free Tier subscription
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from azure.identity import DefaultAzureCredential
    from azure.storage.filedatalake import DataLakeServiceClient
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure SDK not installed. Azure features disabled.")


class AzureStorageClient:
    """
    Client for Azure Data Lake Storage Gen2.
    Falls back to local storage if Azure credentials not configured.
    """

    def __init__(self, account_name: str = None, account_key: str = None):
        self.account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT")
        self.account_key = account_key or os.getenv("AZURE_STORAGE_KEY")
        self.is_connected = False
        self._client = None

        if AZURE_AVAILABLE and self.account_name and self.account_key:
            self._connect()
        else:
            logger.warning(
                "⚠️  Azure credentials not found. Running in LOCAL mode. "
                "Set AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_KEY in .env to enable Azure."
            )

    def _connect(self):
        """Establish connection to Azure Data Lake."""
        try:
            account_url = f"https://{self.account_name}.dfs.core.windows.net"
            self._client = DataLakeServiceClient(
                account_url=account_url,
                credential=self.account_key
            )
            self.is_connected = True
            logger.info(f"✅ Connected to Azure ADLS: {self.account_name}")
        except Exception as e:
            logger.error(f"❌ Azure connection failed: {e}")
            self.is_connected = False

    def upload_file(
        self,
        local_path: str,
        container: str,
        remote_path: str
    ) -> bool:
        """Upload a file to Azure Data Lake."""
        if not self.is_connected:
            logger.info(f"[LOCAL MODE] Skipping upload: {local_path} → {remote_path}")
            return False

        try:
            fs_client = self._client.get_file_system_client(container)
            file_client = fs_client.get_file_client(remote_path)

            with open(local_path, "rb") as f:
                data = f.read()
                file_client.upload_data(data, overwrite=True)

            logger.info(f"✅ Uploaded: {local_path} → adls://{container}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            return False

    def download_file(
        self,
        container: str,
        remote_path: str,
        local_path: str
    ) -> bool:
        """Download a file from Azure Data Lake."""
        if not self.is_connected:
            logger.info(f"[LOCAL MODE] Skipping download: {remote_path} → {local_path}")
            return False

        try:
            fs_client = self._client.get_file_system_client(container)
            file_client = fs_client.get_file_client(remote_path)

            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                download = file_client.download_file()
                f.write(download.readall())

            logger.info(f"✅ Downloaded: adls://{container}/{remote_path} → {local_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False

    def list_files(self, container: str, path: str = "") -> list:
        """List files in an Azure Data Lake path."""
        if not self.is_connected:
            logger.info(f"[LOCAL MODE] Cannot list Azure files.")
            return []

        try:
            fs_client = self._client.get_file_system_client(container)
            paths = fs_client.get_paths(path=path)
            files = [p.name for p in paths]
            logger.info(f"Found {len(files)} files in adls://{container}/{path}")
            return files
        except Exception as e:
            logger.error(f"❌ List files failed: {e}")
            return []

    def create_container(self, container: str) -> bool:
        """Create an ADLS container (filesystem)."""
        if not self.is_connected:
            logger.info(f"[LOCAL MODE] Skipping container creation: {container}")
            return False

        try:
            self._client.create_file_system(file_system=container)
            logger.info(f"✅ Container created: {container}")
            return True
        except Exception as e:
            logger.warning(f"Container may already exist: {e}")
            return False
