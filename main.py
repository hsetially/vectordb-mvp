import argparse
import logging
import os
import signal
import sys
from pathlib import Path

import yaml

from api.grpc_service import serve
from api.query_router import SegmentManager, QueryRouter
from core.config import IndexConfig, SegmentConfig
from storage.wal import WALWriter
from storage.binlog import BinlogWriter, BinlogReader
from storage.snapshot import SnapshotManager


def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str = None) -> dict:
    """
    Loads configuration from a YAML file or returns hardcoded defaults.
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    return {
        "storage": {
            "data_dir": "./data",
            "wal_dir": "./data/wal",
            "binlog_dir": "./data/binlog"
        },
        "index": {
            "M": 16,
            "ef_construction": 200,
            "metric": "l2",
            "dim": 128
        },
        "segment": {
            "max_vectors_per_segment": 100000
        },
        "server": {
            "grpc_port": 9000
        },
        "logging": {
            "level": "INFO"
        }
    }


class VectorDBServer:
    """
    Initializes and manages all components of the VectorBase system,
    including the storage layer, indexing, and gRPC service.
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        for key in ["wal_dir", "binlog_dir"]:
            Path(config["storage"][key]).mkdir(parents=True, exist_ok=True)
        
        self.wal_writer = WALWriter(wal_dir=config["storage"]["wal_dir"])
        self.binlog_writer = BinlogWriter(binlog_dir=config["storage"]["binlog_dir"])
        self.binlog_reader = BinlogReader(binlog_dir=config["storage"]["binlog_dir"])
        
        index_config = IndexConfig(
            M=config["index"]["M"],
            ef_construction=config["index"]["ef_construction"],
            metric=config["index"]["metric"]
        )
        
        segment_config = SegmentConfig(
            max_vectors_per_segment=config["segment"]["max_vectors_per_segment"],
            dim=config["index"]["dim"]
        )
        
        self.snapshot_manager = SnapshotManager(
            storage_dir=config["storage"]["data_dir"],
            wal_writer=self.wal_writer,
            binlog_writer=self.binlog_writer,
            binlog_reader=self.binlog_reader,
            config=index_config,
            dim=config["index"]["dim"]
        )
        
        self.logger.info("Attempting to recover from checkpoint...")
        try:
            recovered_segments, last_lsn = self.snapshot_manager.recover()
            self.logger.info(f"Recovered {len(recovered_segments)} segments at LSN {last_lsn}")
        except Exception as e:
            self.logger.warning(f"Recovery failed: {e}. Starting fresh.")
            recovered_segments = []
        
        self.segment_manager = SegmentManager(
            dim=config["index"]["dim"],
            wal_writer=self.wal_writer,
            binlog_writer=self.binlog_writer,
            snapshot_manager=self.snapshot_manager,
            index_config=index_config,
            segment_config=segment_config
        )
        
        if recovered_segments:
            self.logger.info(f"Restoring {len(recovered_segments)} recovered segments...")
            for seg in recovered_segments:
                self.segment_manager.sealed_segments.append(seg)
        
        self.query_router = QueryRouter(segment_manager=self.segment_manager)
        
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    def _shutdown_handler(self, signum, frame):
        """Handles shutdown signals gracefully."""
        self.logger.info(f"Received shutdown signal {signum}")
        self._shutdown()
        sys.exit(0)

    def _shutdown(self):
        """Performs graceful shutdown tasks."""
        try:
            self.logger.info("Creating final checkpoint before shutdown...")
            all_segments = self.segment_manager.get_all_segments()
            self.snapshot_manager.checkpoint(all_segments)
            self.logger.info("Checkpoint complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown checkpoint: {e}")
        finally:
            self.wal_writer.close()
            self.logger.info("Shutdown complete")

    def start(self):
        """Starts the gRPC server."""
        self.logger.info(
            f"Starting VectorDB gRPC server on port {self.config['server']['grpc_port']}"
        )
        serve(
            port=self.config["server"]["grpc_port"],
            segment_manager=self.segment_manager,
            wal_writer=self.wal_writer,
            query_router=self.query_router
        )


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="VectorBase Vector Database")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory from config"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Override gRPC port from config"
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override log level from config"
    )
    
    args = parser.parse_args()
    
    config = load_config(config_path=args.config)
    
    if args.data_dir:
        config["storage"]["data_dir"] = args.data_dir
        config["storage"]["wal_dir"] = os.path.join(args.data_dir, "wal")
        config["storage"]["binlog_dir"] = os.path.join(args.data_dir, "binlog")
    
    if args.port:
        config["server"]["grpc_port"] = args.port
    
    if args.log_level:
        config["logging"]["level"] = args.log_level
    
    setup_logging(level=config["logging"]["level"])
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("VectorBase Vector Database")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config}")
    
    try:
        server = VectorDBServer(config=config)
        server.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
