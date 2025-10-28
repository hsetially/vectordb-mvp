import time
from concurrent import futures

import grpc
import numpy as np

import common_pb2
import vectordb_pb2
import vectordb_pb2_grpc

from api.query_router import QueryRouter, SegmentManager
from storage.wal import WALWriter

class VectorDBServicer(vectordb_pb2_grpc.VectorDBServicer):
    def __init__(self, segment_manager: SegmentManager, wal_writer: WALWriter, query_router: QueryRouter):
        self.segment_manager = segment_manager
        self.wal_writer = wal_writer
        self.query_router = query_router

    def Insert(self, request: vectordb_pb2.InsertRequest, context) -> vectordb_pb2.InsertResponse:
        inserted_ids = []
        last_lsn = -1
        
        for record in request.vectors:
            if len(record.vector) != self.segment_manager.dim:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Invalid vector dimension. Expected {self.segment_manager.dim}, got {len(record.vector)}.")
                return vectordb_pb2.InsertResponse()

            vector = np.array(record.vector, dtype=np.float32)
            last_lsn = self.wal_writer.insert(record.id, vector)
            self.segment_manager.insert(record.id, vector)
            inserted_ids.append(record.id)
            
        return vectordb_pb2.InsertResponse(
            lsn=last_lsn, inserted_count=len(inserted_ids), inserted_ids=inserted_ids
        )

    def Search(self, request: vectordb_pb2.SearchRequest, context) -> vectordb_pb2.SearchResponse:
        start_time = time.perf_counter()
        if len(request.query_vector) != self.segment_manager.dim:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid query vector dimension.")
            return vectordb_pb2.SearchResponse()
        
        query_vector = np.array(request.query_vector, dtype=np.float32)
        results = self.query_router.search(query_vector, k=request.k, ef=request.ef)
        
        query_time_ms = (time.perf_counter() - start_time) * 1000
        
        search_results = [common_pb2.SearchResult(id=res_id, distance=res_dist) for res_id, res_dist in results]
        
        return vectordb_pb2.SearchResponse(results=search_results, query_time_ms=query_time_ms)

    def Delete(self, request: vectordb_pb2.DeleteRequest, context) -> vectordb_pb2.DeleteResponse:
        last_lsn = -1
        for vec_id in request.ids:
            last_lsn = self.wal_writer.delete(vec_id)
            self.segment_manager.delete(vec_id)
            
        return vectordb_pb2.DeleteResponse(lsn=last_lsn, deleted_count=len(request.ids))

    def GetStats(self, request: vectordb_pb2.StatsRequest, context) -> vectordb_pb2.StatsResponse:
        stats = self.segment_manager.get_stats()
        return vectordb_pb2.StatsResponse(
            total_vectors=stats.get("total_vectors", 0),
            num_segments=stats.get("num_segments", 0),
            num_sealed_segments=stats.get("num_sealed", 0),
            current_lsn=self.wal_writer.current_lsn
        )

def serve(port: int, segment_manager: SegmentManager, wal_writer: WALWriter, query_router: QueryRouter):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vectordb_pb2_grpc.add_VectorDBServicer_to_server(
        VectorDBServicer(segment_manager, wal_writer, query_router), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"VectorDB gRPC server started on port {port}. Press Ctrl+C to stop.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.stop(0)
