import os
import random
import signal
from collections import defaultdict
from concurrent import futures

import faiss
import grpc
import leveldb
import numpy as np

from vsearch_pb2 import (DbData, DbStatsResponse, DeleteAllResponse,
                         DeleteResponse, FaissConfig, GetVectorsResponse,
                         SearchResponse, TrainResponse, UpsertResponse)
from vsearch_pb2_grpc import (VsearchServiceServicer,
                              add_VsearchServiceServicer_to_server)

DIMENSION = 128
TRAINING_THRESHOLD = 300
# INDEX_TYPE = os.environ.get("INDEX_TYPE", "IndexFlatL2")
INDEX_TYPE = "IndexIVFFlat"
# INDEX_TYPE = "IndexIVFPQ"
QUANTIZER_TYPE = os.environ.get("QUANTIZER_TYPE", "Flat")
NLIST = int(os.environ.get("NLIST", "100"))
M = int(os.environ.get("M", "8"))  # The number of subquantizers for IndexIVFPQ

FLAT_TYPES = ["IndexFlatL2"]
IS_FLAT_TYPE = INDEX_TYPE in FLAT_TYPES


def key_exists(db, target_key):
    try:
        value = db.Get(target_key.encode())
        return True
    except KeyError:
        return False


class VsearchServer(VsearchServiceServicer):
    def __init__(self):
        self._faiss_indexes = {}
        self._db = leveldb.LevelDB("vsearch_db")
        self._training_data = defaultdict(list)

    def _create_faiss_index(self):
        if INDEX_TYPE == "IndexFlatL2":
            return faiss.IndexFlatL2(DIMENSION)
        elif INDEX_TYPE == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(
                DIMENSION) if QUANTIZER_TYPE == "Flat" else faiss.IndexIVFFlat(quantizer, DIMENSION, NLIST)
            return faiss.IndexIVFFlat(quantizer, DIMENSION, NLIST)
        elif INDEX_TYPE == "IndexIVFPQ":
            quantizer = faiss.IndexFlatL2(
                DIMENSION) if QUANTIZER_TYPE == "Flat" else faiss.IndexIVFFlat(quantizer, DIMENSION, NLIST)
            # 8 is the number of bits per code for PQ
            return faiss.IndexIVFPQ(quantizer, DIMENSION, NLIST, M, 8)
        else:
            raise ValueError(f"Unsupported index type: {INDEX_TYPE}")

    def _get_faiss_index(self, namespace):
        if namespace not in self._faiss_indexes:
            self._faiss_indexes[namespace] = self._create_faiss_index()
        return self._faiss_indexes[namespace]

    def Upsert(self, request, context):
        n_stored = 0
        n_error = 0
        for data in request.data:
            vec = np.array(data.values, dtype=np.float32)
            namespaces = ["default"] + \
                data.namespaces if data.namespaces else ["default"]

            for namespace in namespaces:
                index = self._get_faiss_index(namespace)
                faiss_id = index.ntotal

                if not index.is_trained:
                    self._db.Put(f"vector_data,{namespace},{data.id}".encode(),
                                 vec.tobytes())
                    continue

                try:
                    if key_exists(self._db, f"map,{data.id},{namespace}"):
                        # idが重複した場合はスキップ
                        raise

                    self._db.Put(f"map,{data.id},{namespace}".encode(),
                                 str(faiss_id).encode())
                    self._db.Put(f"inv_map,{faiss_id},{namespace}".encode(
                    ), f"{data.id},{namespace}".encode())
                    self._db.Put(f"vector_data,{namespace},{data.id}".encode(),
                                 vec.tobytes())

                    index.add(vec.reshape(1, -1))
                    n_stored += 1
                except Exception as e:
                    print(f"Error: {e}")
                    n_error += 1

        return UpsertResponse(nstored=n_stored, nerror=n_error)

    def Train(self, request, context):
        success = False
        proportion = request.proportion
        force = request.force

        for namespace, index in self._faiss_indexes.items():
            if not index.is_trained or force:
                stucked_data = []
                for key, value in self._db.RangeIter(key_from=f"vector_data,{namespace},".encode(),
                                                     key_to=f"vector_data,{namespace}z".encode()):
                    stucked_data.append(np.frombuffer(value, dtype=np.float32))

                if proportion < 1.0:
                    num_samples = int(len(stucked_data) * proportion)
                    train_data = random.sample(stucked_data, num_samples)
                else:
                    train_data = stucked_data

                training_data = np.vstack(train_data)
                index.train(training_data)
                success = True

        return TrainResponse(success=success)

    def DeleteAll(self, request, context):
        for key, _ in self._db.RangeIter():
            self._db.Delete(key)

        for namespace in self._faiss_indexes:
            self._faiss_indexes[namespace].reset()

        return DeleteAllResponse()

    def Delete(self, request, context):
        for data_id in request.ids:
            for key, _ in self._db.RangeIter(key_from=f"map,{data_id},".encode(), key_to=f"map,{data_id}z".encode()):
                _, _, namespace = key.decode().split(",")
                try:
                    faiss_id = int(self._db.Get(key))
                    index = self._get_faiss_index(namespace)
                    index.remove_ids(np.array([faiss_id]))
                    self._db.Delete(key)
                    self._db.Delete(f"inv_map,{faiss_id},{namespace}".encode())

                    if IS_FLAT_TYPE:
                        # Update ID mapping for all following IDs in the same namespace
                        for update_key, update_val in self._db.RangeIter(
                            key_from=f"inv_map,{faiss_id + 1},{namespace}".encode(),
                            key_to=f"inv_map,z,{namespace}".encode(),
                        ):
                            _, update_faiss_id, update_namespace = update_key.decode().split(",")
                            if update_namespace != namespace:
                                break

                            update_data_id, _ = self._db.Get(
                                update_key).decode().split(",")
                            new_faiss_id = int(update_faiss_id) - 1
                            self._db.Put(f"map,{update_data_id},{namespace}".encode(), str(
                                new_faiss_id).encode())
                            self._db.Delete(update_key)
                            self._db.Put(f"inv_map,{new_faiss_id},{namespace}".encode(
                            ), f"{update_data_id},{namespace}".encode())
                except KeyError:
                    continue

        return DeleteResponse()

    def Search(self, request, context):
        vec = np.array(request.values, dtype=np.float32)
        namespace = request.namespace if request.namespace else "default"
        index = self._get_faiss_index(namespace)
        D, I = index.search(vec.reshape(1, -1), request.k)

        ids = []
        distances = []
        for i in range(I.shape[1]):
            faiss_id = I[0][i]
            try:
                data_id = self._db.Get(
                    f"inv_map,{faiss_id},{namespace}".encode()).decode().split(",")[0]
                ids.append(data_id)
                distances.append(D[0][i])
            except KeyError:
                continue

        return SearchResponse(ids=ids, distances=distances)

    def DbStats(self, request, context):
        dbs = []
        for namespace, index in self._faiss_indexes.items():
            ntotal = index.ntotal
            faiss_config = self._get_faiss_config(index)

            dbs.append(DbData(namespace=namespace,
                       ntotal=ntotal, faissconfig=faiss_config))

        return DbStatsResponse(
            istrained=True,  # Not applicable to all index types
            dimension=128,  # Adjust the dimension according to your needs
            status=0,
            dbs=dbs,
        )

    def _get_faiss_config(self, index):
        index_type = type(index).__name__
        metric = "L2"
        nprobe = index.nprobe if hasattr(index, 'nprobe') else 1
        dimension = index.d

        return FaissConfig(
            description=index_type,
            metric=metric,
            nprobe=nprobe,
            dimension=dimension,
        )

    def save_indexes(self, path="faiss_indexes"):
        os.makedirs(path, exist_ok=True)
        for namespace, index in self._faiss_indexes.items():
            file_path = os.path.join(path, f"{namespace}.index")
            faiss.write_index(index, file_path)

    def load_indexes(self, path="faiss_indexes"):
        if not os.path.exists(path):
            return

        for file_name in os.listdir(path):
            if file_name.endswith(".index"):
                namespace = file_name[:-6]
                file_path = os.path.join(path, file_name)
                index = faiss.read_index(file_path)
                self._faiss_indexes[namespace] = index

    def GetVectors(self, request, context):
        namespace = request.namespace if request.namespace else "default"
        response_data = []
        for data_id in request.ids:
            try:
                value = self._db.Get(
                    f"vector_data,{namespace},{data_id}".encode())
                vec = np.frombuffer(value, dtype=np.float32)
                response_data.append(
                    GetVectorsResponse.Data(id=f"{data_id}", values=vec, namespace=namespace))
            except KeyError as e:
                continue
        return GetVectorsResponse(data=response_data)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vsearch_server = VsearchServer()
    vsearch_server.load_indexes()  # Load indexes from files
    add_VsearchServiceServicer_to_server(vsearch_server, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Vector search server started on port 50051")

    def stop_server(signal, frame):
        print("Stopping server...")
        server.stop(0)
        vsearch_server.save_indexes()  # Save indexes to files
        print("Server stopped")

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
