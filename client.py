import grpc
import numpy as np
from vsearch_pb2 import (Data, DeleteRequest, DeleteAllRequest, SearchRequest,
                         UpsertRequest, TrainRequest, DbStatsRequest)
from vsearch_pb2_grpc import VsearchServiceStub

DIMENSION = 128


def generate_random_data(num_data, dimension):
    return [Data(id=f"data_{i}", values=np.random.rand(dimension).tolist()) for i in range(num_data)]


def main():
    channel = grpc.insecure_channel("localhost:50051")
    client = VsearchServiceStub(channel)

    # Generate random data
    data = generate_random_data(100, DIMENSION)

    # Upsert data
    upsert_request = UpsertRequest(data=data)
    upsert_response = client.Upsert(upsert_request)
    print(
        f"Upserted {upsert_response.nstored} vectors, {upsert_response.nerror} errors")

    # Perform a search
    query_vec = np.random.rand(DIMENSION).tolist()
    search_request = SearchRequest(values=query_vec, k=5)
    search_response = client.Search(search_request)
    print(
        f"Search results: {search_response.ids}, distances: {search_response.distances}")

    # Train the index (not needed for IndexFlatL2, but included for completeness)
    train_request = TrainRequest(proportion=1.0, force=True)
    train_response = client.Train(train_request)
    print("Training complete")

    # Delete a vector
    delete_request = DeleteRequest(ids=["data_0"])
    delete_response = client.Delete(delete_request)
    print("Vector deleted")

    # Get database stats
    db_stats_request = DbStatsRequest()
    db_stats_response = client.DbStats(db_stats_request)
    print("Database stats:")
    print(f"  - Trained: {db_stats_response.istrained}")
    print(f"  - Dimension: {db_stats_response.dimension}")
    print(f"  - Status: {db_stats_response.status}")
    for db_data in db_stats_response.dbs:
        conf_info = str(db_data.faissconfig).replace('\n', ', ')
        print(
            f"  - Namespace={db_data.namespace}, {db_data.ntotal} vectors, config {conf_info}")

    # Delete all data
    delete_all_request = DeleteAllRequest()
    delete_all_response = client.DeleteAll(delete_all_request)
    print("All data deleted")


if __name__ == "__main__":
    main()
