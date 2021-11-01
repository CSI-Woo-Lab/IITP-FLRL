import argparse
import flwr as fl


NUM_CLIENTS = 4
NUM_ROUNDS = 10

def main(args):
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=NUM_CLIENTS,  # Minimum number of clients to be sampled for the next round
        min_available_clients=NUM_CLIENTS,  # Minimum number of clients that need to be connected to the server before a training round can start
        min_eval_clients=NUM_CLIENTS,
    )

    # Start server
    fl.server.start_server(
        server_address=f"[::]:{args.port}",
        config={"num_rounds": NUM_ROUNDS},
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8080)
    args = parser.parse_args()
    main(args)