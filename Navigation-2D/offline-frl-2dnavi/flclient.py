import argparse
from collections import OrderedDict
from datetime import datetime
import torch
import gym, navigation_2d
import flwr as fl
from d3rlpy.algos import CQL, BCQ
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from sklearn.model_selection import train_test_split


def main(args):
    """Create model, load data, define Flower client, start Flower client."""
    # Load data
    dataset = MDPDataset.load(args.dataset_path)
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    env_name = f"Navi-Acc-Lidar-Obs-Task0_easy-v0"
    env = gym.make(env_name)
    evaluate_scorer = evaluate_on_environment(env, 10)

    algos_dict = {
        "cql": CQL,
        "bcq": BCQ,
    }
    algo_cls = algos_dict[args.client_algo]
    agent = algo_cls(
        use_gpu=True,
    )

    agent.build_with_dataset(dataset)
    # td_error = td_error_scorer(agent, test_episodes)
    len_dataset = len(train_episodes)
    date = datetime.now().strftime("%Y%m%d%H%M%S")

    class D3rlpyBaseClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [
                val.cpu().numpy() for _, val in agent.impl.policy.state_dict().items()
            ] + [
                val.cpu().numpy()
                for _, val in agent.impl.q_function.state_dict().items()
            ]

        def set_parameters(self, parameters):
            policy_len = len(agent.impl.policy.state_dict())

            policy_param, q_param = parameters[:policy_len], parameters[policy_len:]

            policy_params_dict = zip(
                agent.impl.policy.state_dict().keys(), policy_param
            )
            policy_state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in policy_params_dict}
            )
            agent.impl.policy.load_state_dict(policy_state_dict, strict=True)

            qfunc_params_dict = zip(agent.impl.q_function.state_dict().keys(), q_param)
            qfunc_state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in qfunc_params_dict}
            )
            agent.impl.q_function.load_state_dict(qfunc_state_dict, strict=True)
            # train(net, trainloader, epochs=1)
            # return self.get_parameters(), num_examples["trainset"], {}
            return self.get_parameters(), len_dataset, {}

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            agent.fit(
                train_episodes,
                n_epochs=args.num_local_epoch,
                verbose=True,
                eval_episodes=test_episodes,
                # scorers={
                #     "td_error": td_error_scorer,
                #     "environment": evaluate_on_environment(env)
                # },
                experiment_name=self.client_setting,
                logdir="forl_logs",
            )
            return self.get_parameters(), len_dataset, {}

        def evaluate(self, parameters, config):
            # print("pass evaluation")
            self.set_parameters(parameters)
            score = evaluate_scorer(agent)
            # loss, accuracy = test(net, testloader)
            # return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}
            return score, 100, {"Score": score}

    class BCQClient(D3rlpyBaseClient):
        client_setting = f"{date}_BCQ_{args.dataset_path}"

        def get_parameters(self):
            return (
                [val.cpu().numpy() for _, val in agent.impl.policy.state_dict().items()]
                + [
                    val.cpu().numpy()
                    for _, val in agent.impl.q_function.state_dict().items()
                ]
                + [
                    val.cpu().numpy()
                    for _, val in agent.impl._imitator.state_dict().items()
                ]
            )

        def set_parameters(self, parameters):
            policy_len = len(agent.impl.policy.state_dict())
            q_len = len(agent.impl.q_function.state_dict())
            policy_param, q_param, imitator_param = (
                parameters[:policy_len],
                parameters[policy_len : policy_len + q_len],
                parameters[policy_len + q_len :],
            )

            # policy
            policy_params_dict = zip(
                agent.impl.policy.state_dict().keys(), policy_param
            )
            policy_state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in policy_params_dict}
            )
            agent.impl.policy.load_state_dict(policy_state_dict, strict=True)

            # q function
            qfunc_params_dict = zip(agent.impl.q_function.state_dict().keys(), q_param)
            qfunc_state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in qfunc_params_dict}
            )
            agent.impl.q_function.load_state_dict(qfunc_state_dict, strict=True)

            # imitator
            imitator_params_dict = zip(
                agent.impl._imitator.state_dict().keys(), imitator_param
            )
            imitator_state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in imitator_params_dict}
            )
            agent.impl._imitator.load_state_dict(imitator_state_dict, strict=True)

            return self.get_parameters(), len_dataset, {}

    class CQLClient(D3rlpyBaseClient):
        client_setting = f"{date}_CQL_{args.dataset_path}"

    client_dict = {
        "cql": CQLClient,
        "bcq": BCQClient,
    }
    client_cls = client_dict[args.client_algo]

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=client_cls())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env_name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--dataset_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--client_algo",
        type=str,
        default=argparse.SUPPRESS,
        choices=["cql", "bcq"],
    )
    parser.add_argument("--num_local_epoch", type=int, default=argparse.SUPPRESS)

    args = parser.parse_args()
    main(args)
