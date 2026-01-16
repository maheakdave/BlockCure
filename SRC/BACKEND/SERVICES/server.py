from ..UTILS.p2p import P2PNetwork,NodeConfig
from ..UTILS.log import get_logger
import yaml
import os

if __name__ == "__main__":

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../config.yaml")
    with open(path,'r') as file:
        config = yaml.safe_load(file)
    
    logger = get_logger().bind(component="P2PNetwork")
    
    try:
        nodes = [
            NodeConfig(location=node["location"],port=node["port"])
            for node in config["p2p_nodes"]
        ]
        network = P2PNetwork(
        host="localhost",
        nodes=nodes,
        logger=logger
        )
        network.setup_network()
        network.start()
    except Exception as e:
        logger.error(f"Trouble starting P2P Network with config: {config['p2p_nodes']}\nError: {e}")
        raise