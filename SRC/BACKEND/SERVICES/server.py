from UTILS.p2p import P2PNetwork,NodeConfig
from UTILS.log import get_logger


if __name__ == "__main__":

    logger = get_logger().bind(component="P2PNetwork")

    nodes = [
        NodeConfig(location="node1", port=8001),
        NodeConfig(location="node2", port=8002),
        NodeConfig(location="node3", port=8003),
    ]
    
    network = P2PNetwork(
    host="localhost",
    nodes=nodes,
    logger=logger
    )

    network.setup_network()
    network.start()