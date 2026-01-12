from fastapi import FastAPI,WebSocket,WebSocketDisconnect
import threading
from pydantic import BaseModel,ValidationError
from typing import Literal
from websockets.asyncio.client import connect
import json
from datetime import datetime
from hashlib import sha256
import uvicorn
import httpx
from dataclasses import dataclass
from log import StructlogMiddleware,get_logger

class Transaction():
    def __init__(self,origin:str,content:object) -> None:
        self._origin:str = origin
        self._timestamp = str(datetime.now().isoformat())
        self._content = content

class Block():
    def __init__(self):
        self._transactions:list[Transaction] = []
        self.currhash:str =  self.hasher(self._transactions)
        self.prevhash:str = None
        self._timetamp:datetime = datetime.now().isoformat()

    @staticmethod
    def hasher(transactions:list[Transaction])->str:
        transactions = [data.__dict__ for data in transactions]
        json_string = json.dumps(transactions) 
        utf8_bytes = json_string.encode("utf-8")
        return sha256(utf8_bytes).hexdigest()
    
    def addTransaction(self,transaction:Transaction)->None:
        self._transactions.append(transaction)
        self.currhash = self.hasher(self._transactions)
    
class Blockchain():
    def __init__(self) -> None:
        self._block_list:list[Block] = []
        self._latest_block_hash  = None
        self.curr_block = Block()

    def addBlock(self,block:Block):
        self._block_list = [*self._block_list,block]
        self._latest_block_hash = block.currhash
        prev_hash = self.curr_block.currhash
        self.curr_block = Block()
        self.curr_block.prevhash = prev_hash

class PeerMessage(BaseModel):
        type:str = Literal['broadcast','direct']
        origin:str
        id:str
        diagnostics:str
        symptoms:str
        treatment:str

        class Config:
            extra = 'forbid'
            
class Peer():
    def __init__(self,location:str,port:int,peers:list[str]) -> None:
        self.blockchain = Blockchain()
        self.location = location
        self.peers:list[str] = peers
        self.port = port
        self._app = FastAPI(
            docs_url=None, 
            redoc_url=None, 
            openapi_url=None
        )
        self._app.add_middleware(StructlogMiddleware)
        self.http_client = httpx.AsyncClient()
        self.logger = get_logger().bind(peer_location=self.location)

        @self._app.websocket(f"/ws/{self.location}")
        async def websocket_endpoint(websocket: WebSocket)->None:

                await websocket.accept()
                try:
                    while True:
                            data = await websocket.receive_json()
                            try:
                                message = PeerMessage.model_validate(data)
                            except ValidationError as e:
                                self.logger.warning(f"Invalid message received: {e}")
                                await websocket.close(code=1003)
                                return
                            if data:
                                if data['type'] == 'broadcast':
                                    
                                    self.logger.info(f"Received broadcast data: {data}|| Logging-location: {self.location}")

                                    await self.http_client.post(f"http://:8010/api",data=data)
                                    
                                    self.blockchain.curr_block.addTransaction(
                                        Transaction(
                                                    origin=message.origin,
                                                    content={
                                                        "id":message.id
                                                        ,"diagnosis":message.diagnostics
                                                        ,"symptoms":message.symptoms    
                                                        ,"treatment":message.treatment
                                                    }))
                                else:
                                    data['type'] = 'broadcast'
                                    for peer_uri in self.peers:
                                        
                                        async with connect(peer_uri) as websocket:
                                            await websocket.send(json.dumps(data,sort_keys=True))

                except WebSocketDisconnect:
                    self.logger.info("WebSocket disconnected, Successfull Transmission")
    def run(self):
        uvicorn.run(self._app,port=self.port)

@dataclass
class NodeConfig:
    location: str
    port: int

class P2PNetwork:
    def __init__(self, host: str, nodes: list[NodeConfig],logger=None)->None:
        self.host = host
        self.nodes = nodes
        self.logger = logger or get_logger().bind(component="P2PNetwork")
        self.peers: list[Peer] = []

    def _build_ws_uri(self, node: NodeConfig) -> str:
        return f"ws://{self.host}:{node.port}/ws/{node.location}"

    def setup_network(self) -> None:
        for node in self.nodes:
            peer_uris = [
                self._build_ws_uri(other)
                for other in self.nodes
                if other.location != node.location
            ]

            peer = Peer(
                location=node.location,
                port=node.port,
                peers=peer_uris,
                logger=self.logger,
            )

            self.peers.append(peer)

    def start(self) -> None:
        for peer in self.peers:
            thread = threading.Thread(
                target=peer.run,
                daemon=True
            )
            thread.start()

        self.logger.info("P2P Network started with %d nodes", len(self.peers))
        