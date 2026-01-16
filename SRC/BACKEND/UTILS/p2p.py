from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import threading
from pydantic import BaseModel,ValidationError,Field,TypeAdapter
from typing import Union,Literal
from websockets.asyncio.client import connect
import json
from datetime import datetime
from hashlib import sha256
import uvicorn
import httpx
from dataclasses import dataclass
from .log import StructlogMiddleware,get_logger
from .database import lifespan

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
            
class BaseMessage(BaseModel):
    origin: str
    message_id: str = Field(default_factory=lambda: sha256(str(datetime.now()).encode()).hexdigest()[:12])

class TransactionPayload(BaseMessage):
    type: Literal["transaction"]
    id: str
    diagnostics: str
    symptoms: str
    treatment: str

class BlockPayload(BaseMessage):
    type: Literal["block_proposal"]
    block_data: dict # Or a specific Block model
    signature: str

class QueryPayload(BaseMessage):
    type: Literal["query_request"]
    target_record_id: str          

PeerMessage = Union[TransactionPayload, BlockPayload, QueryPayload]

class Peer():
    def __init__(self,location:str,port:int,peers:list[str]) -> None:
        self.blockchain = Blockchain()
        self.location = location
        self.peers:list[str] = peers
        self.port = port
        self._app = FastAPI(
            docs_url=None, 
            redoc_url=None, 
            openapi_url=None,
            lifespan=lifespan,
        )
        self._app.add_middleware(StructlogMiddleware)

        # Will modify allowed origins later
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.http_client = httpx.AsyncClient()
        self.logger = get_logger().bind(peer_location=self.location)
        self.dispatch_map = {
            "transaction":self._handle_transaction,
            "block_proposal":self._handle_block_proposal,
            "query_request":self._handle_query_request,
        }

        @self._app.websocket(f"/ws/{self.location}")
        async def websocket_endpoint(websocket: WebSocket)->None:

                await websocket.accept()
                try:
                    while True:
                            raw_data = await websocket.receive_json()
                            try:
                                message = TypeAdapter(PeerMessage).validate_python(raw_data)
                                handler = self.dispatch_map.get(message.type)
                                
                                if handler:
                                    await handler(message)
                                else:
                                    self.logger.warning(f"No handler for type: {message.type}")

                            except ValidationError as e:
                                self.logger.warning(f"Invalid message received: {e}")
                                await websocket.close(code=1003)
                                return
                            
                            except ValidationError as e:
                                self.logger.error(f"Schema mismatch: {e.json()}")
                                continue

                except WebSocketDisconnect:
                    self.logger.info("WebSocket disconnected, Successfull Transmission")
    
    async def _handle_transaction(self,message:PeerMessage)->None:
        self.logger.info("Adding new transaction to mempool")
        self.blockchain.curr_block.addTransaction(Transaction(message.origin, message.model_dump()))
        await self.broadcast_to_peers()
    
    async def broadcast_to_peers(self,message:PeerMessage)->None:
        for peer_uri in self.peers:
            async with connect(peer_uri) as websocket:
                await websocket.send(json.dumps(message.model_dump(),sort_keys=True))

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
        