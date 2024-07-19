from abc import ABC, abstractmethod
from enum import Enum
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_community.vectorstores.redis import Redis
import redis

import os

from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch

class Database(Enum):
    PINECONE = 'pinecone'
    REDIS = 'redis'


class VectorIndexManager(ABC):
    def __init__(self, embed_model, index_name="ojk"):
        self.embed_model = embed_model
        self.index_name = index_name
        self.vector_store = None

    @abstractmethod
    def store_vector_index(self, docs=None, delete=False):
        pass

    @abstractmethod
    def load_vector_index(self):
        pass


class PineconeIndexManager(VectorIndexManager):
    def __init__(self, embed_model, index_name="ojk", config: dict = {}):
        super().__init__(embed_model, index_name)
        self.api_key = config["pinecone_api_key"]
        self.pc = Pinecone(api_key=self.api_key)
        self._create_index_if_not_exists()
        self.index = self.pc.Index(self.index_name)

    def _create_index_if_not_exists(self):
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

    def store_vector_index(self, docs, delete=False):
        log_path = './logs'

        if delete:
            self.index.delete(delete_all=True)

        self.vector_store = PineconeVectorStore(index=self.pc.Index(self.index_name), embedding=self.embed_model)

        time.sleep(4)

        if os.path.exists(os.path.join(log_path, 'start_store_idx.txt')):
            with open(os.path.join(log_path, 'start_store_idx.txt'), 'r') as f:
                start_split_idx = int(f.read())
            print(f"Start loading from idx: {start_split_idx}")

        for i in range(start_split_idx, len(docs), 200):
            documents = docs[i:i+200]
            self.vector_store.add_documents(documents)
            time.sleep(4)
            last_split_idx = i+200
            if last_split_idx > len(docs):
                last_split_idx = len(docs)
            with open(os.path.join(log_path, 'start_store_idx.txt'), 'w') as f:
                f.write(str(last_split_idx))
            print(f"Loaded {i+1}-{last_split_idx} documents")

    def load_vector_index(self):
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            raise ValueError(f"Index {self.index_name} does not exist.")

        self.vector_store = PineconeVectorStore(index=self.pc.Index(self.index_name), embedding=self.embed_model)
        return self.vector_store


class RedisIndexManager(VectorIndexManager):
    def __init__(self, embed_model, host='localhost', port=6379, db=0, index_name="ojk"):
        super().__init__(embed_model, index_name)
        self.redis_client = redis.Redis(host=host, port=port, db=db)

    def _create_index_if_not_exists(self):
        if not self.redis_client.exists(self.index_name):
            self.redis_client.hset(self.index_name, mapping={})

    def store_vector_index(self, docs=None, delete=False):
        self._create_index_if_not_exists()
        self.vector_store = Redis.from_documents(
            documents=docs,
            redis_url="redis://localhost:6379",
            index_name=self.index_name,
            embedding=self.embed_model,
        )
        return self.vector_store

    def load_vector_index(self):
        if not self.redis_client.exists(self.index_name):
            raise ValueError(f"Index {self.index_name} does not exist.")

        self.vector_store = Redis(index_name=self.index_name, redis_url="redis://localhost:6379", embedding=self.embed_model)
        return self.vector_store
    

class MongoAtlasIndexManager(VectorIndexManager):
    def __init__(self, embed_model, uri, index_name="ojk", db_name="vector_db", collection_name="vector_index_bi_ojk"):
        super().__init__(embed_model, index_name)
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.mongodb_client = MongoClient(uri)
        self.db = self.mongodb_client[db_name]
        self.collection = self.db[collection_name]

    def store_vector_index(self, docs=None, delete=False):
        if delete:
            # Delete all documents in the collection
            delete_result = self.collection.delete_many({})
            print(
                f"Deleted {delete_result.deleted_count} index from the collection.")

        vectorstore = MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            collection=self.collection,
            embedding=self.embed_model,
            index_name=self.index_name,
        )

        return vectorstore

    def load_vector_index(self):
        if not self.collection.find_one():
            raise ValueError(f"Index {self.index_name} does not exist.")

        vectorstore = MongoDBAtlasVectorSearch(
            collection=self.collection, embedding=self.embed_model, index_name=self.index_name)
        return vectorstore