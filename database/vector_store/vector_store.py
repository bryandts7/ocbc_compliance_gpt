from abc import ABC, abstractmethod
import time
from langchain_core.embeddings import Embeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

import psycopg2
from langchain_community.vectorstores import PGVector
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from langchain_community.vectorstores.redis import Redis
import redis

from elasticsearch import Elasticsearch, NotFoundError as ESNotFoundError
from langchain_community.vectorstores import ElasticsearchStore
from langchain_elasticsearch import DenseVectorStrategy


import os


class VectorIndexManager(ABC):
    def __init__(self, embed_model, index_name="ojk"):
        self.embed_model = embed_model
        self.index_name = index_name
        self.vector_store = None

    @abstractmethod
    def delete_index(self):
        pass

    @abstractmethod
    def store_vector_index(self, docs=None, delete=False):
        pass

    @abstractmethod
    def load_vector_index(self):
        pass


# ================== ELASTIC SEARCH ==================
class ElasticIndexManager(VectorIndexManager):
    def __init__(self, embed_model, index_name="ojk", config=None):
        super().__init__(embed_model, index_name)
        self.host = config["es_uri"]
        self.user = config["es_username"]
        self.password = config["es_password"]

        self.es_client = Elasticsearch(
            hosts=self.host,
            basic_auth=(self.user, self.password),
            max_retries=10,
            verify_certs=False,
            retry_on_timeout=True,
        )

    def delete_index(self):
        log_path = './database/store_logs'
        try:
            self.es_client.indices.delete(index=self.index_name)
            print(f"Deleted index '{self.index_name}'.")
            with open(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt'), 'w') as f:
                f.write(str(0))
        except Exception as e:
            print(f"Error deleting index: {e}")

    def store_vector_index(self, docs, batch_size=200):
        log_path = './database/store_logs'

        if os.path.exists(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt')):
            with open(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt'), 'r') as f:
                start_split_idx = int(f.read())
            print(f"Start loading from idx: {start_split_idx}")
        else:
            with open(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt'), 'w') as f:
                f.write(str(0))
            start_split_idx = 0

        if start_split_idx < batch_size:
            self.vector_store = ElasticsearchStore.from_documents(
                documents=docs[0:batch_size],
                embedding=self.embed_model,
                es_connection=self.es_client,
                index_name=self.index_name,
                strategy=DenseVectorStrategy(hybrid=True)
            )
            time.sleep(4)

            last_split_idx = min(batch_size, len(docs))
            with open(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt'), 'w') as f:
                f.write(str(last_split_idx))
            print(f"Loaded 1-{last_split_idx} documents")
            start_split_idx = batch_size
        else:
            self.vector_store = ElasticsearchStore(
                embedding=self.embed_model,
                es_connection=self.es_client,
                index_name=self.index_name,
                strategy=DenseVectorStrategy(hybrid=True)
            )

        for i in range(start_split_idx, len(docs), batch_size):
            documents = docs[i:i+batch_size]
            self.vector_store.add_documents(documents)
            time.sleep(4)
            last_split_idx = min(i+batch_size, len(docs))
            with open(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt'), 'w') as f:
                f.write(str(last_split_idx))
            print(f"Loaded {i+1}-{last_split_idx} documents")

    def load_vector_index(self):
        self.vector_store = ElasticsearchStore(
            embedding=self.embed_model,
            es_connection=self.es_client,
            index_name=self.index_name
        )
        print(f"Loaded index '{self.index_name}'.")
        return self.vector_store

# ================== POSTGRE ==================


class PostgresIndexManager(VectorIndexManager):
    def __init__(self, embed_model: Embeddings, index_name: str = "ojk", config: dict = {}):
        super().__init__(embed_model, index_name)
        self.base_connection_string = config["postgres_uri"]
        self.db_name = "vector_store"
        self.connection_string = f"{self.base_connection_string}/{self.db_name}"
        self.collection_name = f"{index_name}_collection"
        self._create_database_and_extension()

    def _create_database_and_extension(self):
        # Connect to default 'postgres' database to create a new database
        conn = psycopg2.connect(f"{self.base_connection_string}")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        try:
            # Check if database exists
            cur.execute(
                f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (self.db_name,))
            exists = cur.fetchone()
            if not exists:
                cur.execute(f"CREATE DATABASE {self.db_name}")
                print(f"Database '{self.db_name}' created successfully.")
            else:
                print(f"Database '{self.db_name}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the database: {e}")

        cur.close()
        conn.close()

        # Connect to the new database to create the extension
        conn = psycopg2.connect(self.connection_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("Vector extension created successfully (if it didn't exist).")
        except Exception as e:
            print(
                f"An error occurred while creating the vector extension: {e}")

        cur.close()
        conn.close()

    def delete_index(self):
        log_path = './database/store_logs'
        try:
            PGVector.from_existing_index(
                embedding=self.embed_model,
                collection_name=self.collection_name,
                connection_string=self.connection_string
            ).delete_collection()
            with open(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt'), 'w') as f:
                f.write(str(0))
            print(f"Deleted collection '{self.collection_name}'.")
        except Exception as e:
            print(f"Error deleting collection: {e}")

    def store_vector_index(self, docs, batch_size: int = 200):
        log_path = './database/store_logs'

        if os.path.exists(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt')):
            with open(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt'), 'r') as f:
                start_split_idx = int(f.read())
            print(f"Start loading from idx: {start_split_idx}")
        else:
            with open(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt'), 'w') as f:
                f.write(str(0))
            start_split_idx = 0

        if start_split_idx < batch_size:
            self.vector_store = PGVector.from_documents(
                documents=docs[0:batch_size],
                embedding=self.embed_model,
                collection_name=self.collection_name,
                connection_string=self.connection_string
            )
            time.sleep(4)

            last_split_idx = min(batch_size, len(docs))
            with open(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt'), 'w') as f:
                f.write(str(last_split_idx))
            print(f"Loaded 1-{last_split_idx} documents")
            start_split_idx = batch_size
        else:
            self.vector_store = PGVector.from_existing_index(
                embedding=self.embed_model,
                collection_name=self.collection_name,
                connection_string=self.connection_string
            )

        for i in range(start_split_idx, len(docs), batch_size):
            documents = docs[i:i+batch_size]
            self.vector_store.add_documents(documents)
            time.sleep(4)
            last_split_idx = min(i+batch_size, len(docs))
            with open(os.path.join(log_path, f'start_store_idx_{self.index_name}.txt'), 'w') as f:
                f.write(str(last_split_idx))
            print(f"Loaded {i+1}-{last_split_idx} documents")

    def load_vector_index(self):
        self.vector_store = PGVector.from_existing_index(
            embedding=self.embed_model,
            collection_name=self.collection_name,
            connection_string=self.connection_string
        )
        print(f"Loaded collection '{self.collection_name}'.")
        return self.vector_store

# ================== REDIS ==================


class RedisIndexManager(VectorIndexManager):
    def __init__(self, embed_model: Embeddings, index_name="ojk", config: dict = {}, db_id: int = 0):
        super().__init__(embed_model, index_name)
        self.redis_uri = config["redis_uri"] + "/" + str(db_id)
        self.redis_client = redis.from_url(self.redis_uri)
        self.index_name = index_name
        self.embed_model = embed_model

    def delete_index(self):
        log_path = './database/store_logs'
        # Delete the index
        try:
            self.redis_client.ft(self.index_name).dropindex(
                delete_documents=True)
            print(
                f"Deleted index '{self.index_name}' and its associated documents.")
            with open(os.path.join(log_path, 'start_store_idx_' + self.index_name + '.txt'), 'w') as f:
                f.write(str(0))
        except redis.exceptions.ResponseError as e:
            if "Unknown index name" in str(e):
                print(f"Index '{self.index_name}' does not exist.")
            else:
                raise

    def store_vector_index(self, docs, batch_size=200):
        log_path = './database/store_logs'
        batch_size = batch_size

        if os.path.exists(os.path.join(log_path, 'start_store_idx_' + self.index_name + '.txt')):
            with open(os.path.join(log_path, 'start_store_idx_' + self.index_name + '.txt'), 'r') as f:
                idx = int(f.read())
                start_split_idx = idx
            if start_split_idx < batch_size:
                start_split_idx = 0
            print(f"Start loading from idx: {idx}")
        else:
            # crreate log file
            with open(os.path.join(log_path, 'start_store_idx_' + self.index_name + '.txt'), 'w') as f:
                f.write(str(0))
            start_split_idx = 0

        if start_split_idx < batch_size:
            self.vector_store = Redis.from_documents(
                documents=docs[0:batch_size],
                embedding=self.embed_model,
                redis_url=self.redis_uri,
                index_name=self.index_name

            )
            time.sleep(4)

            if len(docs) <= batch_size:
                with open(os.path.join(log_path, 'start_store_idx_' + self.index_name + '.txt'), 'w') as f:
                    f.write(str(len(docs)))
                print(f"Loaded 1-{len(docs)} documents")
            else:
                with open(os.path.join(log_path, 'start_store_idx_' + self.index_name + '.txt'), 'w') as f:
                    f.write(str(batch_size))
                print(f"Loaded 1-{batch_size} documents")
            start_split_idx = batch_size

            self.vector_store.write_schema(
                './database/vector_store/redis_schema/vectorstore_redis_schema_' + self.index_name + '.yaml')
        else:
            self.vector_store = Redis.from_existing_index(
                index_name=self.index_name,
                redis_url=self.redis_uri,
                embedding=self.embed_model,
                schema="./database/vector_store/redis_schema/vectorstore_redis_schema_" +
                self.index_name + '.yaml',
            )

        for i in range(start_split_idx, len(docs), batch_size):
            documents = docs[i:i+batch_size]
            self.vector_store.add_documents(documents)
            time.sleep(4)
            last_split_idx = min(i+batch_size, len(docs))
            with open(os.path.join(log_path, 'start_store_idx_' + self.index_name + '.txt'), 'w') as f:
                f.write(str(last_split_idx))
            print(f"Loaded {i+1}-{last_split_idx} documents")

    def load_vector_index(self):
        self.vector_store = Redis.from_existing_index(
            index_name=self.index_name,
            redis_url=self.redis_uri,
            embedding=self.embed_model,
            schema="./database/vector_store/redis_schema/vectorstore_redis_schema_" +
            self.index_name + '.yaml',
        )
        print(f"Loaded index '{self.index_name}'.")
        return self.vector_store


# ================== PINECONE ==================
class PineconeIndexManager(VectorIndexManager):
    def __init__(self, embed_model: Embeddings, index_name="ojk", config: dict = {}):
        super().__init__(embed_model, index_name)
        self.api_key = config["pinecone_api_key"]
        self.pc = Pinecone(api_key=self.api_key)
        self._create_index_if_not_exists()
        self.index = self.pc.Index(self.index_name)

    def delete_index(self):
        log_path = './database/store_logs'
        self.index.delete(delete_all=True)
        with open(os.path.join(log_path, 'start_store_idx_' + self.index_name + '.txt'), 'w') as f:
            f.write(str(0))
        print("Deleted all keys in the database.")

    def _create_index_if_not_exists(self):
        existing_indexes = [index_info["name"]
                            for index_info in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

    def store_vector_index(self, docs):
        log_path = './logs'

        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name), embedding=self.embed_model)

        time.sleep(4)

        if os.path.exists(os.path.join(log_path, 'start_store_idx_' + self.index_name + '.txt')):
            with open(os.path.join(log_path, 'start_store_idx_' + self.index_name + '.txt'), 'r') as f:
                start_split_idx = int(f.read())
            print(f"Start loading from idx: {start_split_idx}")

        for i in range(start_split_idx, len(docs), 200):
            documents = docs[i:i+200]
            self.vector_store.add_documents(documents)
            time.sleep(4)
            last_split_idx = i+200
            if last_split_idx > len(docs):
                last_split_idx = len(docs)
            with open(os.path.join(log_path, 'start_store_idx_' + self.index_name + '.txt'), 'w') as f:
                f.write(str(last_split_idx))
            print(f"Loaded {i+1}-{last_split_idx} documents")

    def load_vector_index(self):
        existing_indexes = [index_info["name"]
                            for index_info in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            raise ValueError(f"Index {self.index_name} does not exist.")

        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name), embedding=self.embed_model)
        print(f"Loaded index '{self.index_name}'.")
        return self.vector_store
