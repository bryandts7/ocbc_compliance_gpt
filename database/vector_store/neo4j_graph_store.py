from langchain_community.graphs import Neo4jGraph


class Neo4jGraphStore:
    def __init__(self, config: dict):
        self.config = config
        self.graph = Neo4jGraph(url=config["neo4j_uri"], username=config["neo4j_username"], password=config["neo4j_password"], database=config["neo4j_db"])

    def get_graph(self):
        return self.graph