import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever

load_dotenv()

#? connect to Neo4j database
URL = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD = "Haya_ahm08"
driver = GraphDatabase.driver(URL, auth=(USERNAME, PASSWORD))

driver.execute_query("""
MERGE (f:Person {name: 'Frodo Baggins', country: 'Australia'})
MERGE (l:Person {name: 'Linus Torvalds', country: 'Finland'})
MERGE (c:YTChannel {name:'NeuralNine', subscribers: 1000000})
MERGE (o:OS {name:'Linux', type: 'Operating System'})

MERGE (f)-[:OWNS]->(c)
MERGE (l)-[:CREATED]->(o)
MERGE (f)-[:USES]->(o)
""")

SCHEMA = """
Node_labels:
    Person(name, country)
    YTChannel(name, subscribers)
    OS(name, type)
Relationships:
    Person-[:OWNS]->YTChannel
    Person-[:CREATED]->OS
    Person-[:USES]->OS
"""

retriever = Text2CypherRetriever(
    driver=driver,
    llm=OpenAILLM(model_name="gpt-4o-mini"),
    neo4j_schema=SCHEMA
)

@tool
def query_KG(query: str) -> str:
    """Query the knowledge graph for information. You can pass entire user questions / queries. Returns graph rows."""
    result= retriever.run(query)
    return '/n'.join(item.content for item in result.items) or '(no content)'

agent = create_agent(
    model="gpt-4o-mini",
    tools=[query_KG],
    system_prompt='You are a helpful assistant that can answer questions about a knowledge graph. You have access to a tool called query_KG that allows you to query the knowledge graph for information. Use this tool to answer user questions.'
)

if __name__ == "__main__":
    q = 'What country is the creator of the operating system used by the person who runs NeuralNine from?'
    response = agent.invoke({'messages': [('user', q)]})
    print(response['output_text'])