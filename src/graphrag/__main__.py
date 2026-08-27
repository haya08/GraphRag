import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase
from neo4j_graphrag.llm import LLMBase, LLMResponse
from neo4j_graphrag.retrievers import Text2CypherRetriever


load_dotenv()


# ============================================================
# 1. Connect to Neo4j
# ============================================================

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URL"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

driver.verify_connectivity()

print("Connected to Neo4j!")


# ============================================================
# 2. Create Knowledge Graph
# ============================================================

driver.execute_query(
    """
    MERGE (f:Person {
        name: 'Frodo Baggins',
        country: 'Australia'
    })

    MERGE (l:Person {
        name: 'Linus Torvalds',
        country: 'Finland'
    })

    MERGE (c:YTChannel {
        name: 'NeuralNine',
        subscribers: 1000000
    })

    MERGE (o:OS {
        name: 'Linux',
        type: 'Operating System'
    })

    MERGE (f)-[:OWNS]->(c)
    MERGE (l)-[:CREATED]->(o)
    MERGE (f)-[:USES]->(o)
    """,
    database_="graphrag"
)


# ============================================================
# 3. Neo4j Schema
# ============================================================

SCHEMA = """
Node labels:

Person(name, country)
YTChannel(name, subscribers)
OS(name, type)

Relationships:

Person-[:OWNS]->YTChannel
Person-[:CREATED]->OS
Person-[:USES]->OS
"""


# ============================================================
# 4. Gemini
# ============================================================

gemini = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)


# ============================================================
# 5. Gemini -> Neo4j GraphRAG Adapter
# ============================================================

def extract_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    return str(content)


class GeminiLLM(LLMBase):

    def __init__(self):
        super().__init__(
            model_name="gemini-3.6-flash"
        )

    def invoke(
        self,
        input,
        message_history=None,
        system_instruction=None,
        **kwargs
    ):
        prompt = input

        if system_instruction:
            prompt = f"""
System instruction:

{system_instruction}

User:

{input}
"""

        response = gemini.invoke(prompt)

        return LLMResponse(
            content=extract_text(response.content)
        )

    async def ainvoke(
        self,
        input,
        message_history=None,
        system_instruction=None,
        **kwargs
    ):
        prompt = input

        if system_instruction:
            prompt = f"""
System instruction:

{system_instruction}

User:

{input}
"""

        response = await gemini.ainvoke(prompt)

        return LLMResponse(
            content=extract_text(response.content)
        )


gemini_graphrag = GeminiLLM()


# ============================================================
# 6. Text2Cypher Retriever
# ============================================================

retriever = Text2CypherRetriever(
    driver=driver,
    llm=gemini_graphrag,
    neo4j_schema=SCHEMA,
    neo4j_database="graphrag"
)


# ============================================================
# 7. Ask a question
# ============================================================

if __name__ == "__main__":

    q = """
    What country is the creator of the operating system
    used by the person who runs NeuralNine from?
    """

    result = retriever.search(q)


    # ========================================================
    # 8. Generated Cypher
    # ========================================================

    print("\n" + "=" * 60)
    print("GENERATED CYPHER")
    print("=" * 60)

    print(result.metadata.get("cypher"))


    # ========================================================
    # 9. Neo4j Result
    # ========================================================

    print("\n" + "=" * 60)
    print("NEO4J RESULT")
    print("=" * 60)

    for item in result.items:
        print(item.content)


    # ========================================================
    # 10. Cleanup
    # ========================================================

    driver.close()