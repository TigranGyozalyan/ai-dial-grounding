import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient


SYSTEM_PROMPT = """
You are a RAG-enabled assistant. You need to provide an answer based on the question and the provided context.

- ## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
- Be conversational and helpful in your responses.
- When presenting user information, format it clearly and include relevant details.
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}"""


def format_user_document(user: dict[str, Any]) -> str:
    return f"""
                User:
                  name: {user['name']}
                  surname: {user['surname']}
                  about_me: {user['about_me']}
                  gender: {user['gender']}
                  email: {user['email']}
                  company: {user['company']}
                  salary: {user['salary']}
                """

def as_batches(users: list[Any], batch_size: int = 100) -> list[list[Any]]:
    batches = []
    idx = 0
    for user in users:
        batch_idx = int(idx / batch_size)
        if batch_idx not in range(len(batches)):
            batches.append([])
        batches[batch_idx].append(user)
        idx += 1
    return batches

class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        user_client = UserClient()
        users = user_client.get_all_users()
        docs = [Document(page_content=format_user_document(user)) for user in users]

        self.vectorstore = await self._create_vectorstore_with_batching(documents=docs)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        doc_batches = as_batches(documents, batch_size)
        tasks = []
        for doc_batch in doc_batches:
            tasks.append(FAISS.afrom_documents(documents=doc_batch, embedding=self.embeddings))

        results = await asyncio.gather(*tasks)
        final_store = results[0]

        for i in range(1, len(results)):
            FAISS.merge_from(self=final_store, target=results[i])

        return final_store

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        context_parts = []
        result = FAISS.similarity_search_with_relevance_scores(self=self.vectorstore, k=k, query=query, score_threshold=score)
        for (doc, relevance_score) in result:
            context_parts.append(doc.page_content)
            print(f"document content: {doc.page_content}, relevance score: {relevance_score}")

        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        """Combine user query with retrieved context into a formatted prompt."""
        augmented = USER_PROMPT.replace("{context}", context).replace("{query}", query)
        print(f"augmented prompt: {augmented}")
        return augmented

    def generate_answer(self, augmented_prompt: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]

        result = self.llm_client.invoke(input=messages)
        print(f"response content: {result.content}")
        return result.content

async def main():

    embeddings = AzureOpenAIEmbeddings(
        api_key=SecretStr(API_KEY),
        azure_endpoint=DIAL_URL,
        azure_deployment='text-embedding-3-small-1',
        dimensions=384
    )
    llm_client = AzureChatOpenAI(
        temperature=0.0,
        azure_deployment='gpt-4o',
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            context = await rag.retrieve_context(query=user_question)
            augmented = rag.augment_prompt(query=user_question, context=context)
            rag.generate_answer(augmented_prompt=augmented)

asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce