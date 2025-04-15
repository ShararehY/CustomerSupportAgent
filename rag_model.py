# rag_model.py
import os
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CustomerSupportRAG:
    def __init__(self):
        # Initialize the language model (ChatGPT via OpenAI)
        self.llm = ChatOpenAI(temperature=0.7)

         # Load the vector store
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create the conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory
        )

    def get_response(self, query):
        """
        Get a response from the RAG model based on the query.
        """
        # Check if we need to escalate
        if self._should_escalate(query):
            return {
                "response": "I'll need to connect you with a human support agent for further assistance. Please wait while I transfer your conversation.",
                "escalate": True
            }

        # Get response from the chain
        response = self.chain.invoke({"question": query})

        return {
            "response": response["answer"],
            "escalate": False
        }

    def _should_escalate(self, query):
        """
        Use the LLM to determine if the query should be escalated to a human
        """
        escalation_prompt = f"""
        As a customer support system, evaluate if the following customer query needs to be escalated to a human agent.
        
        Customer query: "{query}"
        
        Escalate to a human if:
        1. The query involves a complex problem that requires human judgment
        2. The customer is expressing strong negative emotions or frustration
        3. The query involves a sensitive issue (e.g., billing disputes, account security)
        4. The customer has explicitly requested human assistance
        5. The query suggests a potential emergency situation
        
        Return only "escalate" or "handle" as your answer.
        """
        
        response = self.llm.invoke(escalation_prompt)
        return "escalate" in response.content.lower()


if __name__ == "__main__":
    # Example usage of CustomerSupportRAG class
    rag_model = CustomerSupportRAG()
    user_query = "What is your return policy?"
    response_data = rag_model.get_response(user_query)

    print(f"Response: {response_data['response']}")
    if response_data["escalate"]:
        print("Escalation required!")
