import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
from typing import List, Dict, Union
import json

class RAGChatbot:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        llm_model: str = "llama3.2:latest",
        index_name: str = "embedding_index",
        doc_prefix: str = "doc:"
    ):
        """
        Initialize the RAG Chatbot with specified configuration.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            llm_model: Name of the LLM model to use
            index_name: Name of the Redis index
            doc_prefix: Prefix for document keys in Redis
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.llm_model = llm_model
        self.index_name = index_name
        self.doc_prefix = doc_prefix

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text using Ollama.
        
        Args:
            text: The input text to be embedded
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = ollama.embeddings(model='nomic-embed-text', prompt=text)
            return response["embedding"]
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")

    def generate_suggested_questions(self, response: str) -> List[str]:
        """
        Generate suggested follow-up questions based on the response.
        
        Args:
            response: The bot's response to generate questions from
            
        Returns:
            List of suggested follow-up questions
        """
        try:
            prompt = f"""Based on the following response, generate 3-4 relevant follow-up questions that a user might ask.
            The questions should be specific, clear, and directly related to the information provided.
            Format the response as a JSON array of strings.
            
            Response:
            {response}
            
            Example format:
            ["What are the specific AI tools used in investment analysis?", "How does human oversight work with AI tools?", "What are the main risks of using AI in investment decisions?"]
            """
            
            result = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates relevant follow-up questions."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            # Parse the response to get the questions
            questions = json.loads(result["message"]["content"])
            return questions[:4]  # Return at most 4 questions
            
        except Exception as e:
            print(f"Error generating suggested questions: {str(e)}")
            return []

    def query_llm(self, query: str, matching_chunks: List[str]) -> str:
        """
        Query the Language Model with a given question and relevant context.
        
        Args:
            query: The user's question
            matching_chunks: List of relevant text chunks
            
        Returns:
            The model's response as a string
        """
        try:
            context = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(matching_chunks)])
            prompt_to_send = (
                f"User's Question: {query}\n\n"
                f"Relevant Context (if applicable):\n{context}\n\n"
                "Your task: Answer the user's question as clearly and accurately as possible."
                "If the question is unclear or not actually a question, state that explicitly."
            )
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an HR representative for Arrowstreet Capital, a Boston-based Quantitative Investment and Asset Manager. 
                        You are tasked with assisting new hirings through the onboarding process and general questions about the firm's operations.
                        
                        Please format your responses in a clear, readable way:
                        1. Use **bold** for important terms and key points
                        2. Use bullet points (•) for lists and steps
                        3. Break down complex information into sections
                        4. Use clear headings when appropriate
                        5. Keep paragraphs short and focused
                        6. Use simple, professional language
                        
                        Example format:
                        **Key Point:** [Important information]
                        
                        • First step
                        • Second step
                        • Third step
                        
                        **Additional Information:**
                        [Supporting details]
                        
                        Please only use the context you are given."""
                    },
                    {"role": "user", "content": prompt_to_send}
                ],
            )
            return response["message"]["content"]
        except Exception as e:
            raise Exception(f"Error querying LLM: {str(e)}")

    def perform_knn_search(self, query_text: str, k: int = 5) -> Dict[str, Union[str, List[str]]]:
        """
        Perform a K-Nearest Neighbors search and get a response from the LLM.
        
        Args:
            query_text: The text query to search for
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Dictionary containing the response and matching chunks
        """
        try:
            # Generate embedding for the query
            query_embedding = self.get_embedding(query_text)
            
            q = (
                Query(f"*=>[KNN {k} @embedding $vec AS vector_distance]")
                .sort_by("vector_distance")
                .return_fields("text", "vector_distance")
                .dialect(2)
            )
            res = self.redis_client.ft(self.index_name).search(
                q, query_params={"vec": np.array(query_embedding, dtype=np.float32).tobytes()}
            )
            
            matching_chunks = [doc.text for doc in res.docs]
            if not matching_chunks:
                return {
                    "response": "I couldn't find any relevant information to answer your question.",
                    "matching_chunks": []
                }
            
            response = self.query_llm(query_text, matching_chunks)
            
            return {
                "response": response,
                "matching_chunks": matching_chunks
            }
        except Exception as e:
            raise Exception(f"Error performing KNN search: {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Initialize the chatbot with default settings
    chatbot = RAGChatbot()
    
    # Example query
    result = chatbot.perform_knn_search("What is the company's mission?")
    print(result["response"])