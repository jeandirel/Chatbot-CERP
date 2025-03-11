

import sqlite3
import sys
sys.modules["sqlite3"] = sqlite3
import chromadb

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import  Any, Dict, cast, List


# Classe d'embedding SentenceTransformer
class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    # Since we do dynamic imports we have to type this as Any
    models: Dict[str, Any] = {}
 
    def __init__(
        self,
        model_path_or_name: str = "./multilingual-e5-base",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ):
        """Initialize SentenceTransformerEmbeddingFunction.

        Args:
            model_name (str, optional): Identifier of the SentenceTransformer model, defaults to "all-MiniLM-L6-v2"
            device (str, optional): Device used for computation, defaults to "cpu"
            normalize_embeddings (bool, optional): Whether to normalize returned vectors, defaults to False
            **kwargs: Additional arguments to pass to the SentenceTransformer model.
        """
        if model_path_or_name not in self.models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ValueError(
                    "The sentence_transformers python package is not installed. Please install it with `pip install sentence_transformers`"
                )
            self.models[model_path_or_name] = SentenceTransformer(
                 model_path_or_name, device=device, **kwargs
            )
        self._model = self.models[model_path_or_name]
        self._normalize_embeddings = normalize_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        return cast(
            Embeddings,
            self._model.encode(
                list(input),
                convert_to_numpy=True,
                normalize_embeddings=self._normalize_embeddings,
            ).tolist(),
        )
#Classe d'embedding SentenceTransformer    
    def embed_query(self, query: str) -> List[float]:
        """Encode une requête de recherche en vecteur d'embedding."""
        # Assurez-vous que la requête est dans une liste pour que SentenceTransformer puisse la traiter
        query_embedding = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=self._normalize_embeddings)
        # Retournez le premier (et unique) vecteur d'embedding
        return query_embedding[0].tolist()




# this is specific to Llama-2.

system_prompt = """"Tu es un assistant serviable qui parle francais et répond toujours en francais,\
    tu utiliseras le contexte fourni pour répondre aux questions des utilisateurs.\
    Lis le contexte donné avant de répondre aux questions et réfléchis étape par étape.\
    Si tu ne peux pas répondre à une question de l'utilisateur basée sur\
    le contexte fourni, informe l'utilisateur. N'utilise aucune autre information pour répondre.\
    Fournis une réponse détaillée à la question en français, concise et bien structurée.\
    Ne te base pas sur tes connaissances personnelles et réponds toujours en francais.
         """


def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False):
    if promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        prompt_template = (
            B_INST
            + system_prompt
            + """
        
        Context: {context}
        User: {question}"""
            + E_INST
        )
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    else:
        # change this based on the model you have selected.
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return (
        prompt,
        memory,
    )