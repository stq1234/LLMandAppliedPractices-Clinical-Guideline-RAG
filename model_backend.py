"""Embedding and grounded generation clients for the clinical-guideline prototype."""
import os
import threading
import time

import numpy as np
import requests


QUERY_TASK = ("Given a question on adult blood pressure measurement, screening or diagnosis, "
              "retrieve applicable clinical guideline recommendations and systematic-review evidence, "
              "matching the population, named authority, setting and measurement method.")


class Backend:
    def __init__(self):
        self.embed_url=os.getenv("EMBED_URL","http://127.0.0.1:8765").rstrip("/")
        self.chat_url=os.getenv("CHAT_URL","http://127.0.0.1:18882/v1").rstrip("/")
        self.model=os.getenv("CHAT_MODEL","qwen3.6-35b-a3b")
        self._local=threading.local()
        res=self._session().get(self.embed_url+"/health",timeout=(5,15))
        res.raise_for_status()
        health=res.json()
        self.embedding={k:health[k] for k in ("model","revision","pooling","max_length","normalized")}
        if self.embedding["pooling"]!="last_token" or not self.embedding["normalized"]:
            raise RuntimeError("Incompatible embedding service.")
        self.embedding_signature="|".join(map(str,self.embedding.values()))+"|"+QUERY_TASK
        self.dim=4096

    def _session(self):
        if not hasattr(self._local,"session"):
            session=requests.Session()
            session.trust_env=False
            self._local.session=session
        return self._local.session

    def reset_usage(self):
        self._local.stats={"chat_calls":0,"doc_vectors":0,"query_vectors":0,"prompt_tokens":0,
                            "completion_tokens":0,"output_tokens":0,"chat_seconds":0.0,
                            "embedding_seconds":0.0,"length_limited":0}
    def usage(self):
        if not hasattr(self._local,"stats"):
            self.reset_usage()
        return self._local.stats

    def encode(self,texts,query=False):
        texts=list(texts)
        vectors=[]
        for start in range(0,len(texts),12):
            batch=texts[start:start+12]
            payload=[f"Instruct: {QUERY_TASK}\nQuery:{t}" for t in batch] if query else batch
            started=time.perf_counter()
            response=self._session().post(self.embed_url+"/embed",
                          json={"kind":"document","texts":payload},timeout=(10,180))
            response.raise_for_status()
            obj=response.json()
            if (obj["model"],obj["revision"])!=(self.embedding["model"],self.embedding["revision"]):
                raise RuntimeError("Embedding revision changed.")
            a=np.asarray(obj["vectors"],dtype=np.float32)
            if a.shape!=(len(batch),self.dim) or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1.0,atol=1e-4):
                raise RuntimeError("Embedding response has incorrect shape or normalization.")
            vectors.append(a)
            stats=self.usage()
            stats["query_vectors" if query else "doc_vectors"]+=len(batch)
            stats["embedding_seconds"]+=time.perf_counter()-started
        return np.concatenate(vectors) if vectors else np.empty((0,self.dim),dtype=np.float32)

    def chat(self, system, user):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        config = {"model": self.model, "messages": messages, "temperature": 0, "top_p": 1,
                  "chat_template_kwargs": {"enable_thinking": False}}
        stats = self.usage()
        for output_limit in (768, 2048):
            started = time.perf_counter()
            response = self._session().post(
                self.chat_url + "/chat/completions",
                json={**config, "max_tokens": output_limit}, timeout=(10, 240),
            )
            response.raise_for_status()
            obj = response.json()
            choice = obj["choices"][0]
            stats["chat_calls"] += 1
            stats["prompt_tokens"] += obj["usage"]["prompt_tokens"]
            stats["completion_tokens"] += obj["usage"]["completion_tokens"]
            stats["output_tokens"] += obj["usage"]["completion_tokens"]
            stats["chat_seconds"] += time.perf_counter() - started
            if choice.get("finish_reason") == "length":
                stats["length_limited"] += 1
                continue
            answer = choice["message"].get("content")
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("The model returned no answer content.")
            return answer.strip()
        raise RuntimeError("Model output remained truncated after the extended completion.")
