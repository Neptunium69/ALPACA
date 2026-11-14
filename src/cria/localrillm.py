from __future__ import annotations

import base64
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
from threading import Lock,Thread
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit

DEFAULT_MODEL="qwen3-vl:4b-instruct-q4_K_M"


class VLMReciever:
    def __init__(self,host:str,model_name:str=DEFAULT_MODEL,request_timeout:float=600.0,token:str|None=None) -> None:
        self.host=host.rstrip("/")
        address=urlsplit(self.host)
        if address.scheme not in ("http","https") or not address.hostname or address.path or address.query or address.fragment or address.username or address.password:
            raise ValueError("host must be an HTTP(S) server URL, e.g. http://192.168.1.50:8765")
        if not math.isfinite(request_timeout) or request_timeout<=0:
            raise ValueError("request_timeout must be positive and finite")
        if not isinstance(model_name,str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.ollama_url=self.host
        self.model_name=model_name
        self.request_timeout=request_timeout
        self.token=token
        self._opener=urllib_request.build_opener(urllib_request.ProxyHandler({}))

    def _request_json(self,endpoint:str,payload:dict[str,Any]|None=None,method:str="POST",timeout:float|None=None)->dict[str,Any]:
        if (method,endpoint) not in (("GET","/health"),("GET","/api/version"),("POST","/api/show"),("POST","/api/chat")):
            raise ValueError("Unsupported CRIA endpoint or method")
        data=None if payload is None else json.dumps(payload,ensure_ascii=False,allow_nan=False).encode("utf-8")
        headers={"Content-Type":"application/json","Accept":"application/json"}
        if self.token:
            headers["Authorization"]=f"Bearer {self.token}"
        request=urllib_request.Request(f"{self.ollama_url}{endpoint}",data=data,method=method,headers=headers)
        try:
            with self._opener.open(request,timeout=self.request_timeout if timeout is None else timeout) as response:
                body=response.read().decode("utf-8")
        except urllib_error.HTTPError as error:
            details=error.read().decode("utf-8",errors="replace")
            raise RuntimeError(f"VLM server returned HTTP {error.code}: {details}") from error
        except (urllib_error.URLError,TimeoutError,OSError,HTTPException) as error:
            raise RuntimeError(f"Could not connect to VLM server at {self.ollama_url}: {error}") from error
        except UnicodeDecodeError as error:
            raise RuntimeError("VLM server returned invalid UTF-8") from error
        try:
            result=json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"VLM server returned invalid JSON: {body[:500]}") from error
        if not isinstance(result,dict):
            raise RuntimeError(f"VLM server returned an unexpected response: {result!r}")
        if result.get("error"):
            raise RuntimeError(f"VLM server error: {result['error']}")
        return result

    @staticmethod
    def _encode_image(image_path:str)->str:
        path=Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _convert_messages(self,messages:list[dict[str,Any]])->list[dict[str,Any]]:
        if not isinstance(messages,list) or not messages:
            raise ValueError("messages must be a non-empty list")
        converted=[]
        for message in messages:
            if not isinstance(message,dict):
                raise ValueError("Messages must be dictionaries")
            content=message.get("content")
            text_parts=[]
            images=[]
            if isinstance(content,str):
                text_parts.append(content)
                images=message.get("images",[])
                if not isinstance(images,list) or any(not isinstance(image,str) for image in images):
                    raise ValueError("images must be a list of base64 strings")
            elif isinstance(content,list):
                for part in content:
                    if not isinstance(part,dict):
                        raise ValueError("Message content parts must be dictionaries")
                    if part.get("type")=="text":
                        text=part.get("text")
                        if not isinstance(text,str):
                            raise ValueError("Text message content must be a string")
                        text_parts.append(text)
                    elif part.get("type")=="image":
                        image_path=part.get("path")
                        if not isinstance(image_path,str):
                            raise ValueError("Image message content must contain a file path")
                        images.append(self._encode_image(image_path))
                    else:
                        raise ValueError(f"Unsupported message content type: {part.get('type')!r}")
            else:
                raise ValueError("Message content must be a string or a list")
            converted_message={"role":message.get("role"),"content":"\n".join(text_parts)}
            if images:
                converted_message["images"]=images
            converted.append(converted_message)
        return converted

    def check(self)->dict[str,Any]:
        version=self._request_json("/api/version",method="GET",timeout=min(5.0,self.request_timeout))
        self._request_json("/api/show",payload={"model":self.model_name},timeout=min(30.0,self.request_timeout))
        return {"version":version.get("version","unknown"),"model":self.model_name}

    def chat(self,messages:list[dict[str,Any]],max_tok:int=512)->dict[str,Any]:
        if not isinstance(max_tok,int) or isinstance(max_tok,bool) or max_tok<1:
            raise ValueError("max_tok must be a positive integer")
        payload={
            "model":self.model_name,
            "messages":self._convert_messages(messages),
            "stream":False,
            "keep_alive":"30m",
            "options":{"num_ctx":8192,"num_predict":max_tok,"temperature":0.0,"seed":0},
        }
        response=self._request_json("/api/chat",payload=payload)
        message=response.get("message")
        if response.get("done") is not True or not isinstance(message,dict) or not isinstance(message.get("content"),str):
            raise RuntimeError(f"VLM server returned an invalid or incomplete chat response: {response}")
        return response

    def predict(self,messages:list[dict[str,Any]],max_tok:int=512)->str:
        return self.chat(messages,max_tok)["message"]["content"].strip()


class VLMHost:
    def __init__(self,reciever:str|None=None,host:str="0.0.0.0",port:int=8765,ollama_url:str="http://127.0.0.1:11434",model_name:str=DEFAULT_MODEL,request_timeout:float=600.0,token:str|None=None,max_body:int=64*1024*1024) -> None:
        if not isinstance(max_body,int) or max_body<1:
            raise ValueError("max_body must be a positive integer")
        self.reciever=reciever
        self.host=host
        self.port=port
        self.token=token
        self.max_body=max_body
        self.backend=VLMReciever(ollama_url,model_name,request_timeout)
        self.server:ThreadingHTTPServer|None=None
        self._thread:Thread|None=None
        self._inference=Lock()

    def _dispatch(self,method:str,path:str,payload:dict[str,Any]|None)->dict[str,Any]:
        if path=="/health":
            return {"status":"ready",**self.backend.check()}
        if method=="GET":
            return self.backend._request_json(path,method="GET",timeout=min(5.0,self.backend.request_timeout))
        if payload is None:
            raise ValueError("POST requests require a JSON object")
        payload=dict(payload)
        if payload.get("model",self.backend.model_name)!=self.backend.model_name:
            raise ValueError(f"This host serves model {self.backend.model_name}")
        payload["model"]=self.backend.model_name
        if path=="/api/chat":
            if payload.get("stream",False) is not False:
                raise ValueError("CRIA only supports stream=false")
            messages=payload.get("messages")
            if not isinstance(messages,list) or not messages:
                raise ValueError("messages must be a non-empty list")
            for message in messages:
                if not isinstance(message,dict) or message.get("role") not in ("system","user","assistant","tool") or not isinstance(message.get("content"),str):
                    raise ValueError("Each message must contain a valid role and string content")
                images=message.get("images",[])
                if not isinstance(images,list) or any(not isinstance(image,str) for image in images):
                    raise ValueError("images must be a list of base64 strings")
                for image in images:
                    try:
                        base64.b64decode(image,validate=True)
                    except ValueError as error:
                        raise ValueError("images must contain base64 data, not file paths") from error
            payload["stream"]=False
        result=self.backend._request_json(path,payload=payload)
        if path=="/api/chat" and (result.get("done") is not True or not isinstance(result.get("message"),dict) or not isinstance(result["message"].get("content"),str)):
            raise RuntimeError("Ollama returned an invalid or incomplete chat response")
        return result

    def start(self,blocking:bool=True)->VLMHost:
        if self.server is not None:
            raise RuntimeError("CRIA host is already running")
        self.backend.check()
        host=self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(min(30.0,host.backend.request_timeout))

            def _send_json(self,status:int,payload:dict[str,Any]):
                body=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type","application/json; charset=utf-8")
                self.send_header("Content-Length",str(len(body)))
                self.send_header("Connection","close")
                self.end_headers()
                self.wfile.write(body)

            def _handle(self):
                locked=False
                try:
                    if host.reciever and self.client_address[0]!=host.reciever:
                        self._send_json(403,{"error":"UGV address is not allowed"})
                        return
                    credential=self.headers.get("Authorization","")
                    if host.token and not secrets.compare_digest(credential.encode(),f"Bearer {host.token}".encode()):
                        self._send_json(401,{"error":"Invalid CRIA token"})
                        return
                    if (self.command,self.path) not in (("GET","/health"),("GET","/api/version"),("POST","/api/show"),("POST","/api/chat")):
                        self._send_json(404,{"error":"Unknown CRIA endpoint or method"})
                        return
                    payload=None
                    if self.command=="POST":
                        if self.headers.get("Transfer-Encoding"):
                            raise ValueError("Transfer-Encoding is unsupported; send Content-Length")
                        length=int(self.headers.get("Content-Length","0"))
                        if length<=0:
                            raise ValueError("A non-empty JSON body and Content-Length are required")
                        if length>host.max_body:
                            self._send_json(413,{"error":"Request body exceeds the CRIA size limit"})
                            return
                        body=self.rfile.read(length)
                        if len(body)!=length:
                            raise ValueError("Request body is shorter than Content-Length")
                        payload=json.loads(body.decode("utf-8"))
                        if not isinstance(payload,dict):
                            raise ValueError("Request body must be a JSON object")
                    if self.path=="/api/chat":
                        locked=host._inference.acquire(blocking=False)
                        if not locked:
                            self._send_json(503,{"error":"CRIA host is busy; inference was not queued"})
                            return
                    self._send_json(200,host._dispatch(self.command,self.path,payload))
                except (ValueError,UnicodeError) as error:
                    self._send_json(400,{"error":str(error)})
                except RuntimeError as error:
                    self._send_json(502,{"error":str(error)})
                except (TimeoutError,ConnectionError,OSError):
                    pass # The requester disconnected or stopped sending its body.
                finally:
                    if locked:
                        host._inference.release()

            def do_GET(self):
                self._handle()

            def do_POST(self):
                self._handle()

        self.server=ThreadingHTTPServer((self.host,self.port),Handler)
        self._thread=Thread(target=self.server.serve_forever,kwargs={"poll_interval":0.1},daemon=True)
        self._thread.start()
        if blocking:
            try:
                self._thread.join()
            finally:
                self.stop()
        return self

    def stop(self)->None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            if self._thread is not None:
                self._thread.join()
            self.server=None
            self._thread=None
