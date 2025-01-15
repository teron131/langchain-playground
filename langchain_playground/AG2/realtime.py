import os
from logging import getLogger
from pathlib import Path
from typing import Annotated

import autogen
import uvicorn
from autogen.agentchat.realtime_agent import RealtimeAgent
from config import llm_config
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

realtime_config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4o-mini-realtime-preview"],
    },
)
print(realtime_config_list)

realtime_llm_config = {
    "timeout": 600,
    "config_list": realtime_config_list,
    "temperature": 0.8,
}
