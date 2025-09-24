from fastapi import FastAPI
import uvicorn
from app.api import router
from .db import create_tables


app = FastAPI()

app.include_router(router)
create_tables()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Document Q&A API. Use the /docs endpoint for API documentation."}


if __name__ == "__main__":    
    uvicorn.run(app, host="0.0.0.0", port=8000)