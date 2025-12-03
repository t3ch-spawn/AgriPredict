from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import all_routes

app = FastAPI()

origins = [
    "*"
]

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins= origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(all_routes.router)

@app.get("/")
def root():
    return {"message": "API is running!"}

