
from backend.indexing import build_index


queries = [
    "Facebook automation language:Python",
    "Facebook Selenium language:Python",
    "Facebook bot language:Python"
]

build_index(queries, pages=1)
