import os
from dotenv import load_dotenv
import json, glob, os, psycopg2

load_dotenv()

pg_url = os.getenv("GC_POSTGRESS_URL")
db_password = os.getenv("GC_POSTGRESS_PASSWORD")

conn = psycopg2.connect(f"postgresql://user:{db_password}@{pg_url}/dbname")
