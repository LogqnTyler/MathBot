import json
import os
from pathlib import Path

from google.cloud.sql.connector import Connector, IPTypes
import pg8000
import sqlalchemy
from dataclasses import dataclass, asdict

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    Column,
    ForeignKey,
    ARRAY,
    Table,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import declarative_base, Session

from embedding import embed_doc



@dataclass
class Chunk:
    kind: str
    mat_id: int  ## this is the reference to the lesson file
    name: str = ""
    learning_objective: int | None = None
    content_plain: str = ""
    content_latex: str = ""
    problem_context_plain: str = ""
    problem_context_latex: str = ""
    sub_prob_part: str = ""
    Q_plain: str = ""
    Q_latex: str = ""
    A_plain: str = ""
    A_latex: str = ""
    keywords: list[str] = None


def parse_contents(contents: dict, mat_id: int, lesson_name: str) -> list[Chunk]:
    keys = contents.keys()
    chunks = []
    for key in keys:
        if key == "definitions":
            for idx, definition in enumerate(contents[key]):
                defined_term = definition["term"]
                chunk = Chunk(
                    kind="definition",
                    name=defined_term + f"_{idx}",  ## ensures unique name for each definition
                    mat_id=mat_id,
                    content_plain=f"Def: {defined_term} \n"
                    + definition["definition_plain"],
                    content_latex=f"Def: {defined_term} \n"
                    + definition["definition_latex"],
                    keywords=[defined_term.lower()],
                )
                chunks.append(chunk)
        elif key == "other_material":
            for idx, material in enumerate(contents[key]):
                # kind = material["type"] we comment this out so that the chunk is listed as being of kind "other material" instead of "discussion" or "mini_lecture" etc
                kind = key
                name = material["type"] + f"_{idx}"  ## ensures unique name for each material
                chunk = Chunk(
                    kind=kind,
                    mat_id=mat_id,
                    name=name,
                    content_plain=material["content_plain"],
                    content_latex=material["content_latex"],
                )
                chunks.append(chunk)
        elif key == "problems":
            for idx, problem in enumerate(contents[key]):
                context_plain = problem["context_plain"]
                context_latex = problem["context_latex"]
                name = (
                    problem["name"] + f"_{idx}"
                )  ## ensures unique name for each problem
                keywords = problem["keywords"]
                for sub_prob in problem["subproblems"]:
                    chunk = Chunk(
                        kind="problem",
                        name=name
                        + "_"
                        + sub_prob["part"]
                        + lesson_name,  ## ensures unique name for each subproblem
                        mat_id=mat_id,
                        problem_context_plain=context_plain,
                        problem_context_latex=context_latex,
                        sub_prob_part=sub_prob["part"],
                        Q_plain=sub_prob["plain_text"]["question"],
                        Q_latex=sub_prob["latex"]["question"],
                        A_plain=sub_prob["plain_text"]["answer"],
                        A_latex=sub_prob["latex"]["answer"],
                        keywords=[kw.lower() for kw in keywords],
                    )
                    chunks.append(chunk)
    return chunks


def main():
    try:
        connector = Connector(refresh_strategy="LAZY")

        def connect_with_connector() -> sqlalchemy.engine.base.Engine:
            """
            Initializes a connection pool for a Cloud SQL instance of Postgres.

            Uses the Cloud SQL Python Connector package.
            """
            instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]
            db_user = os.environ["DB_USER"]
            db_pass = os.environ["DB_PASS"]
            db_name = os.environ["DB_NAME"]

            ip_type = (
                IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC
            )

            def getconn() -> pg8000.dbapi.Connection:
                conn: pg8000.dbapi.Connection = connector.connect(
                    instance_connection_name,
                    "pg8000",
                    user=db_user,
                    password=db_pass,
                    db=db_name,
                    ip_type=ip_type,
                )
                return conn

            return create_engine(
                "postgresql+pg8000://",
                creator=getconn,
            )

        engine = connect_with_connector()
        session = Session(engine)

        # Make the chunks table if it doesn't already exist
        Base = declarative_base()

        materials = Table(
            "materials",
            Base.metadata,
            Column("id", Integer, primary_key=True),
            Column("type", String),
            Column("name", String),
            Column("data", JSONB),
        )

        class Chunks(Base):
            __tablename__ = "chunks"
            __table_args__ = (
                UniqueConstraint("name", "mat_id", name="uq_chunks_name_mat_id"),
            )

            id = Column(Integer, primary_key=True)

            kind = Column(String)
            mat_id = Column(Integer, ForeignKey("materials.id"))
            name = Column(String)
            learning_objective = Column(Integer)
            content_plain = Column(String)
            content_latex = Column(String)
            problem_context_plain = Column(String)
            problem_context_latex = Column(String)
            sub_prob_part = Column(String)
            Q_plain = Column(String)
            Q_latex = Column(String)
            A_plain = Column(String)
            A_latex = Column(String)
            keywords = Column(ARRAY(String))
            embedding = Column(Vector(1024))

        with engine.begin() as conn:
            conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))

        Base.metadata.create_all(engine)

        with engine.begin() as conn:
            conn.execute(sqlalchemy.text("DROP TABLE IF EXISTS embeddings"))
            conn.execute(
                sqlalchemy.text(
                    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(1024)"
                )
            )

        # Load the JSON files into chunks

        lessons = session.execute(sqlalchemy.select(materials)).mappings().all()

        chunks = []
        for lesson in lessons:
            lesson_chunks = parse_contents(
                lesson["data"]["contents"],
                mat_id=lesson["id"],
                lesson_name=lesson["name"],
            )
            chunks.extend(lesson_chunks)

        print(f"Built {len(chunks)} chunks from the lessons")

        print("Inserting chunks into database...")
        rows = []
        for chunk in chunks:
            row = asdict(chunk)
            rows.append(row)

        stmt = insert(Chunks).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["name", "mat_id"])
        result = session.execute(stmt)

        print(f"Inserted {result.rowcount} new chunks into the database.")
        print(f"{len(chunks) - result.rowcount} chunks were already in the database.")

        session.commit()

        # compute and update embeddings
        saved_chunks = session.execute(sqlalchemy.select(Chunks)).scalars().all()

        print("Updating chunk embeddings in database...")

        updated_count = 0
        for chunk in saved_chunks:
            if chunk.embedding is not None:
                continue

            if chunk.kind == "definition":
                text = chunk.content_plain
            elif chunk.kind == "problem":
                context = chunk.problem_context_plain
                question = chunk.Q_plain
                text = context + "\n\n" + question
            else:
                text = chunk.content_plain

            chunk.embedding = embed_doc(text)
            updated_count += 1

        session.commit()

        print(f"Updated embeddings for {updated_count} chunks in the database.")
        print(f"{len(saved_chunks) - updated_count} chunks already had embeddings.")

    finally:
        session.close()
        engine.dispose()
        connector.close()


if __name__ == "__main__":
    main()
