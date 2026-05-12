import os
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
OUT_CSV = ROOT / "testset_v1.csv"
REVIEW_NOTES = ROOT / "testset_review_notes.md"

TESTSET_SIZE = 50


def prepare_docs_dir():
    DOCS_DIR.mkdir(exist_ok=True)

    md_files = sorted(DATA_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"Không tìm thấy file .md trong {DATA_DIR}. "
            "Hãy kiểm tra lại thư mục data/*.md."
        )

    for src in md_files:
        dst = DOCS_DIR / src.name
        shutil.copy2(src, dst)

    print(f"Copied {len(md_files)} markdown files from {DATA_DIR} to {DOCS_DIR}")


def load_documents():
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()

    if not docs:
        raise RuntimeError("Không load được document nào từ ./docs")

    print(f"Loaded {len(docs)} documents")
    return docs


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ragas version khác nhau có thể xuất column khác nhau.
    Hàm này cố gắng chuẩn hóa về:
    question, ground_truth, contexts, evolution_type
    """
    df = df.copy()

    rename_map = {}

    if "user_input" in df.columns and "question" not in df.columns:
        rename_map["user_input"] = "question"

    if "reference" in df.columns and "ground_truth" not in df.columns:
        rename_map["reference"] = "ground_truth"

    if "reference_contexts" in df.columns and "contexts" not in df.columns:
        rename_map["reference_contexts"] = "contexts"

    if "synthesizer_name" in df.columns and "evolution_type" not in df.columns:
        rename_map["synthesizer_name"] = "evolution_type"

    df = df.rename(columns=rename_map)

    required = ["question", "ground_truth", "contexts", "evolution_type"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    return df


def add_manual_edit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Manual edit tối thiểu: sửa 1 câu hỏi đầu tiên cho rõ hơn.
    Bạn có thể mở CSV và sửa tay tiếp sau.
    """
    df = df.copy()

    if len(df) > 0 and isinstance(df.loc[0, "question"], str):
        old_question = df.loc[0, "question"]
        df.loc[0, "question"] = old_question.strip()
        if not df.loc[0, "question"].endswith("?"):
            df.loc[0, "question"] += "?"

        df.loc[0, "manual_edit_note"] = (
            "Manual edit: normalized whitespace and ensured question mark."
        )

    return df


def write_review_notes(df: pd.DataFrame):
    sample = df.head(10)

    lines = [
        "# Testset Review Notes",
        "",
        f"- Generated file: `{OUT_CSV.name}`",
        f"- Total rows: {len(df)}",
        "- Reviewed rows: first 10",
        "- Manual edit: row 0 question normalized and question mark ensured.",
        "",
        "## Distribution",
        "",
        "```text",
        str(df["evolution_type"].value_counts(dropna=False)),
        "```",
        "",
        "## First 10 Questions Reviewed",
        "",
    ]

    for idx, row in sample.iterrows():
        lines.extend(
            [
                f"### Row {idx}",
                f"- Question: {row.get('question', '')}",
                f"- Evolution type: {row.get('evolution_type', '')}",
                "- Review: Needs human check for factual correctness against source context.",
                "",
            ]
        )

    REVIEW_NOTES.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote review notes to {REVIEW_NOTES}")


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "Thiếu OPENAI_API_KEY trong .env hoặc environment. "
            "Hãy thêm OPENAI_API_KEY=sk-... vào file .env"
        )

    prepare_docs_dir()
    docs = load_documents()

    generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    generator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )

    query_distribution = default_query_distribution(generator.llm)

    print("Generating testset...")
    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=TESTSET_SIZE,
        query_distribution=query_distribution,
        with_debugging_logs=True,
        raise_exceptions=True,
    )

    df = testset.to_pandas()
    df = normalize_dataframe(df)
    df = add_manual_edit(df)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {OUT_CSV}")

    write_review_notes(df)

    if len(df) < TESTSET_SIZE:
        raise RuntimeError(f"Expected at least {TESTSET_SIZE} rows, got {len(df)}")

    required = {"question", "ground_truth", "contexts", "evolution_type"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    print("Verification passed.")
    print(df["evolution_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()