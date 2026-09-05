"""The engine may touch its own data root and nothing else on the machine."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabint.analysis.db import persistence
from tabint.shared import server as shared_server


@pytest.fixture()
def base(tmp_path, monkeypatch):
    monkeypatch.setenv("TABULAR_BASE", str(tmp_path))
    monkeypatch.setattr(shared_server, "_BASE", str(tmp_path))
    csv = tmp_path / "orders.csv"
    csv.write_text("region,amount\nnorth,10\nsouth,25\n")
    return tmp_path, csv


@pytest.fixture()
def session(base):
    root, csv = base
    return persistence.create_session([str(csv)], base=str(root))


def _tool(name):
    import tabint.analysis.tools as tools
    return tools.mcp._tool_manager._tools[name].fn


def test_ingest_and_query_inside_the_root_still_work(session):
    assert session.tables == ["orders"]
    rows = session.run_sql("SELECT sum(amount) AS t FROM orders").to_dict("records")
    assert rows[0]["t"] == 35


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM read_csv_auto('/etc/hosts')",
        "SELECT * FROM read_csv_auto('/etc/../etc/hosts')",
        "SELECT * FROM glob('/etc/*')",
    ],
)
def test_sql_cannot_reach_outside_the_root(session, query):
    with pytest.raises(Exception, match="(?i)permission|cannot access"):
        session.run_sql(query)


def test_the_allow_list_cannot_be_widened(session):
    for statement in (
        "SET allowed_directories=['/']",
        "PRAGMA allowed_directories=['/']",
        "RESET allowed_directories",
        "SET enable_external_access=true",
    ):
        with pytest.raises(Exception):
            session.workspace._ibis.con.execute(statement)
    with pytest.raises(Exception, match="(?i)permission|cannot access"):
        session.run_sql("SELECT * FROM read_csv_auto('/etc/hosts')")


def test_derived_tables_cannot_reach_outside_either(session):
    with pytest.raises(Exception, match="(?i)permission|cannot access"):
        session.workspace.create_table("x", select_sql="SELECT * FROM read_csv_auto('/etc/hosts')")
    with pytest.raises(Exception, match="(?i)permission|cannot access"):
        session.workspace.insert_into("orders", "SELECT * FROM read_csv_auto('/etc/hosts')")


def test_create_session_ingests_a_path_outside_the_root(base, tmp_path_factory):
    """Ingest is a copy into DuckDB, so the file may live anywhere readable."""
    root, _ = base
    stray = tmp_path_factory.mktemp("elsewhere") / "sales.csv"
    stray.write_text("region,amount\nnorth,10\nsouth,25\n")
    result = _tool("create_session")(paths=[str(stray)])
    assert result["tables"] == ["sales"]  # table named for the file, not the staged copy
    assert not list((root / ".staging").glob("*"))  # staged copy cleaned up


def test_add_table_ingests_a_path_outside_the_root(base, session, tmp_path_factory):
    from tabint.shared.server import register_session
    register_session(session)
    root, _ = base
    stray = tmp_path_factory.mktemp("elsewhere2") / "returns.csv"
    stray.write_text("k,v\na,1\n")
    result = _tool("add_table")(session_key=session.id, path=str(stray))
    assert result["added_table"] == "returns"
    assert not list((root / ".staging").glob("*"))


def test_ingesting_a_staged_file_does_not_unseal_sql(base, tmp_path_factory):
    """Reading an outside CSV must not widen what agent-authored SQL can reach."""
    stray = tmp_path_factory.mktemp("elsewhere3") / "sales.csv"
    stray.write_text("region,amount\nnorth,10\n")
    result = _tool("create_session")(paths=[str(stray)])
    from tabint.shared.server import get_session
    live = get_session(result["session_key"])
    with pytest.raises(Exception, match="(?i)permission|cannot access"):
        live.run_sql("SELECT * FROM read_csv_auto('/etc/hosts')")
    with pytest.raises(Exception, match="(?i)permission|cannot access"):
        live.run_sql(f"SELECT * FROM read_csv_auto('{stray}')")


def test_a_missing_path_is_a_readable_error(base):
    result = _tool("create_session")(paths=["/no/such/file.csv"])
    assert result["error"] == "file_not_found"


def test_reopening_in_the_same_process_stays_confined(base, session):
    root, _ = base
    again = persistence.open_session(session.id, base=str(root))
    assert again.tables == ["orders"]
    with pytest.raises(Exception, match="(?i)permission|cannot access"):
        again.run_sql("SELECT * FROM read_csv_auto('/etc/hosts')")
