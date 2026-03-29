"""Shared pytest fixtures."""

from __future__ import annotations

import sqlite3

import pytest

from src.annotation.db import init_db


SAMPLE_VERDICT_TEXT = """[1.1] Yang mengadili perkara konstitusi pada tingkat pertama dan terakhir,
menjatuhkan putusan dalam perkara Pengujian Undang-Undang Nomor 1 Tahun 2021.

[2.1] Menimbang bahwa Pemohon telah mengajukan permohonan bertanggal 20 November 2020.

[2.2] Menimbang bahwa untuk membuktikan dalilnya, Pemohon mengajukan bukti-bukti.

[3.1] Menimbang bahwa berdasarkan Pasal 24C ayat (1) UUD 1945, Mahkamah berwenang.

[3.2] Menimbang bahwa Pemohon memiliki kedudukan hukum sebagai perorangan WNI.

[3.3] Menimbang bahwa pokok permohonan tidak beralasan menurut hukum.

[4.1] Mahkamah berwenang mengadili permohonan a quo.
[4.2] Pemohon memiliki kedudukan hukum.
[4.3] Pokok permohonan tidak beralasan menurut hukum.

5. AMAR PUTUSAN
Mengadili:
Menolak permohonan Pemohon untuk seluruhnya.

Demikian diputus pada hari Selasa, tanggal dua puluh sembilan, bulan Juni,
tahun dua ribu dua puluh satu.
KETUA,
Anwar Usman
"""

SAMPLE_VERDICT_ID = "test_verdict_001"


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_VERDICT_TEXT


@pytest.fixture
def sample_verdict_id() -> str:
    return SAMPLE_VERDICT_ID


@pytest.fixture
def sample_chunks():
    from src.indexing.chunkers.fixed_size import Chunk
    return [
        Chunk(text="Mahkamah berwenang mengadili perkara ini.", verdict_id="v1", chunk_idx=0, chunk_size=512, n_chars=42, section="pertimbangan"),
        Chunk(text="Pemohon memiliki kedudukan hukum yang sah.", verdict_id="v1", chunk_idx=1, chunk_size=512, n_chars=43, section="pertimbangan"),
        Chunk(text="Menolak permohonan Pemohon untuk seluruhnya.", verdict_id="v1", chunk_idx=2, chunk_size=512, n_chars=44, section="amar_putusan"),
    ]


@pytest.fixture
def sqlite_connection():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()
