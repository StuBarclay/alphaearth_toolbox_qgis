#!/usr/bin/env python3
"""Build the combined Word document from the Markdown documentation set.

The ``docs/`` folder holds the canonical documentation as Markdown, one file per
audience/topic. This script concatenates them, in reading order, into a single
Word document (``docs/AlphaEarth_Toolbox_Documentation.docx``) with a real,
updatable Word table of contents, so there is one downloadable/printable file
that always matches the Markdown sources.

Pipeline
--------
1. Concatenate the ordered Markdown files, inserting a page break before each
   document after the first so every section starts on a fresh page.
2. Run ``pandoc`` to produce the ``.docx``. ``--toc`` inserts a genuine Word TOC
   field (not a static list); the existing document, if present, is reused as
   the ``--reference-doc`` so paragraph/heading/TOC styles stay identical
   release to release.
3. Open the result once in headless LibreOffice via UNO and call ``.update()``
   on every document index, so the TOC's page numbers are correct on first open
   rather than showing zeros until the reader refreshes the field.
4. Copy the bytes over the destination with ``shutil.copyfile`` (truncate in
   place), which a cloud-synced (OneDrive) folder allows even when an unlink /
   rename would be blocked.

Run it from anywhere::

    python scripts/build_docs_docx.py

Requires ``pandoc`` and LibreOffice (``soffice``) with the Python ``uno``
bridge, both of which are present in the doc-tooling environment.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

#: Markdown files, in the order they appear in the combined document.
ORDER = (
    "README.md",
    "ARCHITECTURE.md",
    "METHODOLOGY.md",
    "USER_GUIDE.md",
    "LIMITATIONS.md",
    "ROADMAP.md",
)

#: Output document (also reused as the pandoc style reference when it exists).
OUTPUT_NAME = "AlphaEarth_Toolbox_Documentation.docx"

#: A raw-OpenXML hard page break (pandoc passes ``{=openxml}`` blocks through
#: untouched), used to start each document on a new page.
_PAGE_BREAK = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'

#: Headless LibreOffice UNO socket.
_UNO_PORT = 2002
_UNO_URL = f"uno:socket,host=localhost,port={_UNO_PORT};urp;StarOffice.ComponentContext"


def _docs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "docs"


def build_combined_markdown(docs_dir: Path, dest: Path) -> Path:
    """Concatenate the ordered Markdown files (page-broken) into ``dest``."""
    chunks: list[str] = []
    for index, name in enumerate(ORDER):
        source = docs_dir / name
        if not source.is_file():
            raise SystemExit(f"Missing documentation source: {source}")
        if index > 0:
            chunks.append(_PAGE_BREAK)
        chunks.append(source.read_text(encoding="utf-8").rstrip())
        chunks.append("")
    dest.write_text("\n".join(chunks), encoding="utf-8")
    return dest


def run_pandoc(combined_md: Path, reference: Path | None, out_docx: Path) -> None:
    """Render ``combined_md`` to ``out_docx`` with an updatable Word TOC."""
    cmd = [
        "pandoc",
        str(combined_md),
        "--from=markdown",
        "--to=docx",
        "--toc",
        "--toc-depth=2",
        "--metadata",
        "toc-title=Table of Contents",
        "-o",
        str(out_docx),
    ]
    if reference is not None and reference.is_file():
        cmd += ["--reference-doc", str(reference)]
    subprocess.run(cmd, check=True)


def update_indexes(docx_path: Path) -> None:
    """Open ``docx_path`` in headless LibreOffice and refresh every TOC field."""
    import uno
    from com.sun.star.beans import PropertyValue

    subprocess.run(["pkill", "-x", "soffice.bin"], check=False)
    time.sleep(1)

    profile = Path(tempfile.mkdtemp(prefix="lo_profile_")).as_uri()
    soffice = subprocess.Popen(
        [
            "soffice",
            "--headless",
            "--invisible",
            "--nologo",
            "--norestore",
            "--nofirststartwizard",
            f"-env:UserInstallation={profile}",
            f"--accept=socket,host=localhost,port={_UNO_PORT};urp;",
        ]
    )
    try:
        local_ctx = uno.getComponentContext()
        resolver = local_ctx.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", local_ctx
        )
        ctx = None
        for _ in range(40):
            try:
                ctx = resolver.resolve(_UNO_URL)
                break
            except Exception:
                time.sleep(0.5)
        if ctx is None:
            raise SystemExit("Could not connect to headless LibreOffice.")

        smgr = ctx.ServiceManager
        desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        hidden = PropertyValue()
        hidden.Name = "Hidden"
        hidden.Value = True
        doc = desktop.loadComponentFromURL(docx_path.as_uri(), "_blank", 0, (hidden,))
        try:
            indexes = doc.getDocumentIndexes()
            for i in range(indexes.getCount()):
                indexes.getByIndex(i).update()
            doc.store()
        finally:
            doc.close(False)
    finally:
        soffice.terminate()
        try:
            soffice.wait(timeout=20)
        except subprocess.TimeoutExpired:
            soffice.kill()
        subprocess.run(["pkill", "-x", "soffice.bin"], check=False)


def build(docs_dir: Path | None = None) -> Path:
    """Build the combined docx and return its path."""
    docs_dir = docs_dir or _docs_dir()
    output = docs_dir / OUTPUT_NAME

    with tempfile.TemporaryDirectory(prefix="ae_docx_") as tmp_name:
        tmp = Path(tmp_name)
        combined = build_combined_markdown(docs_dir, tmp / "combined.md")

        reference: Path | None = None
        if output.is_file():
            reference = tmp / "reference.docx"
            shutil.copyfile(output, reference)

        scratch = tmp / "out.docx"
        run_pandoc(combined, reference, scratch)
        update_indexes(scratch)

        # Truncate-in-place copy: OneDrive allows this even when it would block
        # an unlink/replace of a file that still has a stale lock beside it.
        shutil.copyfile(scratch, output)

    print(f"Built {output.name} ({output.stat().st_size} bytes)")
    return output


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
