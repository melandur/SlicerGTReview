#!/usr/bin/env python3
"""Embed the panel screenshots into panel-guide.html.

The panel guide is a single self-contained file -- it has to work when it is
copied on its own to a share -- so its screenshots are data URIs rather than
links.  Each one names its source in ``data-src``: a PNG under ``Docs/img``,
the same files the tutorial links to.  After retaking the screenshots, run

    python Docs/embed_screenshots.py

and every ``<img data-src="img/...">`` in the guide gets that file's current
content.  Standard library only.
"""

import base64
import os
import re
import sys

DOCS = os.path.dirname(os.path.abspath(__file__))
GUIDE = os.path.join(DOCS, "panel-guide.html")
_IMG = re.compile(r'<img\b[^>]*\bdata-src="([^"]+)"[^>]*>')


def embed(match):
    tag, source = match.group(0), match.group(1)
    with open(os.path.join(DOCS, source), "rb") as handle:
        uri = "data:image/png;base64," + base64.b64encode(handle.read()).decode("ascii")
    if re.search(r'\bsrc="', tag.replace("data-src=", "")):
        return re.sub(r'(?<!data-)\bsrc="[^"]*"', 'src="{}"'.format(uri), tag)
    return tag.replace("<img", '<img src="{}"'.format(uri), 1)


def main():
    with open(GUIDE, encoding="utf-8") as handle:
        html = handle.read()
    updated, count = _IMG.subn(embed, html)
    with open(GUIDE, "w", encoding="utf-8") as handle:
        handle.write(updated)
    print("{} screenshot(s) embedded into {}".format(count, os.path.basename(GUIDE)))
    return 0 if count else 1


if __name__ == "__main__":
    sys.exit(main())
