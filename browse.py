"""
Terminal browser for the indexed course data.

    python browse.py

Controls:
  ↑ / ↓        navigate list
  PgUp / PgDn  scroll by page
  Enter        view course details
  /            focus filter box (type to narrow results)
  Esc          clear filter / exit detail view
  q            quit
"""

import curses
import sys
import textwrap

import chromadb

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "curriculum_itc"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_courses() -> list[dict]:
    """Load all courses from ChromaDB, sorted by code."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(
            "ERROR: Index not found. Run `python fetch_and_index.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    total = col.count()
    if total == 0:
        print("ERROR: Index is empty. Run `python fetch_and_index.py`.", file=sys.stderr)
        sys.exit(1)

    # Fetch everything (get() without IDs returns all)
    result = col.get(include=["documents", "metadatas"])
    courses = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        courses.append(
            {
                "code": meta.get("code", ""),
                "name": meta.get("name", ""),
                "credits": _credit_str(meta),
                "doc": doc,
            }
        )

    courses.sort(key=lambda c: c["code"])
    return courses


def _credit_str(meta: dict) -> str:
    cmin = meta.get("credits_min", 0)
    cmax = meta.get("credits_max", 0)
    if not cmin and not cmax:
        return "? ECTS"
    if cmin == cmax:
        return f"{cmin:.0f} ECTS"
    return f"{cmin:.0f}–{cmax:.0f} ECTS"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _attr(stdscr, name: str):
    mapping = {
        "header":    curses.color_pair(1) | curses.A_BOLD,
        "selected":  curses.color_pair(2) | curses.A_BOLD,
        "footer":    curses.color_pair(3),
        "highlight": curses.color_pair(4) | curses.A_BOLD,
        "normal":    curses.A_NORMAL,
        "dim":       curses.A_DIM,
    }
    return mapping.get(name, curses.A_NORMAL)


def _init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE,  curses.COLOR_BLUE)   # header
    curses.init_pair(2, curses.COLOR_BLACK,  curses.COLOR_CYAN)   # selected row
    curses.init_pair(3, curses.COLOR_WHITE,  curses.COLOR_BLUE)   # footer
    curses.init_pair(4, curses.COLOR_CYAN,   -1)                  # highlight text


def _draw_bar(win, y: int, text: str, attr):
    h, w = win.getmaxyx()
    win.attron(attr)
    win.addstr(y, 0, text.ljust(w)[:w])
    win.attroff(attr)


def _draw_text_block(win, start_y: int, start_x: int, width: int, text: str, scroll: int) -> int:
    """Wrap and draw text, respecting scroll offset. Returns number of lines drawn."""
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            lines.extend(textwrap.wrap(paragraph, width) or [""])
        else:
            lines.append("")

    h, w = win.getmaxyx()
    visible = lines[scroll:]
    drawn = 0
    for i, line in enumerate(visible):
        y = start_y + i
        if y >= h - 1:
            break
        try:
            win.addstr(y, start_x, line[:width])
        except curses.error:
            pass
        drawn += 1
    return len(lines)  # total line count (for scroll limit)


# ---------------------------------------------------------------------------
# List view
# ---------------------------------------------------------------------------

def list_view(stdscr, courses: list[dict]) -> None:
    curses.curs_set(0)
    cursor = 0       # index in filtered list
    scroll = 0       # top row of visible window
    filter_str = ""
    filter_mode = False

    while True:
        filtered = [
            c for c in courses
            if filter_str.lower() in c["code"].lower()
            or filter_str.lower() in c["name"].lower()
        ] if filter_str else courses

        h, w = stdscr.getmaxyx()
        list_h = h - 3  # rows for the course list (header + footer)

        cursor = max(0, min(cursor, len(filtered) - 1))
        if filtered and cursor < scroll:
            scroll = cursor
        if filtered and cursor >= scroll + list_h:
            scroll = cursor - list_h + 1

        stdscr.erase()

        # Header
        title = f" CurriculumMatcher Browser  [{len(filtered)}/{len(courses)} courses]"
        _draw_bar(stdscr, 0, title, _attr(stdscr, "header"))

        # Filter bar (row 1)
        filter_label = " Filter: "
        stdscr.addstr(1, 0, filter_label, _attr(stdscr, "dim"))
        fbox_w = w - len(filter_label) - 1
        fbox_text = (filter_str + ("_" if filter_mode else ""))[:fbox_w]
        stdscr.addstr(1, len(filter_label), fbox_text.ljust(fbox_w)[:fbox_w])

        # Course list (rows 2 … h-2)
        for i in range(list_h):
            idx = scroll + i
            y = i + 2
            if idx >= len(filtered):
                break
            c = filtered[idx]
            code_field = c["code"].ljust(16)[:16]
            credits_field = c["credits"].rjust(10)
            name_w = w - len(code_field) - len(credits_field) - 3
            name_field = c["name"][:name_w].ljust(name_w)
            row = f" {code_field} {name_field} {credits_field} "
            if idx == cursor:
                stdscr.attron(_attr(stdscr, "selected"))
                try:
                    stdscr.addstr(y, 0, row[:w])
                except curses.error:
                    pass
                stdscr.attroff(_attr(stdscr, "selected"))
            else:
                try:
                    stdscr.addstr(y, 0, row[:w])
                except curses.error:
                    pass

        # Footer
        footer = " ↑↓ navigate  PgUp/PgDn page  Enter: view  /: filter  Esc: clear  q: quit "
        _draw_bar(stdscr, h - 1, footer, _attr(stdscr, "footer"))

        stdscr.refresh()

        key = stdscr.getch()

        if filter_mode:
            if key == 27:  # Esc
                filter_mode = False
                filter_str = ""
                cursor = 0
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                filter_str = filter_str[:-1]
                cursor = 0
            elif key == ord("\n"):
                filter_mode = False
                curses.curs_set(0)
            elif 32 <= key <= 126:
                filter_str += chr(key)
                cursor = 0
            continue

        if key == ord("q"):
            break
        elif key == ord("/"):
            filter_mode = True
            curses.curs_set(1)
        elif key == 27:  # Esc — clear filter
            filter_str = ""
            cursor = 0
        elif key == curses.KEY_UP:
            cursor = max(0, cursor - 1)
        elif key == curses.KEY_DOWN:
            cursor = min(len(filtered) - 1, cursor + 1)
        elif key == curses.KEY_PPAGE:
            cursor = max(0, cursor - list_h)
        elif key == curses.KEY_NPAGE:
            cursor = min(len(filtered) - 1, cursor + list_h)
        elif key == curses.KEY_HOME:
            cursor = 0
        elif key == curses.KEY_END:
            cursor = len(filtered) - 1
        elif key == ord("\n") and filtered:
            detail_view(stdscr, filtered[cursor])


# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------

def detail_view(stdscr, course: dict) -> None:
    curses.curs_set(0)
    scroll = 0

    while True:
        h, w = stdscr.getmaxyx()
        stdscr.erase()

        title = f" {course['code']} — {course['name']} "
        _draw_bar(stdscr, 0, title, _attr(stdscr, "header"))

        text_w = w - 2
        total_lines = _draw_text_block(stdscr, 2, 1, text_w, course["doc"], scroll)

        footer = " ↑↓ scroll  PgUp/PgDn page  Esc/q: back to list "
        _draw_bar(stdscr, h - 1, footer, _attr(stdscr, "footer"))

        stdscr.refresh()

        key = stdscr.getch()
        visible_rows = h - 3
        max_scroll = max(0, total_lines - visible_rows)

        if key in (ord("q"), 27, ord("b")):
            break
        elif key == curses.KEY_UP:
            scroll = max(0, scroll - 1)
        elif key == curses.KEY_DOWN:
            scroll = min(max_scroll, scroll + 1)
        elif key == curses.KEY_PPAGE:
            scroll = max(0, scroll - visible_rows)
        elif key == curses.KEY_NPAGE:
            scroll = min(max_scroll, scroll + visible_rows)
        elif key == curses.KEY_HOME:
            scroll = 0
        elif key == curses.KEY_END:
            scroll = max_scroll


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(stdscr):
    _init_colors()
    stdscr.keypad(True)

    stdscr.addstr(0, 0, "Loading courses from index...")
    stdscr.refresh()

    courses = load_courses()
    list_view(stdscr, courses)


if __name__ == "__main__":
    curses.wrapper(main)
