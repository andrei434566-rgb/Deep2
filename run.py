"""Entry point for Kern Analyzer.

The same portable executable serves both the graphical workplace and the
no-dialog processing routes used by the accompanying BAT shortcuts.  With no
arguments it opens the GUI; with a core-tape command it delegates to the
automatic pipeline and writes results beside the supplied data.
"""

import sys


def _enable_command_console() -> None:
    """Attach a console for BAT-driven routes in a windowed PyInstaller EXE.

    ``console=False`` is right for the normal graphical start.  It leaves
    ``sys.stdout`` and ``sys.stderr`` unavailable, however, which breaks
    argparse and status messages when the very same executable is started by
    a command shortcut.  Reuse the parent BAT console when possible; if the
    command was launched directly, create a small one for readable progress.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            if not kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
                kernel32.AllocConsole()
            console = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
            sys.stdout = console
            sys.stderr = console
            return
        except OSError:
            pass
    # Last-resort fallback keeps command routes functional even if a console
    # cannot be created (for example under a restricted launcher).
    sink = open("NUL" if sys.platform == "win32" else "/dev/null", "w", encoding="utf-8")
    sys.stdout = sink
    sys.stderr = sink


def main() -> int:
    if len(sys.argv) > 1:
        # Keep the command-line workflow inside the full GUI distribution so
        # that v1.5 never depends on a second, lightweight executable.
        _enable_command_console()
        from build_core_tape import main as core_tape_main

        return core_tape_main()
    # Keep Qt out of command-line modes.  This makes the automatic agents
    # usable even on a machine where the graphical Qt runtime cannot start.
    from PySide6.QtWidgets import QApplication
    from app.ui.windows.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Kern Analyzer")
    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
