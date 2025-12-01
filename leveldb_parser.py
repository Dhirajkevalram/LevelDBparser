"""
LevelDB Parser

Features:
- Desktop GUI using PySide6 (works with PyQt5 with minimal changes)
- Open a LevelDB directory or upload a zipped LevelDB folder
- Background scanning using QThread to keep UI responsive
- Key list with paging, search, and prefix filter
- Snappy decompression attempt and automatic detection (text / JSON / hex)
- View selected value in a viewer with pretty JSON support
- Export matched rows to CSV

Requirements:
    pip install plyvel python-snappy pandas PySide6
    
    For IndexedDB databases (custom comparators):
    - Option 1: pip install leveldb (alternative Python package)
    - Option 2: Install leveldb-tools from https://github.com/google/leveldb

Notes:
- plyvel must be installed and LevelDB system libs present on your platform.
- For IndexedDB databases, the tool will automatically try alternative methods.
- If using PyQt5 instead of PySide6, replace imports accordingly.

Run:
    python leveldb_browser_pyside6.py

"""

import base64
import binascii
import sys
import os
import tempfile
import zipfile
import csv
import json
import subprocess
import struct
import glob
from functools import partial

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QTableWidget,
    QTableWidgetItem, QSplitter, QTextEdit, QVBoxLayout,
    QWidget, QToolBar, QHBoxLayout, QLabel, QLineEdit, QSpinBox, QPushButton,
    QTableView,QHeaderView, QAbstractItemView, QComboBox, QProgressBar, QMessageBox
)


from PySide6.QtGui import QAction



from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSize, QThread, Signal

# Optional imports that require native deps
try:
    import plyvel
except Exception as e:
    plyvel = None

try:
    import snappy
except Exception:
    snappy = None

# ---------------------------
# Utilities for decoding
# ---------------------------

def try_snappy(b):
    if b is None:
        return b
    if snappy is None:
        return b
    try:
        return snappy.uncompress(b)
    except Exception:
        return b



def try_json_pretty(text):
    """Return pretty JSON string if text parses as JSON else None."""
    try:
        obj = json.loads(text)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return None

def try_base64_then_decode(data_bytes):
    """
    If data_bytes looks like base64-encoded ASCII, try to base64-decode it and return bytes.
    Returns decoded bytes or None.
    """
    # Quick heuristic: base64 uses only ASCII letters, digits, +/= and usually length multiple of 4
    try:
        s = data_bytes.decode('ascii', errors='ignore').strip()
    except Exception:
        return None
    if not s:
        return None
    # Clean whitespace
    s_clean = ''.join(s.split())
    # Must be divisible by 4 (padding)
    if len(s_clean) % 4 != 0:
        # allow non-multiple-of-4 but try anyway
        pass
    # check allowed chars (basic)
    if any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in s_clean[:min(len(s_clean), 256)]):
        return None
    try:
        decoded = base64.b64decode(s_clean, validate=False)
        # consider only if decoded something non-trivial
        if decoded:
            return decoded
    except Exception:
        return None
    return None

def decode_hex_string_if_needed(text):
    """
    If text looks like a hex string (especially UTF-16LE), try to decode it.
    Returns decoded text or original text if decoding fails.
    """
    if not isinstance(text, str) or not text:
        return text
    
    # Remove whitespace and check if it's all hex characters
    text_clean = ''.join(text.split())
    
    # Check if it looks like hex (only 0-9, a-f, A-F)
    if len(text_clean) >= 4 and all(c in '0123456789abcdefABCDEF' for c in text_clean):
        # Check if length is even (hex pairs)
        if len(text_clean) % 2 == 0:
            try:
                # Try to decode as hex bytes
                hex_bytes = bytes.fromhex(text_clean)
                
                # Try UTF-16LE first (common in Windows/WhatsApp data)
                try:
                    decoded = hex_bytes.decode('utf-16le')
                    # Remove null terminators
                    decoded = decoded.rstrip('\x00')
                    if decoded and len(decoded) > 0:
                        return decoded
                except Exception:
                    pass
                
                # Try UTF-16BE
                try:
                    decoded = hex_bytes.decode('utf-16be')
                    decoded = decoded.rstrip('\x00')
                    if decoded and len(decoded) > 0:
                        return decoded
                except Exception:
                    pass
                
                # Try UTF-8
                try:
                    decoded = hex_bytes.decode('utf-8')
                    decoded = decoded.rstrip('\x00')
                    if decoded and len(decoded) > 0:
                        return decoded
                except Exception:
                    pass
                
                # Try ASCII
                try:
                    decoded = hex_bytes.decode('ascii', errors='replace')
                    decoded = decoded.rstrip('\x00')
                    # Only return if it's mostly printable
                    if decoded and sum(1 for c in decoded if 32 <= ord(c) <= 126) / max(1, len(decoded)) > 0.7:
                        return decoded
                except Exception:
                    pass
                    
            except Exception:
                pass
    
    return text

def decode_and_detect(b):
    """
    Enhanced decoding that aggressively tries to decode all values:
    - Tries Snappy decompression
    - Tries multiple encodings (utf-8, utf-16le, utf-16be)
    - Handles null bytes and special prefixes
    - Tries base64 decoding
    - Removes null bytes and tries again
    - Extracts printable text even from binary data
    - Tries JSON parsing with null byte removal
    - Returns dict: {'type': 'json'|'text'|'bytes'|'kv'|'missing', 'pretty': str, 'hex': str}
    """
    if b is None:
        return {"type": "missing", "pretty": "", "hex": ""}

    raw_hex = b.hex()

    # 1) Try Snappy first (if installed)
    data = try_snappy(b)

    # 2) Try common text encodings
    encodings = ["utf-8", "utf-16le", "utf-16be"]
    decoded_text = None
    used_encoding = None

    for enc in encodings:
        try:
            decoded_text = data.decode(enc)
            used_encoding = enc
            break
        except Exception:
            decoded_text = None

    # 3) If still nothing but data length is even, try utf-16le fallback (common case)
    if decoded_text is None:
        try:
            if len(data) % 2 == 0 and len(data) > 2:
                # Try utf-16le heuristically
                decoded_text = data.decode("utf-16le")
                used_encoding = "utf-16le"
        except Exception:
            decoded_text = None

    # 4) Try removing null bytes from start/end and decode again
    if decoded_text is None:
        try:
            # Remove leading null bytes (common in some formats)
            stripped_start = data.lstrip(b'\x00')
            if stripped_start != data and len(stripped_start) > 0:
                for enc in encodings:
                    try:
                        decoded_text = stripped_start.decode(enc)
                        used_encoding = f"{enc}(strip-leading-null)"
                        break
                    except Exception:
                        pass
        except Exception:
            pass

    # 5) Try removing null bytes from both ends
    if decoded_text is None:
        try:
            stripped = data.strip(b'\x00')
            if stripped != data and len(stripped) > 0:
                for enc in encodings:
                    try:
                        decoded_text = stripped.decode(enc)
                        used_encoding = f"{enc}(strip-null)"
                        break
                    except Exception:
                        pass
        except Exception:
            pass

    # 6) Try removing ALL null bytes (aggressive approach for embedded nulls)
    if decoded_text is None:
        try:
            no_nulls = data.replace(b'\x00', b'')
            if len(no_nulls) > 0 and len(no_nulls) < len(data):
                for enc in encodings:
                    try:
                        decoded_text = no_nulls.decode(enc)
                        used_encoding = f"{enc}(remove-all-nulls)"
                        break
                    except Exception:
                        pass
        except Exception:
            pass

    # 7) Try base64 heuristics (many values embed base64)
    if decoded_text is None:
        decoded_b64 = try_base64_then_decode(data)
        if decoded_b64:
            # try decode the base64 result as text using the encodings above
            for enc in encodings:
                try:
                    decoded_text = decoded_b64.decode(enc)
                    used_encoding = "base64+" + enc
                    break
                except Exception:
                    decoded_text = None
        # also try base64->utf-16 fallback
        if decoded_text is None and decoded_b64 and len(decoded_b64) % 2 == 0:
            try:
                decoded_text = decoded_b64.decode("utf-16le")
                used_encoding = "base64+utf-16le"
            except Exception:
                decoded_text = None

    # 8) If still not decoded, try to extract printable text from binary data
    if decoded_text is None:
        # Extract all printable ASCII characters (including those after nulls)
        printable_parts = []
        current_text = b''
        for byte in data:
            if 32 <= byte <= 126:  # Printable ASCII
                current_text += bytes([byte])
            else:
                if len(current_text) > 0:
                    try:
                        printable_parts.append(current_text.decode('ascii'))
                    except Exception:
                        pass
                    current_text = b''
        if len(current_text) > 0:
            try:
                printable_parts.append(current_text.decode('ascii'))
            except Exception:
                pass
        
        if printable_parts:
            decoded_text = ''.join(printable_parts)
            used_encoding = "extracted-printable"
            # If we got substantial text, use it
            if len(decoded_text) >= 4:
                pass  # Will continue to JSON/text processing below
            else:
                decoded_text = None

    # If no text decoding succeeded, return bytes representation (hex)
    if decoded_text is None:
        # Last resort: show hex representation
        return {"type": "bytes", "pretty": raw_hex, "hex": raw_hex}

    # 9) Clean up decoded text - remove null bytes
    if '\x00' in decoded_text:
        decoded_text = decoded_text.replace('\x00', '')
        if not decoded_text:  # If we removed everything, fall back
            return {"type": "bytes", "pretty": raw_hex, "hex": raw_hex}

    # 9.5) Check if decoded_text is actually a hex string that needs further decoding
    # This handles cases where the data was stored as a hex string representation
    hex_decoded = decode_hex_string_if_needed(decoded_text)
    if hex_decoded != decoded_text and hex_decoded:
        # If we successfully decoded a hex string, use that instead
        decoded_text = hex_decoded
        used_encoding = f"{used_encoding or 'unknown'}+hex-decode"

    # 10) Try to parse as JSON (even if it has some issues)
    # First try direct JSON parse
    pretty = try_json_pretty(decoded_text)
    if pretty is not None:
        return {"type": "json", "pretty": pretty, "hex": raw_hex}

    # Try JSON parse after removing leading/trailing non-JSON chars
    if decoded_text.strip():
        # Try to find JSON-like content (starts with { or [)
        json_start = -1
        for i, char in enumerate(decoded_text):
            if char in '{[':
                json_start = i
                break
        
        if json_start >= 0:
            # Try to find matching closing brace/bracket
            json_candidate = decoded_text[json_start:]
            # Try to extract valid JSON by finding the matching closing char
            try:
                if json_candidate[0] == '{':
                    # Find matching }
                    depth = 0
                    end_pos = -1
                    for i, char in enumerate(json_candidate):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                end_pos = i + 1
                                break
                    if end_pos > 0:
                        json_str = json_candidate[:end_pos]
                        try:
                            obj = json.loads(json_str)
                            return {"type": "json", "pretty": json.dumps(obj, indent=2, ensure_ascii=False), "hex": raw_hex}
                        except Exception:
                            pass
            except Exception:
                pass

    # 11) Try key:value lines -> convert to JSON
    try:
        lines = [ln.strip() for ln in decoded_text.splitlines() if ':' in ln and ln.strip()]
        if lines and len(lines) >= 1:
            kv = {}
            for ln in lines:
                parts = ln.split(":", 1)
                if len(parts) == 2:
                    k = parts[0].strip().strip('"\'')
                    v = parts[1].strip().strip('"\'')
                    if k:  # Only add if key is not empty
                        kv[k] = v
            if kv:
                return {"type": "kv", "pretty": json.dumps(kv, indent=2, ensure_ascii=False), "hex": raw_hex}
    except Exception:
        pass

    # 12) Default: treat as text (fully decoded)
    return {"type": "text", "pretty": decoded_text, "hex": raw_hex}
'''
def decode_and_detect(b):
    """Return dict with fields: type (json/text/bytes), pretty, hex"""
    if b is None:
        return {"type": "missing", "pretty": "", "hex": ""}

    raw_hex = b.hex()

    # try snappy
    tried = try_snappy(b)

    # try utf-8
    try:
        text = tried.decode('"utf-8"')
        # try json
        try:
            j = json.loads(text)
            pretty = json.dumps(j, indent=2, ensure_ascii=False)
            return {"type": "json", "pretty": pretty, "hex": raw_hex}
        except Exception:
            return {"type": "text", "pretty": text, "hex": raw_hex}
    except Exception:
        return {"type": "bytes", "pretty": raw_hex, "hex": raw_hex}
'''
# ---------------------------
# Utilities for opening DB with custom comparators
# ---------------------------

def read_leveldb_via_tools(db_path):
    """
    Try to read LevelDB using leveldb-tools command line tool.
    Returns list of (key, value) tuples or None if tool not available.
    """
    # Try to find leveldb-tools
    tools_to_try = ['leveldb-tools', 'ldb', 'leveldb_dump']
    
    for tool in tools_to_try:
        try:
            # Try to run the tool
            result = subprocess.run(
                [tool, 'dump', db_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                # Parse the output (format: key => value)
                entries = []
                for line in result.stdout.split('\n'):
                    if ' => ' in line:
                        parts = line.split(' => ', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            # Convert hex strings to bytes if needed
                            try:
                                if key.startswith('0x'):
                                    key_bytes = bytes.fromhex(key[2:])
                                else:
                                    key_bytes = key.encode('utf-8')
                                if value.startswith('0x'):
                                    value_bytes = bytes.fromhex(value[2:])
                                else:
                                    value_bytes = value.encode('utf-8')
                                entries.append((key_bytes, value_bytes))
                            except Exception:
                                entries.append((key.encode('utf-8'), value.encode('utf-8')))
                return entries
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            continue
    
    return None

def open_leveldb_with_fallback(db_path):
    """
    Try to open LevelDB with multiple approaches to handle custom comparators.
    Returns DB object or raises exception.
    """
    if plyvel is None:
        raise Exception('plyvel is not installed or importable')
    
    # Try 1: Default opening (standard comparator)
    try:
        return plyvel.DB(db_path, create_if_missing=False)
    except Exception as e1:
        error_str = str(e1)
        # Check if it's a comparator mismatch error
        if 'comparator' in error_str.lower() or 'cmp' in error_str.lower():
            # For IndexedDB databases, we'll use alternative method
            # Return a special marker that ScanWorker will handle
            raise Exception('COMPARATOR_MISMATCH:' + error_str)
        else:
            # Not a comparator error, re-raise original
            raise e1

# ---------------------------
# Worker thread for scanning DB
# ---------------------------

class ScanWorker(QThread):
    progress = Signal(int)
    row_found = Signal(bytes, bytes)
    finished_scan = Signal(int)  # total matched
    error = Signal(str)

    def __init__(self, db_path, prefix_bytes=None, search_bytes=None, limit=None):
        super().__init__()
        self.db_path = db_path
        self.prefix_bytes = prefix_bytes
        self.search_bytes = search_bytes
        self.limit = limit
        self._stop = False
        self.total = 0
        self.db = None
        self.use_alternative_method = False

    def read_via_python_leveldb(self):
        """
        Try using the 'leveldb' Python package (different from plyvel).
        This package might handle custom comparators differently.
        """
        try:
            import leveldb
            db = leveldb.LevelDB(self.db_path)
            entries = []
            matched = 0
            cap = self.limit or (10_000_000)
            
            for k, v in db.RangeIter():
                if self._stop:
                    break
                if self.prefix_bytes and not k.startswith(self.prefix_bytes):
                    continue
                if self.search_bytes and self.search_bytes not in k:
                    continue
                matched += 1
                entries.append((k, v))
                self.row_found.emit(k, v)
                if matched % 100 == 0:
                    self.progress.emit(matched)
                if matched >= cap:
                    break
            
            return entries if entries else None
        except ImportError:
            return None
        except Exception:
            return None

    def run(self):
        # Try standard plyvel first
        try:
            self.db = open_leveldb_with_fallback(self.db_path)
            # Success with plyvel
            try:
                iterator = self.db.iterator(prefix=self.prefix_bytes) if self.prefix_bytes else self.db.iterator()
                scanned = 0
                matched = 0
                cap = self.limit or (10_000_000)
                for k, v in iterator:
                    if self._stop:
                        break
                    scanned += 1
                    if self.search_bytes and self.search_bytes not in k:
                        continue
                    matched += 1
                    self.row_found.emit(k, v)
                    if matched % 100 == 0:
                        self.progress.emit(matched)
                    if matched >= cap:
                        break
                self.total = matched
                self.finished_scan.emit(matched)
            except Exception as e:
                self.error.emit(str(e))
            finally:
                try:
                    if self.db:
                        self.db.close()
                except Exception:
                    pass
            return
            
        except Exception as e:
            error_str = str(e)
            # Check if it's a comparator mismatch
            if 'COMPARATOR_MISMATCH' in error_str:
                # Try alternative methods
                self.progress.emit(0)
                
                # Method 1: Try leveldb-tools
                entries = read_leveldb_via_tools(self.db_path)
                if entries:
                    matched = 0
                    cap = self.limit or (10_000_000)
                    for k, v in entries:
                        if self._stop:
                            break
                        if self.prefix_bytes and not k.startswith(self.prefix_bytes):
                            continue
                        if self.search_bytes and self.search_bytes not in k:
                            continue
                        matched += 1
                        self.row_found.emit(k, v)
                        if matched % 100 == 0:
                            self.progress.emit(matched)
                        if matched >= cap:
                            break
                    self.total = matched
                    self.finished_scan.emit(matched)
                    return
                
                # Method 2: Try python-leveldb package
                entries = self.read_via_python_leveldb()
                if entries:
                    self.total = len(entries)
                    self.finished_scan.emit(len(entries))
                    return
                
                # If all methods fail, show helpful error
                import re
                comparator_name = 'unknown'
                if 'idb_cmp' in error_str:
                    comparator_name = 'idb_cmp1 (IndexedDB)'
                elif 'cmp' in error_str:
                    match = re.search(r'(\w+_cmp\w*)', error_str)
                    if match:
                        comparator_name = match.group(1)
                
                self.error.emit(
                    f'Cannot open IndexedDB database with custom comparator.\n\n'
                    f'Database uses: {comparator_name}\n'
                    f'Plyvel only supports: leveldb.BytewiseComparator\n\n'
                    f'Alternative methods attempted but failed.\n\n'
                    f'To fix this, try one of these:\n'
                    f'1. Install python-leveldb: pip install leveldb\n'
                    f'   (Then restart this application)\n'
                    f'2. Install leveldb-tools and ensure it\'s in PATH:\n'
                    f'   https://github.com/google/leveldb/tree/main/tools\n'
                    f'3. Use Chrome DevTools for IndexedDB databases\n'
                    f'4. Export data using the original application\n\n'
                    f'Original error: {error_str.replace("COMPARATOR_MISMATCH:", "")}'
                )
            else:
                self.error.emit(str(e))

    def stop(self):
        self._stop = True

# ---------------------------
# Table model for key list
# ---------------------------

class KVTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []  # list of dicts: {key, value, key_type, value_type, key_pretty, value_pretty, key_hex, value_hex}

    def rowCount(self, parent=QModelIndex()):
        return len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 3  # Key, Value Type, Value Preview

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        r = index.row()
        c = index.column()
        row = self.rows[r]
        if role == Qt.DisplayRole:
            if c == 0:
                # short key preview
                kp = row.get('key_pretty')
                if isinstance(kp, str):
                    s = kp
                else:
                    s = repr(kp)
                return s if len(s) <= 80 else s[:77] + '...'
            elif c == 1:
                return row.get('value_type', '')
            elif c == 2:
                vp = row.get('value_pretty')
                if isinstance(vp, str):
                    s = vp
                else:
                    s = repr(vp)
                return s if len(s) <= 80 else s[:77] + '...'
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if section == 0:
                return 'Key'
            elif section == 1:
                return 'Value Type'
            elif section == 2:
                return 'Value Preview'
        return None

    def insert_row(self, rowdict):
        self.beginInsertRows(QModelIndex(), len(self.rows), len(self.rows))
        self.rows.append(rowdict)
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self.rows = []
        self.endResetModel()

# ---------------------------
# Main Window
# ---------------------------

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LevelDBParser')
        self.resize(1100, 700)
        self.db_path = None
        self.tmp_extract = None
        self.worker = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # toolbar
        toolbar = QToolBar()
        open_folder_act = QAction('Open Folder', self)
        open_folder_act.triggered.connect(self.on_open_folder)
        toolbar.addAction(open_folder_act)

        open_zip_act = QAction('Open Zip', self)
        open_zip_act.triggered.connect(self.on_open_zip)
        toolbar.addAction(open_zip_act)

        stop_act = QAction('Stop Scan', self)
        stop_act.triggered.connect(self.on_stop_scan)
        toolbar.addAction(stop_act)

        export_act = QAction('Export CSV', self)
        export_act.triggered.connect(self.on_export_csv)
        toolbar.addAction(export_act)

        layout.addWidget(toolbar)

        # top controls
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel('DB Path:'))
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        ctrl_layout.addWidget(self.path_edit)

        ctrl_layout.addWidget(QLabel('Prefix:'))
        self.prefix_edit = QLineEdit()
        ctrl_layout.addWidget(self.prefix_edit)

        ctrl_layout.addWidget(QLabel('Search (in key):'))
        self.search_edit = QLineEdit()
        ctrl_layout.addWidget(self.search_edit)

        ctrl_layout.addWidget(QLabel('Limit rows:'))
        self.limit_spin = QSpinBox()
        self.limit_spin.setMinimum(10)
        self.limit_spin.setMaximum(5_000_000)
        self.limit_spin.setValue(1000)
        ctrl_layout.addWidget(self.limit_spin)

        self.start_btn = QPushButton('Start Scan')
        self.start_btn.clicked.connect(self.on_start_scan)
        ctrl_layout.addWidget(self.start_btn)

        layout.addLayout(ctrl_layout)

        # main split: left table, right viewer
        main_layout = QHBoxLayout()

        # table
        self.table_model = KVTableModel()
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_view.clicked.connect(self.on_table_clicked)
        main_layout.addWidget(self.table_view, 3)

        # right side viewer
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel('Selected Key'))
        self.key_view = QTextEdit()
        self.key_view.setReadOnly(True)
        right_layout.addWidget(self.key_view)

        right_layout.addWidget(QLabel('Selected Value'))
        self.value_view = QTextEdit()
        self.value_view.setReadOnly(True)
        right_layout.addWidget(self.value_view)

        # small controls for decode view
        decode_layout = QHBoxLayout()
        decode_layout.addWidget(QLabel('Show as:'))
        self.view_combo = QComboBox()
        self.view_combo.addItems(['Auto', 'Text', 'JSON', 'Hex'])
        self.view_combo.currentIndexChanged.connect(self.on_view_mode_change)
        decode_layout.addWidget(self.view_combo)
        right_layout.addLayout(decode_layout)

        main_layout.addLayout(right_layout, 2)

        layout.addLayout(main_layout)

        # progress and status
        status_layout = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate initially
        self.progress.setVisible(False)
        status_layout.addWidget(self.progress)

        self.status_label = QLabel('Idle')
        status_layout.addWidget(self.status_label)

        layout.addLayout(status_layout)

    # ---------------------------
    # Actions
    # ---------------------------
    def on_open_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Select LevelDB Directory')
        if not path:
            return
        # check for typical LevelDB files
        if not (os.path.exists(os.path.join(path, 'CURRENT')) and any(fname.endswith('.ldb') for fname in os.listdir(path))):
            reply = QMessageBox.question(self, 'Not sure', 'Selected folder does not look like a LevelDB directory. Continue anyway?', QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        self.db_path = path
        self.path_edit.setText(path)
        self.status_label.setText('DB selected')

    def on_open_zip(self):
        path_tuple = QFileDialog.getOpenFileName(self, 'Open LevelDB Zip', filter='Zip files (*.zip)')
        path = path_tuple[0]
        if not path:
            return
        tmpdir = tempfile.mkdtemp(prefix='leveldb_')
        try:
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(tmpdir)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to extract zip: {e}')
            return
        self.tmp_extract = tmpdir
        self.db_path = tmpdir
        self.path_edit.setText(tmpdir)
        self.status_label.setText('DB extracted and selected')

    def on_start_scan(self):
        if not self.db_path:
            QMessageBox.warning(self, 'No DB', 'Please open a LevelDB directory or zip first')
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, 'Busy', 'Scan already running')
            return

        # prepare filters
        prefix_text = self.prefix_edit.text().strip()
        prefix_bytes = None
        if prefix_text:
            # if looks like hex
            s = prefix_text
            if all(c in '0123456789abcdefABCDEF' for c in s) and len(s) % 2 == 0:
                try:
                    prefix_bytes = bytes.fromhex(s)
                except Exception:
                    prefix_bytes = s.encode()
            else:
                prefix_bytes = prefix_text.encode()

        search_text = self.search_edit.text().strip()
        search_bytes = search_text.encode() if search_text else None

        limit = self.limit_spin.value()

        # clear model
        self.table_model.clear()
        self.key_view.clear()
        self.value_view.clear()

        # start worker
        self.worker = ScanWorker(self.db_path, prefix_bytes=prefix_bytes, search_bytes=search_bytes, limit=limit)
        self.worker.row_found.connect(self.on_row_found)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_scan.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status_label.setText('Scanning...')
        self.worker.start()

    def on_stop_scan(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText('Stopping...')

    def on_export_csv(self):
        if not self.table_model.rows:
            QMessageBox.information(self, 'No data', 'No rows to export. Run a scan first.')
            return
        outpath, _ = QFileDialog.getSaveFileName(self, 'Save CSV', filter='CSV files (*.csv)')
        if not outpath:
            return
        try:
            with open(outpath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Header with requested columns only
                writer.writerow([
                    'key_type',          # Key type (json/text/bytes/etc)
                    'key_pretty',        # Decoded key text
                    'key_hex',           # Raw key as hex
                    'value_type',        # Value type (json/text/bytes/etc)
                    'value_pretty',      # Decoded value text
                    'value_hex',         # Raw value as hex
                    'value_decoded'      # Decoded value (full text)
                ])
                for r in self.table_model.rows:
                    # Get decoded values
                    key_pretty = r.get('key_pretty', '')
                    value_pretty = r.get('value_pretty', '')
                    
                    # If values are still hex strings, try to decode them
                    # This handles cases where hex strings weren't decoded during initial processing
                    key_decoded = decode_hex_string_if_needed(key_pretty)
                    value_decoded = decode_hex_string_if_needed(value_pretty)
                    
                    # If decoding produced different result, use it; otherwise use original
                    if key_decoded != key_pretty and key_decoded:
                        key_pretty = key_decoded
                    if value_decoded != value_pretty and value_decoded:
                        value_pretty = value_decoded
                    
                    # Write only requested columns
                    writer.writerow([
                        r.get('key_type', ''),          # Key type
                        key_pretty,                      # Decoded key
                        r.get('key_hex', ''),           # Key hex
                        r.get('value_type', ''),        # Value type
                        value_pretty,                    # Decoded value
                        r.get('value_hex', ''),         # Value hex
                        value_pretty                     # Decoded value (full)
                    ])
            QMessageBox.information(self, 'Saved', f'CSV saved to {outpath}\nExported {len(self.table_model.rows)} rows.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save CSV: {e}')

    # ---------------------------
    # Worker signals
    # ---------------------------
    def on_row_found(self, k, v):
        # decode and add to model
        kd = decode_and_detect(k)
        vd = decode_and_detect(v)
        row = {
            'key_raw': k,  # Store raw bytes for export
            'key_type': kd['type'],
            'key_pretty': kd['pretty'],
            'key_hex': kd['hex'],
            'key_base64': base64.b64encode(k).decode('ascii') if k else '',  # Base64 representation
            'value_raw': v,  # Store raw bytes for export
            'value_type': vd['type'],
            'value_pretty': vd['pretty'],
            'value_hex': vd['hex'],
            'value_base64': base64.b64encode(v).decode('ascii') if v else ''  # Base64 representation
        }
        self.table_model.insert_row(row)

    def on_progress(self, matched):
        # show matched count
        self.status_label.setText(f'Scanned & matched: {matched}')

    def on_finished(self, total):
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.status_label.setText(f'Scan finished. Matched: {total}.')

    def on_error(self, msg):
        QMessageBox.critical(self, 'Error', msg)
        self.progress.setVisible(False)
        self.status_label.setText('Error')

    # ---------------------------
    # Table selection
    # ---------------------------
    def on_table_clicked(self, index: QModelIndex):
        r = index.row()
        if r < 0 or r >= len(self.table_model.rows):
            return
        row = self.table_model.rows[r]
        self.key_view.setPlainText(row.get('key_pretty') or '')
        # show according to view mode
        mode = self.view_combo.currentText()
        if mode == 'Auto':
            if row.get('value_type') == 'json':
                try:
                    self.value_view.setPlainText(row.get('value_pretty') or '')
                except Exception:
                    self.value_view.setPlainText(row.get('value_hex') or '')
            else:
                self.value_view.setPlainText(row.get('value_pretty') or '')
        elif mode == 'Text':
            self.value_view.setPlainText(row.get('value_pretty') or '')
        elif mode == 'JSON':
            if row.get('value_type') == 'json':
                self.value_view.setPlainText(row.get('value_pretty') or '')
            else:
                QMessageBox.information(self, 'Not JSON', 'Selected value is not detected as JSON')
        elif mode == 'Hex':
            self.value_view.setPlainText(row.get('value_hex') or '')

    def on_view_mode_change(self, idx):
        # refresh currently selected row view
        sel = self.table_view.selectionModel().currentIndex()
        if sel.isValid():
            self.on_table_clicked(sel)

    def closeEvent(self, event):
        # cleanup worker and tmp dir
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        if self.tmp_extract and os.path.isdir(self.tmp_extract):
            try:
                import shutil
                shutil.rmtree(self.tmp_extract)
            except Exception:
                pass
        event.accept()

# ---------------------------
# Entry
# ---------------------------

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
