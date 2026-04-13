import sqlite3
import os
import datetime

class TemporalKnowledgeGraph:
    """
    SQLite-backed Temporal Knowledge Graph representing exact MemPalace concepts.
    Stores explicit entity relationships (subject -> predicate -> object) with temporal validity.
    """
    def __init__(self, db_path="src/data/market_knowledge.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_until TEXT,
                UNIQUE(subject, predicate, object, valid_from)
            )
        ''')
        conn.commit()
        conn.close()

    def add_triple(self, subject, predicate, obj, valid_from=None):
        if not valid_from:
            valid_from = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO knowledge_triples (subject, predicate, object, valid_from, valid_until)
            VALUES (?, ?, ?, ?, ?)
        ''', (subject, predicate, obj, valid_from, None))
        conn.commit()
        conn.close()

    def invalidate(self, subject, predicate, obj, ended=None):
        if not ended:
            ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE knowledge_triples
            SET valid_until = ?
            WHERE subject = ? AND predicate = ? AND object = ? AND valid_until IS NULL
        ''', (ended, subject, predicate, obj))
        conn.commit()
        conn.close()

    def query_entity(self, subject, as_of=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if as_of:
            cursor.execute('''
                SELECT subject, predicate, object, valid_from, valid_until FROM knowledge_triples
                WHERE subject = ? AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?)
            ''', (subject, as_of, as_of))
        else:
            cursor.execute('''
                SELECT subject, predicate, object, valid_from, valid_until FROM knowledge_triples
                WHERE subject = ? AND valid_until IS NULL
            ''', (subject,))
            
        results = cursor.fetchall()
        conn.close()
        return [{"subject": r[0], "predicate": r[1], "object": r[2], "valid_from": r[3], "valid_until": r[4]} for r in results]
        
    def get_all_active(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT subject, predicate, object FROM knowledge_triples
            WHERE valid_until IS NULL
        ''')
        results = cursor.fetchall()
        conn.close()
        return [{"subject": r[0], "predicate": r[1], "object": r[2]} for r in results]
