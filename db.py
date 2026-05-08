import sqlite3
import os
import logging
import threading
from contextlib import contextmanager
from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")
_db_lock = threading.Lock()

@contextmanager
def get_db():
    """Thread-safe database connection with auto-commit and rollback."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Create the users table if it does not exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    logger.info("Database initialized at %s", DB_PATH)

def create_user(email: str, password: str) -> tuple[int | None, str | None]:
    """Register a new user. Returns (user_id, error_message)."""
    email = email.strip().lower()

    if not email or not password:
        return None, "Email and password are required"

    if len(password) < 6:
        return None, "Password must be at least 6 characters"

    password_hash = generate_password_hash(password)

    with _db_lock:
        try:
            with get_db() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                    (email, password_hash),
                )
                user_id = cursor.lastrowid
            logger.info("User registered: %s", email)
            return user_id, None
        except sqlite3.IntegrityError:
            return None, "Email already registered"
        except Exception as e:
            logger.error("Registration error: %s", e)
            return None, "Something went wrong. Please try again."

def authenticate_user(email: str, password: str) -> tuple[dict | None, str | None]:
    """Validate login credentials. Returns (user_dict, error_message)."""
    email = email.strip().lower()

    if not email or not password:
        return None, "Email and password are required"

    with get_db() as conn:
        user = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if not user:
        return None, "No account found with this email"

    if not check_password_hash(user["password_hash"], password):
        return None, "Incorrect password"

    return {"id": user["id"], "email": user["email"]}, None

def get_user_by_id(user_id: int) -> dict | None:
    """Fetch user by ID. Returns None if not found."""
    with get_db() as conn:
        user = conn.execute(
            "SELECT id, email FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    
    if user:
        return {"id": user["id"], "email": user["email"]}
    return None