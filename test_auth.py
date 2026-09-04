import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import auth


class UserSeedTests(unittest.TestCase):
    def test_removed_last_user_is_not_restored_on_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "audioscribe.db"
            with patch.object(auth, "DB_PATH", db_path), patch.dict(
                os.environ, {"ALLOWED_EMAILS": "owner@example.com"}
            ):
                auth.init_db()
                self.assertEqual(auth.list_users(), ["owner@example.com"])

                auth.remove_user("owner@example.com")
                auth.init_db()

                self.assertEqual(auth.list_users(), [])

    def test_existing_empty_user_store_is_not_seeded_during_upgrade(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "audioscribe.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "CREATE TABLE users (email TEXT PRIMARY KEY, created_at REAL NOT NULL)"
                )

            with patch.object(auth, "DB_PATH", db_path), patch.dict(
                os.environ, {"ALLOWED_EMAILS": "former-user@example.com"}
            ):
                auth.init_db()

                self.assertEqual(auth.list_users(), [])

    def test_failed_initial_seed_is_retried_on_next_start(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "audioscribe.db"
            with patch.object(auth, "DB_PATH", db_path), patch.dict(
                os.environ, {"ALLOWED_EMAILS": "owner@example.com"}
            ):
                with patch.object(
                    auth, "_seed_users_from_env", side_effect=RuntimeError("boom")
                ):
                    with self.assertRaises(RuntimeError):
                        auth.init_db()

                auth.init_db()

                self.assertEqual(auth.list_users(), ["owner@example.com"])


if __name__ == "__main__":
    unittest.main()
