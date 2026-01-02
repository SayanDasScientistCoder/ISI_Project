from pymongo import MongoClient
from config import MONGO_URI
from datetime import datetime, timezone

client = MongoClient(MONGO_URI)
users = client["auth_db"]["users"]

users.update_many(
    {},
    {
        "$setOnInsert": {
            "created_at": datetime.now(timezone.utc)
        },
        "$set": {
            "last_login": None,
            "total_uploads": 0,
            "total_predictions": 0,
            "account_type": "Free"
        }
    }
)

print("Migration complete")
