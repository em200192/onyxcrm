# test_gcs_direct_upload.py
from pathlib import Path
from datetime import datetime, timezone
from google.cloud import storage
from google.oauth2 import service_account

# ---- EDIT THESE ----
KEY_PATH = Path(r"C:\Users\Ultimate\Downloads\crm-472718-560ccdc4318b.json")
BUCKET_NAME = "onyx-escalations"  # must already exist
# --------------------

def main():
    if not KEY_PATH.exists():
        raise FileNotFoundError(f"Key not found: {KEY_PATH}")

    creds = service_account.Credentials.from_service_account_file(str(KEY_PATH))
    client = storage.Client(project=creds.project_id, credentials=creds)
    print(f"[OK] Authenticated for project: {client.project}")

    # No RPC here; just a local handle to the bucket name.
    bucket = client.bucket(BUCKET_NAME)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    object_name = f"tests/hello-{ts}.txt"
    content = f"Hello from explicit credentials! UTC: {ts}\n"

    try:
        blob = bucket.blob(object_name)
        blob.upload_from_string(content)  # requires storage.objects.create on the bucket
        print(f"[OK] Uploaded: gs://{BUCKET_NAME}/{object_name}")

        # Read it back (requires storage.objects.get)
        back = blob.download_as_text()
        print(f"[OK] Read back: {back.strip()}")

    except Exception as e:
        print(f"[ERROR] {e.__class__.__name__}: {e}")
        print(
            "\nIf you see 403 on upload/read:\n"
            " - Ensure the bucket exists.\n"
            " - Grant this service account object permissions on that bucket:\n"
            "     roles/storage.objectAdmin (or at minimum storage.objects.create + storage.objects.get)\n"
            "   Bucket-level IAM is enough; project-wide is not required."
        )

if __name__ == "__main__":
    main()
