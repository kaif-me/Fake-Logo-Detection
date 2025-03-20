import firebase_admin
from firebase_admin import credentials, storage

# Firebase Configuration
cred = credentials.Certificate("app/firebase_config.json")
firebase_admin.initialize_app(cred, {"storageBucket": "your-project-id.appspot.com"})

bucket = storage.bucket()

# Upload a test file
def upload_test():
    blob = bucket.blob("test_upload.txt")
    blob.upload_from_string("Hello, Firebase!")
    blob.make_public()
    print("File uploaded to:", blob.public_url)

if __name__ == "__main__":
    upload_test()
