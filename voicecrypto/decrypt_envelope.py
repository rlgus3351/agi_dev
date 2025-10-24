# decrypt_envelope.py
import os, json, struct, base64, wave
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

INPUT_DIR = "output"
OUTPUT_DIR = "restored"
MASTER_PASSWORD = "voicecrypto_master"
KDF_ITERS = 200_000

def unpack_file(path: str):
    with open(path, 'rb') as f:
        raw = f.read()
    meta_len = struct.unpack(">I", raw[:4])[0]
    meta = json.loads(raw[4:4+meta_len].decode('utf-8'))
    ciphertext = raw[4+meta_len:]
    return meta, ciphertext

def derive_master_key(password: str, salt: bytes, iterations: int = KDF_ITERS):
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations, backend=default_backend())
    return kdf.derive(password.encode('utf-8'))

def decrypt_file(enc_path: str, out_wav: str, master_password: str):
    meta, ciphertext = unpack_file(enc_path)
    salt = base64.b64decode(meta["salt"])
    nonce_key = base64.b64decode(meta["nonce_key"])
    encrypted_file_key = base64.b64decode(meta["encrypted_file_key"])
    nonce_file = base64.b64decode(meta["nonce_file"])

    master_key = derive_master_key(master_password, salt, meta.get("kdf_iterations", KDF_ITERS))
    aes_master = AESGCM(master_key)
    file_key = aes_master.decrypt(nonce_key, encrypted_file_key, None)

    aes_file = AESGCM(file_key)
    frames = aes_file.decrypt(nonce_file, ciphertext, None)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with wave.open(out_wav, 'wb') as w:
        w.setnchannels(meta["nchannels"])
        w.setsampwidth(meta["sampwidth"])
        w.setframerate(meta["framerate"])
        w.writeframes(frames)

    print(f"[OK] Decrypted {enc_path} -> {out_wav}")

def main():
    encs = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".enc")]
    if not encs:
        print("No .enc in", INPUT_DIR); return
    for e in encs:
        in_path = os.path.join(INPUT_DIR, e)
        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(e)[0] + "_restored.wav")
        try:
            decrypt_file(in_path, out_path, MASTER_PASSWORD)
        except Exception as ex:
            print("[ERROR]", e, ex)

if __name__ == "__main__":
    main()
