# test_envelope.py
import os, wave, json, struct, base64, secrets
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

INPUT_DIR = "sample"
OUTPUT_DIR = "output"
MASTER_PASSWORD = "voicecrypto_master"  # 사용자 마스터 비밀번호 (하나만 기억)
KDF_ITERS = 200_000
SHIFT_SEMITONES = 4

def derive_master_key(password: str, salt: bytes = None, iterations: int = KDF_ITERS) -> Tuple[bytes, bytes]:
    if salt is None:
        salt = secrets.token_bytes(16)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations, backend=default_backend())
    key = kdf.derive(password.encode('utf-8'))
    return key, salt

def pack_file(path: str, meta: dict, ciphertext: bytes):
    meta_bytes = json.dumps(meta).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(struct.pack(">I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(ciphertext)

def encrypt_with_envelope(frames: bytes, params: dict, out_path: str, master_password: str):
    # 1) 파일별 랜덤 키 생성 (file_key)
    file_key = secrets.token_bytes(32)  # 256-bit

    # 2) file_key로 프레임 암호화
    aes_file = AESGCM(file_key)
    nonce_file = secrets.token_bytes(12)
    ciphertext = aes_file.encrypt(nonce_file, frames, None)

    # 3) master_key 유도 (PBKDF2) 및 file_key 암호화 (encrypted_file_key)
    master_key, salt = derive_master_key(master_password)
    aes_master = AESGCM(master_key)
    nonce_key = secrets.token_bytes(12)
    encrypted_file_key = aes_master.encrypt(nonce_key, file_key, None)

    meta = {
        "nchannels": params["nchannels"],
        "sampwidth": params["sampwidth"],
        "framerate": params["framerate"],
        "nframes": params["nframes"],
        "comptype": params["comptype"],
        "compname": params["compname"],
        # file key envelope info
        "encrypted_file_key": base64.b64encode(encrypted_file_key).decode(),
        "nonce_key": base64.b64encode(nonce_key).decode(),
        "salt": base64.b64encode(salt).decode(),
        "kdf_iterations": KDF_ITERS,
        # file ciphertext nonce
        "nonce_file": base64.b64encode(nonce_file).decode()
    }
    pack_file(out_path, meta, ciphertext)

def modulate_audio(frames: bytes, params: dict, out_path: str, shift: int):
    factor = 2 ** (shift / 12.0)
    new_rate = int(params["framerate"] * factor)
    with wave.open(out_path, 'wb') as w:
        w.setnchannels(params["nchannels"])
        w.setsampwidth(params["sampwidth"])
        w.setframerate(new_rate)
        w.writeframes(frames)

def process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wavs = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav")]
    if not wavs:
        print("No wavs in", INPUT_DIR); return
    for wav in wavs:
        inp = os.path.join(INPUT_DIR, wav)
        base = os.path.splitext(wav)[0]
        enc_out = os.path.join(OUTPUT_DIR, base + ".enc")
        mod_out = os.path.join(OUTPUT_DIR, base + "_mod.wav")
        with wave.open(inp, 'rb') as w:
            params = {
                "nchannels": w.getnchannels(),
                "sampwidth": w.getsampwidth(),
                "framerate": w.getframerate(),
                "nframes": w.getnframes(),
                "comptype": w.getcomptype(),
                "compname": w.getcompname()
            }
            frames = w.readframes(w.getnframes())

        encrypt_with_envelope(frames, params, enc_out, MASTER_PASSWORD)
        modulate_audio(frames, params, mod_out, SHIFT_SEMITONES)
        print(f"[OK] {wav} -> {os.path.basename(enc_out)}, {os.path.basename(mod_out)}")

if __name__ == "__main__":
    process_all()
