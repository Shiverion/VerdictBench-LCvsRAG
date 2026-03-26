import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv('.env')

class GroundingCheckSchema(BaseModel):
    supported: bool
    reason: str

prompt = """Anda adalah evaluator sistem QA hukum.

Tugas: Periksa apakah pernyataan berikut didukung oleh konteks yang diberikan.
Jika didukung (secara eksplisit maupun implisit kuat), kembalikan true.
Jika tidak didukung atau bertentangan, kembalikan false.

Berikan jawaban dalam format JSON:
{"supported": true/false, "reason": "alasan singkat"}

Pernyataan:
Undang-undang mengatur hal tersebut.

Konteks:
Undang-undang nomor 5 mengatur tentang hal tersebut dengan jelas.
"""

client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
try:
    res = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type='application/json',
            response_schema=GroundingCheckSchema
        )
    )
    print('TEXT:', repr(res.text))
except Exception as e:
    print('ERROR:', e)
