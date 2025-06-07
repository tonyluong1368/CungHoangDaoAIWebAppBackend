from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import asyncio
from openai import AsyncOpenAI
import tiktoken

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
DEBUG = os.getenv("DEBUG", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Set up OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Allow CORS for local and LAN access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://192.168.0.182:5173",
        "https://tonyluong1368.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class ZodiacRequest(BaseModel):
    name: str = ""
    birth_date: str
    birth_time: str = None
    gender: str
    language: str

SECTIONS = [
    "Tổng quan",
    "Tính cách",
    "Tình yêu",
    "Sự nghiệp",
    "Gia đình",
    "Tâm linh",
    "Sứ mệnh cuộc đời",
    "Tiềm năng ẩn giấu",
    "Nhân số học",
    "Human Design",
]

@app.post("/zodiac-analysis")
async def analyze_zodiac(request: Request):
    """
    Analyze user's birth information and return metaphysical/astrological insights for each section.
    """
    data = await request.json()
    detail_level = data.get("detail_level", "deep")
    name = data.get("name") or "Ẩn danh"

    prompts, models, max_tokens_list = [], [], []

    high_depth_sections = ["Tổng quan", "Tính cách", "Tâm linh"]

    # Prepare prompts and settings for each section
    for section in SECTIONS:
        prompt = f"""
Dưới đây là thông tin người dùng:
- Họ tên: {name}
- Ngày sinh: {data['birth_date']}
- Giờ sinh: {data.get('birth_time') or 'Không rõ'}
- Giới tính: {data['gender']}
- Ngôn ngữ: {data['language']}

Bạn là một chuyên gia về huyền học, dựa vào tất cả kiến thức và khả năng tổng hợp của bạn ở tất cả bộ môn huyền học bạn đã biết: Tử vi, Kinh dịch, Bát tự, Thần số học, Human Design ... 
và mô phỏng cách truy cập vào thư viện Akashic để tổng kết một bức tranh hoàn chỉnh về cuộc đời, sứ mệnh của {name} trên góc nhìn huyền học đặc biệt chỉ tập trung về chủ đề sau đây:
**{section}**

Viết câu trả lời bằng tiếng {data['language']} sử dụng định dạng **Markdown**, dài khoảng 5–10 đoạn văn chiêm nghiệm, rõ ràng và giàu thông tin, mang chiều sâu, nhiều suy ngẫm.
        """.strip()
        
        # Choose model and tokens
        if detail_level == "fast" or section not in high_depth_sections:
            model = "gpt-3.5-turbo-1106"
            max_tokens = 2000
        else:
            model = "gpt-4-1106-preview"
            max_tokens = 3000

        encoding = tiktoken.encoding_for_model(model)
        prompt_tokens = len(encoding.encode(prompt))
        max_tokens = min(max_tokens, 4096 - prompt_tokens)

        prompts.append(prompt)
        models.append(model)
        max_tokens_list.append(max_tokens)

    # --- Async LLM calls ---
    async def ask_section(section, prompt, model, max_tokens):
        if DEBUG:
            print(f"[PROMPT for {section} | model: {model} | max_tokens: {max_tokens}]\n{prompt}\n---")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            content = response.choices[0].message.content.strip()
            if DEBUG:
                print(f"[Section: {section} | Model: {model}] Length: {len(content)} chars")
            return section, content
        except Exception as e:
            if DEBUG:
                print(f"Error generating section: {section} | {e}")
            return section, f"Đã xảy ra lỗi khi phân tích mục {section}."

    # Run all sections in parallel
    coroutines = [
        ask_section(section, prompt, model, max_tokens)
        for section, prompt, model, max_tokens in zip(SECTIONS, prompts, models, max_tokens_list)
    ]
    results = await asyncio.gather(*coroutines)

    # Return mapping: section name -> result
    return {section: analysis for section, analysis in results}