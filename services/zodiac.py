import openai

def build_zodiac_prompt(payload):
    lang = payload.language
    dob = payload.birth_date.strftime("%Y-%m-%d")
    tob = payload.birth_time or "Unknown"
    gender = payload.gender or "Unknown"

    if lang == "vi":
        return f"Phân tích chi tiết cung hoàng đạo cho người sinh ngày {dob}, lúc {tob}, giới tính {gender}. Hãy viết theo phong cách chuyên gia tử vi, phân tích các khía cạnh: tính cách, sự nghiệp, tình yêu, gia đình, và lời khuyên."
    else:
        return f"Provide a detailed zodiac analysis for a person born on {dob} at {tob}, gender {gender}. Write in the style of a professional astrologer, covering personality, career, love, family, and advice."

def call_chatgpt(prompt, lang="vi"):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Bạn là một chuyên gia chiêm tinh học." if lang == "vi" else "You are an astrology expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content