from fastapi import FastAPI, Form,File, UploadFile, HTTPException,Query
from sqlalchemy import create_engine, Column, Integer, Float, String, Text,LargeBinary,ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,relationship
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.cloud import vision
from PIL import ImageFont, ImageDraw, Image
from typing import List
import requests
import json
from datetime import datetime
import hmac
import hashlib
from pytz import timezone

import asyncio
import numpy as np
import os
import sys
import io
import platform
import re
import cv2
import shutil

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

#권장섭취량 
class Recommended_Intake(Base):
    __tablename__ = "recommended_intake"

    id = Column(Integer, primary_key=True, index=True)
    recommended_cal = Column(Float)
    recommended_nat = Column(Integer)
    recommended_carbs = Column(Float)
    recommended_protein = Column(Float)
    recommended_fat = Column(Float)
    


#일일동안 먹은 총 섭취 영양소(현재 섭취량)
class DayTotalSum(Base):
    __tablename__ = "day_total_sum"

    id = Column(Integer, primary_key=True, index=True)
    Total_food_cal = Column(Float)
    Total_food_nat = Column(Float)
    Total_food_carbs = Column(Float)
    Total_food_protein = Column(Float)
    Total_food_fat = Column(Float)
   

#음식 *먹은양 통합 데이터베이스
class TotalFoodInfo(Base):
    __tablename__ = "food_total_info"

    id = Column(Integer, primary_key=True, index=True)
    Total_food_name = Column(String)
    Total_food_cal = Column(Float)
    Total_food_nat = Column(Float)
    Total_food_carbs = Column(Float)
    Total_food_protein = Column(Float)
    Total_food_fat = Column(Float)
 

#음식사진 인식 결과 저장 테이블
class FoodImageInfo(Base):
    __tablename__ = "food_image_info"

    id = Column(Integer, primary_key=True, index=True)
    food_name = Column(String)
    food_cal = Column(Float)
    food_nat = Column(Float)
    food_carbs = Column(Float)
    food_protein = Column(Float)
    food_fat = Column(Float)
    
    



#회원정보 테이블 추가
class UserJoin(Base):
    __tablename__ = "userjoin"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    identity = Column(String)
    password = Column(String)
    email = Column(String)
    age = Column(Integer)
    weight = Column(Float)
    height = Column(Float)
    gender = Column(String)
    medical_history = Column(String)
    


Base.metadata.create_all(bind=engine)


app = FastAPI()

# CORS 설정
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)




#개인정보 api 
@app.post("/user_join/")
async def submit_join(
    
    name :str=Form(...),
    identity:str=Form(...),
    password :str=Form(...),
    email :str=Form(...),
    age :int=Form(...),
    weight:float=Form(...),
    height:float=Form(...),
    gender :str=Form(...),
    medical_history :str=Form(...),
    
):
    db_data = UserJoin(
        name=name,identity=identity, password=password,email=email, age=age, weight=weight, height=height, 
        gender=gender, medical_history=medical_history)

    db = SessionLocal()
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    db.close()

    response_data = {
        "name":name,
        "identity":identity,
        "password":password,
        "emali":email,
        "age": age,
        "weight": weight,
        "height": height,
        "gender": gender,
        "medical_history": medical_history,
    }

    print("Received data:", response_data)

    return response_data



    
@app.post("/process_foodimage/")
async def process_image(file: UploadFile = File(...), flag: str = "ALL"):
    timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d%H%M%S%f")[:-3]
    client_id = "glabs_638c223a818794216d1ba2d03f8f395054565ac1b5bc948c9ff6f392195615be"
    client_secret = "fb66b6fc7ac25cdd55439205994f85b6729c7f400674c3d1acddd007b003c6e4"
    client_key = "80aa5a78-3ba6-546f-aefb-3aa7a47dfa77"
    signature = hmac.new(
        key=client_secret.encode("UTF-8"),
        msg=f"{client_id}:{timestamp}".encode("UTF-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    url = "https://aiapi.genielabs.ai/kt/vision/food"
    headers = {
        "Accept": "*/*",
        "x-client-key": client_key,
        "x-client-signature": signature,
        "x-auth-timestamp": timestamp,
    }
    fields = {"flag": flag}
    obj = {"metadata": json.dumps(fields), "media": file.file}

    response = requests.post(url, headers=headers, files=obj)

    if response.ok:
        json_data = json.loads(response.text)
        code = json_data["code"]
        data = json_data["data"]
        prediction_top1 = data[0]["region_0"]["prediction_top1"]

        save_to_database(prediction_top1)

       
    
def save_to_database(prediction_top1):
    food_name = prediction_top1.get("food_name", "")
    food_cal = prediction_top1.get("food_cal", 0.0)
    food_nat = prediction_top1.get("food_nat", 0.0)
    food_carbs = prediction_top1.get("food_carbs", 0.0)
    food_protein = prediction_top1.get("food_protein", 0.0)
    food_fat = prediction_top1.get("food_fat", 0.0)
    

    db = SessionLocal()
    db_food_result = FoodImageInfo(
        food_name=food_name,
        food_cal=food_cal,
        food_nat=food_nat,
        food_carbs=food_carbs,
        food_protein=food_protein,
        food_fat=food_fat,
    )
    db.add(db_food_result)
    db.commit()
    db.refresh(db_food_result)
    db.close()

    
   



# 이미지 처리 결과를 반환하는 API 엔드포인트
@app.get("/food_image_info/")
async def  fetch_food_image_info():
    db = SessionLocal()
    data = db.query(FoodImageInfo).all()
    db.close()
    return data

#개인정보 페이지 정보 엔드포인트
@app.get("/user_info/")
async def fetch_user_join():
    db=SessionLocal()
    data=db.query(UserJoin).all()
    user_data = []
    for user in data:
        user_data.append({
            "name": user.name,
            "age": user.age,
            "weight": user.weight,
            "height": user.height,
            "gender": user.gender,
            "medical_history": user.medical_history,
        })
    db.close()
    return user_data

# 사용자 정보를 업데이트하는 PUT 엔드포인트
@app.put("/user_update/")
async def update_user(
    name: str = Form(...),
    weight: float = Form(...),
    height: float = Form(...),
    gender: str = Form(...),
    medical_history: str = Form(...),
    
):
    db = SessionLocal()
    user = db.query(UserJoin).order_by(UserJoin.id.desc()).first()
    if user is None:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")

    # 사용자 정보 업데이트
    user.name=name
    user.weight = weight
    user.height = height
    user.gender = gender
    user.medical_history = medical_history

    # 데이터베이스에 변경사항 반영
    db.commit()
    db.close()
    # 업데이트된 사용자 정보를 반환
    # 업데이트가 성공했음을 알리는 메시지를 반환

    return {"message": "사용자 정보가 성공적으로 업데이트되었습니다."}


#음식*먹은양 ->새로운 데이터베이스 (음식사진)
@app.post("/total_food_result/")
async def submit_join(
    amount_eaten: float = Form(...),
    
):
    db = SessionLocal()
    # 가장 최근의 음식 정보를 가져옴
    food_image_info = db.query(FoodImageInfo).order_by(FoodImageInfo.id.desc()).first()

    if not food_image_info:
        raise HTTPException(status_code=404, detail="음식 정보를 찾을 수 없습니다.")

   # TotalFoodInfo에 저장할 데이터 생성
    total_food_data = TotalFoodInfo(
        Total_food_name=food_image_info.food_name,
        Total_food_cal=food_image_info.food_cal * amount_eaten,
        Total_food_nat=food_image_info.food_nat * amount_eaten,
        Total_food_carbs=food_image_info.food_carbs * amount_eaten,
        Total_food_protein=food_image_info.food_protein * amount_eaten,
        Total_food_fat=food_image_info.food_fat * amount_eaten,
    
    )

    # TotalFoodInfo 테이블에 저장
    db.add(total_food_data)
    db.commit()
    db.refresh(total_food_data)

    response_data = {
        "message": "Data successfully submitted",
        "total_food_data": {
            "Total_food_name": total_food_data.Total_food_name,
            "Total_food_cal": total_food_data.Total_food_cal,
            "Total_food_nat": total_food_data.Total_food_nat,
            "Total_food_carbs": total_food_data.Total_food_carbs,
            "Total_food_protein": total_food_data.Total_food_protein,
            "Total_food_fat": total_food_data.Total_food_fat,
           
        },
    }

    return response_data

#음식*먹은양 ->새로운 데이터베이스 (ocr )
@app.post("/total_text_result/")
async def submit_join(
    amount_eaten: float = Form(...),
    name: str =Form(...),
):
    db = SessionLocal()
    # 가장 최근의 음식 정보를 가져옴
    food_image_info = db.query(FoodImageInfo).order_by(FoodImageInfo.id.desc()).first()

    if not food_image_info:
        raise HTTPException(status_code=404, detail="음식 정보를 찾을 수 없습니다.")

   # TotalFoodInfo에 저장할 데이터 생성
    total_food_data = TotalFoodInfo(
        Total_food_name=name,
        Total_food_cal=1000,
        Total_food_nat=food_image_info.food_nat * amount_eaten,
        Total_food_carbs=food_image_info.food_carbs * amount_eaten,
        Total_food_protein=food_image_info.food_protein * amount_eaten,
        Total_food_fat=food_image_info.food_fat * amount_eaten,
    
    )

    # TotalFoodInfo 테이블에 저장
    db.add(total_food_data)
    db.commit()
    db.refresh(total_food_data)

    response_data = {
        "message": "Data successfully submitted",
        "total_food_data": {
            "Total_food_name": total_food_data.Total_food_name,
            "Total_food_cal": total_food_data.Total_food_cal,
            "Total_food_nat": total_food_data.Total_food_nat,
            "Total_food_carbs": total_food_data.Total_food_carbs,
            "Total_food_protein": total_food_data.Total_food_protein,
            "Total_food_fat": total_food_data.Total_food_fat,
            
        },
    }

    return response_data

##avc

##
#총 섭취량 엔드포인트
@app.get("/today_sum_food/")
async def total_food_sum():
     db = SessionLocal()
     data = db.query(TotalFoodInfo).all()

    # 각 항목별 총합을 계산
     day_sum =  DayTotalSum(
        Total_food_cal = sum(food.Total_food_cal for food in data),
         Total_food_nat= sum(food.Total_food_nat for food in data),
        Total_food_carbs  = sum(food.Total_food_carbs for food in data),
         Total_food_protein  = sum(food.Total_food_protein for food in data),
         Total_food_fat = sum(food.Total_food_fat for food in data),
     )
      #TotalFoodInfo에 저장
     db.add(day_sum)
     db.commit()
     db.refresh(day_sum)

     response_data = {
        "message": "Data successfully submitted",
        "total_food_data": {
        
            "Total_food_cal": day_sum.Total_food_cal,
            "Total_food_nat": day_sum.Total_food_nat,
            "Total_food_carbs": day_sum.Total_food_carbs,
            "Total_food_protein": day_sum.Total_food_protein,
            "Total_food_fat": day_sum.Total_food_fat,
           
        },
      }

     return response_data

#권장섭취량
@app.get("/recommended_intake/")
async def recommended_intake():
     db = SessionLocal()
     user_data = db.query(UserJoin).order_by(UserJoin.id.desc()).first()

     height = user_data.height / 100
     age = user_data.age
     weight = user_data.weight
     
     #남자 활동적 식
     calo = 662 - (9.53 * age) + 1.25 * ((15.91 * weight) + (539.6 * height))

     

    # 각 항목별 총합을 계산
     recommended = Recommended_Intake(
     recommended_cal=calo,
     recommended_nat=2300,
      recommended_carbs=calo*0.65/4,
      recommended_protein=calo*0.15/4,
      recommended_fat=calo*0.2/9,
     
)
    
      #Reocmmended Intake에 저장
     db.add(recommended)
     db.commit()
     db.refresh(recommended)

     response_data = {
    "message": "데이터가 성공적으로 제출되었습니다.",
    "response_data": {
        "recommended_cal": recommended.recommended_cal,
        "recommended_nat": recommended.recommended_nat,
        "recommended_carbs": recommended.recommended_carbs,
        "recommended_protein": recommended.recommended_protein,
        "recommended_fat": recommended.recommended_fat,
    },
}

     return response_data

# 먹은양*음식성분 결과 엔드포인트
@app.get("/list_food_info/")
async def  fetch_food_image_info():
    db = SessionLocal()
    data = db.query(TotalFoodInfo).all()
    db.close()
    return data



# 텍스트 인식 결과 반환
@app.get("/fetch_textimage/")
async def fetch_nutrition_info():
    db = SessionLocal()
    data = db.query(TextImageInfo).all()
    db.close()
    return data

#OCR 인식인듯
@app.post("/OCRprocess_image")
async def process_image(file: UploadFile = File(...), flag: str = Query(None)):
    # 파일 저장 위치 설정
    current_directory = os.path.abspath(os.getcwd())
    directory = os.path.join(current_directory, "temp_images")

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 파일 저장 위치 설정
    file_location = os.path.join(directory, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # OCR 처리 함수 비동기적으로 호출
    result = await asyncio.to_thread(process_ocr, file_location, flag)
    return {"results": result}

def process_ocr(file_path, flag):
    # Google Vision API 클라이언트 설정
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\toong\Documents\카카오톡 받은 파일\linen-walker-216606-76f54386771c.json"
    client_options = {'api_endpoint': 'eu-vision.googleapis.com'}
    client = vision.ImageAnnotatorClient(client_options=client_options)

    # 이미지 파일 읽기
    with io.open(file_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    img1 = cv2.imread(file_path)

    text_to_display = ""
    text_output_flag = True

    for text in texts:
        if text_output_flag:
            ocr_text = text.description
            x1 = text.bounding_poly.vertices[0].x
            y1 = text.bounding_poly.vertices[0].y
            x2 = text.bounding_poly.vertices[1].x
            y2 = text.bounding_poly.vertices[2].y

            cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

            text_to_display += ocr_text + "\n"
            text_output_flag = False

    # 함수를 호출하여 결과 얻기
    nutrition_info = process_nutrition_info(text_to_display)
    json_data = json.dumps(nutrition_info, indent=4, ensure_ascii=False)

# 데이터베이스에 결과 저장
    db = SessionLocal()
    try:
        # 여기서 nutrition_info를 TextImageInfo 모델에 맞게 파싱하고 저장해야 함
        # 예시: new_record = TextImageInfo(food_name="example", food_cal=100, ...)
        new_record = TextImageInfo(
            text_nat=nutrition_info.get("나트륨", "Unknown"),
            text_carbs=nutrition_info.get("탄수화물", "Unknown"),
            text_protein=nutrition_info.get("단백질", "Unknown"),
            text_fat=nutrition_info.get("지방", "Unknown")
        )  # 적절한 필드 값으로 채워야 함
        db.add(new_record)
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

    return json_data


def putText(image, text, x, y, color=(0, 255, 0), font_size=22):
    if type(image) == np.ndarray:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_coverted)

    if platform.system() == 'Darwin':
        font = r"C:\Users\toong\Documents\카카오톡 받은 파일\AppleGothic.ttf"
    elif platform.system() == 'Windows':
        font = r"C:\Users\toong\Documents\카카오톡 받은 파일\GowunBatang-Regular.ttf"
    else:
        font = r"C:\Users\toong\Documents\카카오톡 받은 파일\Orbit-Regular.ttf"

    image_font = ImageFont.truetype(font, font_size)
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)

    draw.text((x, y), text, font=image_font, fill=color)

    numpy_image = np.array(image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image

def process_nutrition_info(text):
    # '%'를 찾는 정규 표현식 패턴
    pattern2 = r'(\d+)\s*%'
    percentages = re.findall(pattern2, text)
    prev_percentage = None
    # 결과를 저장할 딕셔너리
    results = {}

    # 패턴 정의: 항목, 값, 단위를 추출
    pattern = r'(?:(나트륨|탄수화물|지방|단백질)\s+([\d.]+(?:\.[\d.]+)?)\s?(mg|g))'

    matches = re.findall(pattern, text)

    for match, percentage in zip(matches, percentages):
        item_name, ratio, unit = match[0], match[1], match[2]
        if ratio == "0" and percentage != "0":
            percentages.append("0")

        results[item_name] = f"{ratio} {unit} {percentage}%"
        percentages.append(percentage)

    return results

     #깃헙